"""Hydra entry point for explainer fidelity evaluation.

Composes ``configs/explainer_eval.yaml`` (dataset + recommender +
fidelity + budget catalog + artifact source), loads a frozen recommender
checkpoint (local ``.pt`` or W&B artifact), and runs the single-pass
:class:`FidelityEvaluator` over the test set.

Run as a single process with the default eval.batch_size and eval.num_workers:
the held-out test item depends on them.

Example::

    python scripts/run_explainer_eval.py +final=mlp_ml1m budget=q1000
"""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hydra  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import src.interpretability  # noqa: E402, F401  (side-effect imports register explainers + metrics)
import src.recommenders  # noqa: E402, F401  (side-effect imports register recommenders)
from src.config.schemas import (  # noqa: E402
    DatasetCfg,
    FidelityCfg,
    RecommenderArtifactCfg,
    RecommenderCfg,
    WandbCfg,
)
from src.interpretability.fidelity.evaluator import (  # noqa: E402
    DumpSpec,
    FidelityEvaluator,
    MetricSpec,
)
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from src.interpretability.registry import build_explainer  # noqa: E402
from src.recommenders.factory import build_recommender  # noqa: E402
from src.trainers.dataloaders import build_dataloader  # noqa: E402
from src.trainers.wandb_utils import (  # noqa: E402
    infer_num_items,
    load_recommender_from_artifact,
)


@hydra.main(version_base=None, config_path="../configs", config_name="explainer_eval")
def main(cfg: DictConfig) -> None:
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    container.pop("hydra", None)

    dataset_cfg = DatasetCfg.model_validate(container["dataset"])
    recommender_cfg = RecommenderCfg.model_validate(container["recommender"])
    fidelity_cfg = FidelityCfg.model_validate(container["fidelity"])
    artifact_cfg = RecommenderArtifactCfg.model_validate(container["recommender_artifact"])
    wandb_cfg = (
        WandbCfg.model_validate(container["wandb"])
        if container.get("wandb") is not None
        else None
    )

    # DDP init if launched under torchrun.
    rank, world_size, local_rank = _maybe_init_ddp()
    is_rank0 = rank == 0

    if is_rank0:
        print("Resolved config:")
        print(OmegaConf.to_yaml(cfg, resolve=True))

    splits_dir = Path(container["splits_dir"])
    meta = json.loads((splits_dir / "meta.json").read_text())
    num_items = int(meta.get("num_items_trainval", meta["num_items"]))
    if is_rank0:
        print(f"Loaded split meta: dataset={meta['dataset']} num_items={num_items}")

    device = _pick_device(local_rank)

    # Build the recommender, load weights, freeze. When loading from W&B,
    # recover the architecture the checkpoint was trained under (e.g. the
    # sweep-selected hidden_dim), otherwise load_state_dict fails on shape
    # mismatch. None-valued keys are dropped so configs logged with extra
    # optional fields still validate.
    state, recovered_cfg = _load_state_dict(artifact_cfg, device=device)
    if recovered_cfg is not None:
        recommender_cfg = RecommenderCfg.model_validate(
            {k: v for k, v in recovered_cfg.items() if v is not None}
        )
        if is_rank0:
            print(f"Recovered recommender arch from artifact: {recommender_cfg.model_dump()}")

    ckpt_num_items = infer_num_items(state)
    if ckpt_num_items is not None and ckpt_num_items != num_items:
        if is_rank0:
            print(f"Checkpoint vocabulary is {ckpt_num_items} (meta said {num_items}); using it")
        num_items = ckpt_num_items
    warm_items = set(range(1, num_items + 1))

    model = build_recommender(recommender_cfg, num_items=num_items)
    model.load_state_dict(state)
    model.to(device)
    model.freeze()
    if is_rank0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Recommender {recommender_cfg.name} loaded ({n_params:,} params, frozen)")

    test_loader = build_dataloader(
        splits_dir / "test.csv",
        num_items=num_items,
        batch_size=int(container["eval"]["batch_size"]),
        num_workers=int(container["eval"]["num_workers"]),
        phase="test",
        distributed=world_size > 1,
        warm_items=warm_items,
    )

    # active_explainer: comma-separated all_explainers keys, run against one
    # model load in the given order.
    if container.get("active_explainer") and container.get("all_explainers"):
        names = [n.strip() for n in str(container["active_explainer"]).split(",") if n.strip()]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"active_explainer lists the same entry more than once: {dup}")
        container["explainers"] = [container["all_explainers"][n] for n in names]

    explainers = []
    for entry in container["explainers"]:
        params = dict(entry.get("params", {}))
        explainer = build_explainer(entry["name"], model, params=params, device=device)
        # Optional display alias: lets one registry method appear several times
        # in a run (e.g. pls under different selection objectives). All
        # evaluator caches and scalar keys go through explainer.name.
        if entry.get("alias"):
            explainer.name = str(entry["alias"])
        explainers.append(explainer)
    # Every evaluator structure (slices, caches, dumps, scalar keys) is a dict
    # keyed by explainer.name, so a repeated name would silently overwrite the
    # earlier explainer's results under the later one's label.
    resolved = [e.name for e in explainers]
    dup = sorted({n for n in resolved if resolved.count(n) > 1})
    if dup:
        raise ValueError(
            f"duplicate explainer name(s) {dup} after alias resolution; give each "
            "repeated method a distinct 'alias:' in the budget config"
        )
    metric_specs = [
        MetricSpec(name=m.name, params=dict(m.params)) for m in fidelity_cfg.metrics
    ]

    # The Hydra run dir anchors the opt-in per-user CSV dump.
    hydra_run_dir = Path(HydraConfig.get().run.dir)
    dump_spec = DumpSpec(
        save_artifacts=fidelity_cfg.save_artifacts,
        out_dir=hydra_run_dir,
        dataset=dataset_cfg.name,
        recommender=recommender_cfg.name,
    )

    evaluator = FidelityEvaluator(
        recommender=model,
        explainers=explainers,
        metric_specs=metric_specs,
        family=recommender_cfg.family,
        ke_grid=list(fidelity_cfg.ke_grid),
        kr_grid=list(fidelity_cfg.kr_grid),
        item_ranks=list(fidelity_cfg.item_ranks),
        device=device,
        monotonic=fidelity_cfg.monotonic,
        dump=dump_spec,
    )

    wandb_run = _maybe_init_wandb(
        wandb_cfg if is_rank0 else None,
        dataset_cfg,
        recommender_cfg,
        fidelity_cfg,
        explainers_cfg=list(container["explainers"]),
    )

    result = evaluator.evaluate(test_loader)

    if is_rank0:
        print(
            f"Fidelity eval complete: {len(result.scalars)} slices, "
            f"recommender_call_count={result.recommender_call_count}"
        )
        for k, v in sorted(result.scalars.items()):
            print(f"  {k}: {v:.4f}")
        if result.dump_paths:
            print("Dumps written:")
            for p in result.dump_paths:
                print(f"  {p}")
        if wandb_run is not None:
            wandb_run.log(result.scalars)
            wandb_run.finish()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _maybe_init_ddp() -> tuple[int, int, int]:
    """Initialise DDP from ``torchrun`` env vars when present.

    Returns:
        ``(rank, world_size, local_rank)``. When not launched under
        ``torchrun``, returns ``(0, 1, 0)`` and does not init the process
        group.
    """
    if "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size == 1:
        return 0, 1, 0
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    # Per-user explainer runtime varies with history length, so DDP ranks can
    # drift apart; raise the default 10-minute collective timeout.
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=600))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _pick_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _load_state_dict(
    cfg: RecommenderArtifactCfg, *, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict | None]:
    """Load checkpoint weights and, for W&B artifacts, the arch config.

    Returns:
        ``(state_dict, recommender_cfg)``. ``recommender_cfg`` is the raw
        arch dict recovered from the artifact when loading from W&B, or
        ``None`` for a local ``.pt`` (no config travels with a bare file —
        the YAML-composed config is used as-is).
    """
    if cfg.wandb_path is not None:
        return load_recommender_from_artifact(cfg.wandb_path, device=device)
    assert cfg.local_path is not None
    return torch.load(cfg.local_path, map_location=device, weights_only=False), None


def _maybe_init_wandb(
    wandb_cfg: WandbCfg | None,
    dataset_cfg: DatasetCfg,
    recommender_cfg: RecommenderCfg,
    fidelity_cfg: FidelityCfg,
    explainers_cfg: list[dict],
):
    if wandb_cfg is None:
        return None
    import wandb  # noqa: PLC0415

    return wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        tags=list(wandb_cfg.tags),
        config={
            "dataset": dataset_cfg.model_dump(mode="json"),
            "recommender": recommender_cfg.model_dump(mode="json"),
            "fidelity": fidelity_cfg.model_dump(mode="json"),
            "explainers": explainers_cfg,
        },
        job_type="explainer_eval",
    )


if __name__ == "__main__":
    main()
