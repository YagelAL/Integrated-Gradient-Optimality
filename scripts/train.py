"""Hydra entry point for training a recommender.

Composes ``configs/train.yaml`` (dataset + recommender + trainer), reads
the preprocessed split artifacts from ``splits_dir``, builds the model
and dataloaders, and runs the generic :class:`Trainer`.

Examples::

    python scripts/train.py +final=mlp_ml1m
    python scripts/train.py recommender=mlp dataset=yahoo
    python scripts/train.py recommender=mlp_deep dataset=ml1m
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import src.recommenders  # noqa: E402, F401  (side-effect imports register all models)
from src.config.schemas import (  # noqa: E402
    DatasetCfg,
    RecommenderCfg,
    TrainerCfg,
    WandbCfg,
)
from src.datasets.history_vector import ITEM_COL  # noqa: E402
from src.recommenders.factory import build_recommender  # noqa: E402
from src.seeding import seed_everything  # noqa: E402
from src.trainers.base import Trainer  # noqa: E402
from src.trainers.dataloaders import build_dataloader  # noqa: E402
from src.trainers.metrics import default_metrics  # noqa: E402
from src.trainers.wandb_utils import log_checkpoint_artifact  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print("Resolved config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    container.pop("hydra", None)

    dataset_cfg = DatasetCfg.model_validate(container["dataset"])
    recommender_cfg = RecommenderCfg.model_validate(container["recommender"])
    trainer_cfg = TrainerCfg.model_validate(container["trainer"])
    wandb_cfg = (
        WandbCfg.model_validate(container["wandb"])
        if container.get("wandb") is not None
        else None
    )

    final_run = bool(container.get("final_run", False))
    final_run_tag = str(container.get("final_run_tag", "final_optimized"))
    if final_run:
        # Disable early stopping for the final retrain (we no longer hold
        # out a val set; test is used purely for monitoring + final eval).
        trainer_cfg = trainer_cfg.model_copy(
            update={
                "early_stopping_patience": trainer_cfg.num_epochs + 1,
                "early_stopping_min_epochs": trainer_cfg.num_epochs + 1,
            }
        )
        if wandb_cfg is not None:
            wandb_cfg = wandb_cfg.model_copy(
                update={"tags": tuple({*wandb_cfg.tags, final_run_tag})}
            )

    seed_everything(trainer_cfg.seed, deterministic=trainer_cfg.deterministic)

    splits_dir = Path(container["splits_dir"])
    meta = json.loads((splits_dir / "meta.json").read_text())
    num_items = int(meta["num_items"])

    # Tiered preprocessing orders item ids so each phase's warm vocabulary is a
    # contiguous prefix: sweeps see 1..num_items_train, finals 1..num_items_trainval.
    # The prefix bounds the multi-hot width, so a cold item gets neither an
    # input dimension nor an output row.
    key = "num_items_trainval" if final_run else "num_items_train"
    model_num_items = int(meta.get(key, num_items))
    print(
        f"Loaded split meta: dataset={meta['dataset']} num_items={num_items} "
        f"model_vocab={model_num_items}"
        + ("" if model_num_items == num_items else f" (dropped {num_items - model_num_items} cold)")
    )

    model = build_recommender(recommender_cfg, num_items=model_num_items)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {recommender_cfg.name} ({n_params:,} trainable params)")

    if final_run:
        # train+val concatenated as the training corpus; test as eval/monitoring.
        train_csv: Path | list[Path] = [
            splits_dir / "train.csv",
            splits_dir / "val.csv",
        ]
        eval_csv = splits_dir / "test.csv"
        eval_phase = "test"
    else:
        train_csv = splits_dir / "train.csv"
        eval_csv = splits_dir / "val.csv"
        eval_phase = "val"

    # Popularity-weighted negative sampling (``sampled_pop_mse``): item
    # popularity = interaction count over the training split(s), aligned to
    # the multi-hot columns (item id -> column, PAD at 0). Sized to the
    # model's vocabulary; training items are warm by construction.
    pop_counts = np.zeros(model_num_items + 1, dtype=np.float32)
    pop_sources = train_csv if isinstance(train_csv, list) else [train_csv]
    for csv_path in pop_sources:
        vc = pd.read_csv(csv_path)[ITEM_COL].value_counts()
        pop_counts[vc.index.to_numpy()] += vc.to_numpy().astype(np.float32)
    model.set_popularity(torch.from_numpy(pop_counts))
    print(
        f"{recommender_cfg.name} popularity installed: "
        f"{int((pop_counts > 0).sum())}/{model_num_items} items with interactions"
    )

    train_loader = build_dataloader(
        train_csv,
        num_items=model_num_items,
        batch_size=trainer_cfg.batch_size,
        num_workers=trainer_cfg.num_workers,
        phase="train",
    )
    # Eval metrics only count items that appear in the training corpus.
    warm_items = _warm_items(train_csv)
    print(f"Warm items in training corpus: {len(warm_items)}/{model_num_items}")

    val_loader = build_dataloader(
        eval_csv,
        num_items=model_num_items,
        batch_size=trainer_cfg.batch_size,
        num_workers=trainer_cfg.num_workers,
        phase=eval_phase,
        warm_items=warm_items,
    )

    optimizer = _build_optimizer(model, trainer_cfg)
    scheduler = _build_scheduler(optimizer, trainer_cfg)

    wandb_run = _maybe_init_wandb(wandb_cfg, dataset_cfg, recommender_cfg, trainer_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=trainer_cfg,
        metrics=default_metrics(),
        device=device,
        wandb_run=wandb_run,
        primary_metric=str(container.get("primary_metric", "hr@10")),
        primary_metric_goal=str(container.get("primary_metric_goal", "maximize")),
        metric_prefix="test" if final_run else "val",
    )

    best = trainer.fit(restore_best=not final_run)
    metric_label = "test" if final_run else "validation"
    print(f"Best {metric_label}: {best}")

    ckpt_dir = Path(container["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = (
        f"final_{recommender_cfg.name}.pt" if final_run else f"best_{recommender_cfg.name}.pt"
    )
    ckpt_path = ckpt_dir / ckpt_name
    trainer.save_checkpoint(ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    if wandb_run is not None and wandb_cfg is not None:
        metric_prefix = "test" if final_run else "val"
        aliases = (
            (wandb_cfg.artifact_alias, "final") if final_run else (wandb_cfg.artifact_alias,)
        )
        log_checkpoint_artifact(
            wandb_run,
            ckpt_path,
            name=f"model-{wandb_run.id}",
            aliases=aliases,
            metadata={
                "dataset": dataset_cfg.name,
                "recommender": recommender_cfg.name,
                "recommender_cfg": recommender_cfg.model_dump(mode="json"),
                "final_run": final_run,
                "num_items": model_num_items,  # warm items
                **{f"{metric_prefix}/{k}": v for k, v in best.items()},
            },
        )
        wandb_run.finish()


def _warm_items(train_csv: Path | list[Path]) -> set[int]:
    """Item ids present in the training corpus (the only ones that get gradient)."""
    paths = train_csv if isinstance(train_csv, list) else [train_csv]
    warm: set[int] = set()
    for p in paths:
        warm |= set(pd.read_csv(p, usecols=["item_id"])["item_id"].unique())
    return warm


def _build_optimizer(model, cfg: TrainerCfg):
    params = (p for p in model.parameters() if p.requires_grad)
    if cfg.optimizer.name == "adam":
        return torch.optim.Adam(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    return torch.optim.AdamW(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)


def _build_scheduler(optimizer, cfg: TrainerCfg):
    if cfg.scheduler.name == "none":
        return None
    if cfg.scheduler.name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.scheduler.decay_epochs, gamma=cfg.scheduler.decay_gamma
        )
    if cfg.scheduler.name == "cosine_warmup":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    raise ValueError(f"unknown scheduler: {cfg.scheduler.name!r}")


def _maybe_init_wandb(
    wandb_cfg: WandbCfg | None,
    dataset_cfg: DatasetCfg,
    recommender_cfg: RecommenderCfg,
    trainer_cfg: TrainerCfg,
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
            "trainer": trainer_cfg.model_dump(mode="json"),
        },
    )


if __name__ == "__main__":
    main()
