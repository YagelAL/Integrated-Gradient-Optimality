"""Model/context setup for `synth_gt_audit.py`: config validation, checkpoint
loading + vocabulary inference, model build + load + freeze."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.config.schemas import DatasetCfg, RecommenderArtifactCfg, RecommenderCfg
from src.recommenders.factory import build_recommender
from src.trainers.wandb_utils import infer_num_items


def load_model_and_context(
    container: dict,
) -> tuple[DatasetCfg, RecommenderCfg, RecommenderArtifactCfg, torch.device, Path, int, set[int], object]:
    """Validate configs, load the checkpoint, build+load+freeze the recommender.

    Returns (dataset_cfg, recommender_cfg, artifact_cfg, device, splits_dir,
    num_items, warm_items, model).
    """
    dataset_cfg = DatasetCfg.model_validate(container["dataset"])
    recommender_cfg = RecommenderCfg.model_validate(container["recommender"])
    artifact_cfg = RecommenderArtifactCfg.model_validate(container["recommender_artifact"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits_dir = Path(container["splits_dir"])
    meta = json.loads((splits_dir / "meta.json").read_text())
    num_items = int(meta.get("num_items_trainval", meta["num_items"]))

    assert artifact_cfg.local_path is not None, "pass recommender_artifact.local_path=<ckpt>.pt"
    state = torch.load(artifact_cfg.local_path, map_location=device, weights_only=False)
    ckpt_num_items = infer_num_items(state)
    if ckpt_num_items is not None and ckpt_num_items != num_items:
        print(f"Checkpoint vocabulary is {ckpt_num_items} (meta said {num_items}); using it")
        num_items = ckpt_num_items
    warm_items = set(range(1, num_items + 1))

    model = build_recommender(recommender_cfg, num_items=num_items)
    model.load_state_dict(state)
    model.to(device)
    model.freeze()
    print(f"Recommender {recommender_cfg.name} loaded, score_activation={model.score_activation}")

    return dataset_cfg, recommender_cfg, artifact_cfg, device, splits_dir, num_items, warm_items, model
