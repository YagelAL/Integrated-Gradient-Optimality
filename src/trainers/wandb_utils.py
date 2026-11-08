"""W&B checkpoint persistence — log and load recommender state-dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.exceptions import ArtifactNotFound


def log_checkpoint_artifact(
    run: Any,
    ckpt_path: Path,
    *,
    name: str,
    aliases: tuple[str, ...] = ("best",),
    metadata: dict | None = None,
) -> None:
    """Log a checkpoint .pt file as a W&B model artifact.

    Convention: callers name the artifact model-{run_id} so its path
    resolves to {entity}/{project}/model-{run_id}:{alias}.

    Args:
        run: Live wandb.Run instance (kept untyped so this module does
            not require wandb at import time for downstream callers).
        ckpt_path: Filesystem path to the .pt file to upload.
        name: Artifact name (typically f"model-{run.id}").
        aliases: Aliases to attach to the artifact version.
        metadata: Optional JSON-serializable metadata dict.
    """
    import wandb  # noqa: PLC0415

    artifact = wandb.Artifact(name=name, type="model", metadata=metadata or {})
    artifact.add_file(str(ckpt_path))
    run.log_artifact(artifact, aliases=list(aliases))


def infer_num_items(state: dict[str, Any]) -> int | None:
    """Recover the item vocabulary a checkpoint was built with.

    History-vector models infer it from the widest Linear weight's
    ``in_features`` (``num_items + 1``, including PAD at 0).

    Returns:
        The vocabulary size, or ``None`` when the state dict has no 2-D
        weight (the caller then falls back to meta.json).
    """
    in_dims = [
        int(v.shape[1])
        for v in state.values()
        if hasattr(v, "shape") and len(v.shape) == 2
    ]
    return (max(in_dims) - 1) if in_dims else None


def load_recommender_from_artifact(
    artifact_path: str,
    *,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load both the state-dict and the architecture config for an artifact.

    A checkpoint's weights are only loadable into a model built with the
    exact architecture they were trained under (hidden_dim, num_layers,
    ...). That architecture is recovered here rather than re-derived from
    static YAML defaults, which may differ from the sweep-selected values
    the checkpoint actually used.

    Recovery order for the recommender config:

    1. ``artifact.metadata["recommender_cfg"]`` — self-contained, written
       by :func:`log_checkpoint_artifact`. Preferred: survives even if the
       producing run is deleted.
    2. ``artifact.logged_by().config["recommender"]`` — the producing run's
       logged config.

    Args:
        artifact_path: {entity}/{project}/model-{run_id}:{alias} form.
        device: Device onto which the loaded tensors are mapped.

    Returns:
        ``(state_dict, recommender_cfg)`` where ``recommender_cfg`` is the
        raw arch dict, ready for ``RecommenderCfg.model_validate``.

    Raises:
        ArtifactNotFound: If the artifact contains no .pt file, or the
            recommender config cannot be recovered from either source.
    """
    import wandb  # noqa: PLC0415

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")
    ckpt_dir = Path(artifact.download())
    ckpt_files = list(ckpt_dir.glob("*.pt"))
    if not ckpt_files:
        raise ArtifactNotFound(artifact_path, "no .pt files inside the artifact")
    state = torch.load(ckpt_files[0], map_location=device, weights_only=False)

    recommender_cfg = artifact.metadata.get("recommender_cfg")
    if not isinstance(recommender_cfg, dict):
        run = artifact.logged_by()
        run_cfg = dict(run.config) if run is not None else {}
        recommender_cfg = run_cfg.get("recommender")
    if not isinstance(recommender_cfg, dict):
        raise ArtifactNotFound(
            artifact_path,
            "no recommender architecture config found in artifact metadata "
            "or producing-run config; cannot rebuild the model to match the "
            "checkpoint weights",
        )
    return state, recommender_cfg


__all__ = [
    "infer_num_items",
    "load_recommender_from_artifact",
    "log_checkpoint_artifact",
]
