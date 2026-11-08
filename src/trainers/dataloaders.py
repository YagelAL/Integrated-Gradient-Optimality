"""DataLoader factory for the history-vector recommenders."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pandas as pd
from torch.utils.data import DataLoader, DistributedSampler

from src.datasets.history_vector import MultiHotDataset

Phase = Literal["train", "val", "test"]


def build_dataloader(
    csv_path: Path | Sequence[Path],
    *,
    num_items: int,
    batch_size: int,
    num_workers: int,
    phase: Phase,
    distributed: bool = False,
    warm_items: set[int] | None = None,
) -> DataLoader:
    """Build a phase-appropriate dataloader.

    Args:
        csv_path: Path to train.csv / val.csv / test.csv, or a sequence of
            paths (e.g. [train.csv, val.csv]) that are concatenated row-wise.
        num_items: Catalog size (multi-hot width minus the PAD slot).
        batch_size: DataLoader batch size.
        num_workers: Number of worker processes for data loading.
        phase: train shuffles; val/test do not.
        distributed: Wrap the dataset in a DistributedSampler so each DDP
            rank reads a disjoint shard.
        warm_items: Item ids seen in training; other items are dropped
            from the split. ``None`` keeps every item.
    """
    shuffle = phase == "train"
    ds = MultiHotDataset(
        _resolve_csv_path(csv_path),
        num_items=num_items,
        warm_items=warm_items,
    )
    sampler = (
        DistributedSampler(ds, shuffle=shuffle, drop_last=False)
        if distributed
        else None
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def _resolve_csv_path(csv_path: Path | Sequence[Path]) -> Path:
    """Return a single CSV path, concatenating if multiple were supplied.

    Several paths are concatenated row-wise (exact-duplicate rows dropped)
    into a temp file whose path is returned.
    """
    if isinstance(csv_path, Path):
        return csv_path
    paths = list(csv_path)
    if len(paths) == 1:
        return paths[0]
    frames = [pd.read_csv(p) for p in paths]
    combined = pd.concat(frames, axis=0, ignore_index=True).drop_duplicates()
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="combined_"
    )
    combined.to_csv(tmp.name, index=False)
    tmp.close()
    return Path(tmp.name)


__all__ = ["Phase", "build_dataloader"]
