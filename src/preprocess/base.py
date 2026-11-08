from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.preprocess.filters import FilterReport


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Summary of a preprocessing pipeline invocation.

    Attributes:
        dataset_name: Dataset key.
        train_path / val_path / test_path: Split CSVs with the canonical
            ``user_id, item_id[, timestamp]`` columns.
        meta_path: JSON sidecar with counts, filter report and split params.
        num_users: Distinct users after filtering.
        num_items: Distinct items after filtering.
        filter_report: What the sparsity filter removed.
    """

    dataset_name: str
    train_path: Path
    val_path: Path
    test_path: Path
    meta_path: Path
    num_users: int
    num_items: int
    filter_report: FilterReport


__all__ = ["PipelineResult"]
