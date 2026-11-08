"""On-disk paths of the preprocessed split artifacts.

Layout under ``output_root``::

    {output_root}/splits/{dataset_name}/{strategy}/{params_dir}/
        train.csv | val.csv | test.csv     (user_id, item_id[, timestamp][, rating])
        item_encoder.json | user_encoder.json
        meta.json                          (counts, filter report, split params)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.schemas import RandomFractionSplit


def _split_params_dir(split: RandomFractionSplit) -> str:
    """Stable directory name encoding the split parameters."""
    return (
        f"test{split.test_fraction:.2f}"
        f"_val{split.val_fraction:.2f}"
        f"_seed{split.seed}"
    )


@dataclass(frozen=True, slots=True)
class SplitPaths:
    """Resolved on-disk locations for one ``(dataset, split)`` pair."""

    output_root: Path
    dataset_name: str
    split: RandomFractionSplit

    @property
    def split_dir(self) -> Path:
        return (
            self.output_root
            / "splits"
            / self.dataset_name
            / self.split.strategy
            / _split_params_dir(self.split)
        )

    @property
    def train_path(self) -> Path:
        return self.split_dir / "train.csv"

    @property
    def val_path(self) -> Path:
        return self.split_dir / "val.csv"

    @property
    def test_path(self) -> Path:
        return self.split_dir / "test.csv"

    @property
    def meta_path(self) -> Path:
        return self.split_dir / "meta.json"

    def ensure_dirs(self) -> None:
        """Create the split output directory."""
        self.split_dir.mkdir(parents=True, exist_ok=True)


__all__ = ["SplitPaths"]
