from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

USER_COL = "user_id"
ITEM_COL = "item_id"
TIMESTAMP_COL = "timestamp"

# Items are encoded at ids 1..num_items; id 0 is the PAD slot.
PAD_IDX = 0


class MultiHotDataset(Dataset):
    """One sample per user: a multi-hot history vector and one held-out target.

    For each user one item is drawn at random (seeded) as the target and the
    remaining items form the multi-hot input of width ``num_items + 1``.

    Args:
        csv_path: Preprocessed split CSV (user_id, item_id[, timestamp]).
        num_items: Catalog size; read from the split's ``meta.json``.
        seed: RNG seed for the target draw.
        warm_items: Items the model was trained on. When given, other items
            are dropped before the target draw, and users left with fewer
            than 2 interactions are dropped.
    """

    def __init__(
        self,
        csv_path: Path,
        *,
        num_items: int,
        seed: int = 42,
        warm_items: set[int] | None = None,
    ) -> None:
        self.num_items = num_items
        self._rng = np.random.default_rng(seed)

        df = pd.read_csv(csv_path)
        if warm_items is not None:
            df = df[df[ITEM_COL].isin(warm_items)]
            df = df[df.groupby(USER_COL)[ITEM_COL].transform("size") >= 2]
            if df.empty:
                raise ValueError(
                    f"warm-item filtering removed every interaction from {csv_path}"
                )
        if TIMESTAMP_COL in df.columns:
            df = df.sort_values([USER_COL, TIMESTAMP_COL], kind="stable")
        grouped = df.groupby(USER_COL, sort=True).agg({ITEM_COL: list}).reset_index()
        self._users: list[int] = grouped[USER_COL].tolist()
        self._items: list[list[int]] = grouped[ITEM_COL].tolist()

    def __len__(self) -> int:
        return len(self._users)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        items = self._items[idx]
        target_pos = int(self._rng.integers(0, len(items)))
        input_items = items[:target_pos] + items[target_pos + 1 :]
        target = items[target_pos]

        multi_hot = torch.zeros(self.num_items + 1, dtype=torch.float32)
        multi_hot[input_items] = 1.0

        return {
            "user_id": torch.tensor(self._users[idx], dtype=torch.long),
            "history": multi_hot,
            "target": torch.tensor(target, dtype=torch.long),
        }


__all__ = ["ITEM_COL", "PAD_IDX", "MultiHotDataset"]
