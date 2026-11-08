from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config.schemas import RandomFractionSplit


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Train / val / test frames plus split bookkeeping written to ``meta.json``."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    meta: dict[str, Any]


class RandomUserSplitter:
    """Random holdout at the *user* level.

    Disjoint user sets: ``test_fraction`` of users -> test, ``val_fraction``
    -> val, rest -> train. Each test/val user carries their full history.
    """

    def __init__(
        self,
        *,
        user_col: str,
        test_fraction: float,
        val_fraction: float,
        seed: int,
    ) -> None:
        self.user_col = user_col
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction
        self.seed = seed

    def split(self, df: pd.DataFrame) -> SplitResult:
        """Produce the random user-level split."""
        rng = np.random.default_rng(self.seed)
        users = np.array(df[self.user_col].unique())
        rng.shuffle(users)

        n = len(users)
        n_test = int(round(n * self.test_fraction))
        n_val = int(round(n * self.val_fraction))
        test_users = set(users[:n_test])
        val_users = set(users[n_test : n_test + n_val])

        is_test = df[self.user_col].isin(test_users)
        is_val = df[self.user_col].isin(val_users)
        test = df[is_test]
        val = df[is_val]
        train = df[~(is_test | is_val)]

        return SplitResult(
            train=train.reset_index(drop=True),
            val=val.reset_index(drop=True),
            test=test.reset_index(drop=True),
            meta={
                "strategy": "random_fraction",
                "test_fraction": self.test_fraction,
                "val_fraction": self.val_fraction,
                "seed": self.seed,
                "num_train_users": int(n - n_test - n_val),
                "num_val_users": n_val,
                "num_test_users": n_test,
            },
        )


def build_splitter(cfg: RandomFractionSplit, *, user_col: str) -> RandomUserSplitter:
    """Build the splitter from a typed split config."""
    return RandomUserSplitter(
        user_col=user_col,
        test_fraction=cfg.test_fraction,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )


__all__ = ["RandomUserSplitter", "SplitResult", "build_splitter"]
