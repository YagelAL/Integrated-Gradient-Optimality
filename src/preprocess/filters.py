from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.schemas import FilterCfg


@dataclass(frozen=True, slots=True)
class FilterReport:
    """Summary of a single :class:`IterativeNCoreFilter` invocation."""

    iterations: int
    rows_before: int
    rows_after: int
    users_before: int
    users_after: int
    items_before: int
    items_after: int


class IterativeNCoreFilter:
    """≥k users-per-item and ≥k items-per-user, iterated to convergence.

    Optionally drops *consecutive* same-item repeats per user (useful for
    session datasets where double-clicks are noise). Operates on a copy —
    the input DataFrame is not mutated.

    Args:
        cfg: Validated :class:`FilterCfg` (min counts + repeat-drop flag).
        user_col: Column name in the input DataFrame holding user ids.
        item_col: Column name in the input DataFrame holding item ids.
        timestamp_col: Column name for timestamps. Required when
            ``cfg.drop_consecutive_repeats`` is ``True`` (we need to order
            within each user before dropping). Optional otherwise.
    """

    def __init__(
        self,
        cfg: FilterCfg,
        *,
        user_col: str,
        item_col: str,
        timestamp_col: str | None = None,
    ) -> None:
        if cfg.drop_consecutive_repeats and timestamp_col is None:
            raise ValueError(
                "drop_consecutive_repeats requires timestamp_col for ordering"
            )
        self._cfg = cfg
        self._user_col = user_col
        self._item_col = item_col
        self._timestamp_col = timestamp_col

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, FilterReport]:
        """Return the filtered DataFrame and a :class:`FilterReport`."""
        rows_before = len(df)
        users_before = df[self._user_col].nunique()
        items_before = df[self._item_col].nunique()

        out = df
        if self._cfg.drop_consecutive_repeats:
            out = self._drop_consecutive_repeats(out)

        # Removing users can drop items below the per-item threshold and vice
        # versa, so iterate until the table is stable.
        iterations = 0
        while True:
            iterations += 1
            user_counts = out.groupby(self._user_col).size()
            item_counts = out.groupby(self._item_col).size()
            keep_users = user_counts.index[user_counts >= self._cfg.min_items_per_user]
            keep_items = item_counts.index[item_counts >= self._cfg.min_users_per_item]

            before = len(out)
            out = out[
                out[self._user_col].isin(keep_users)
                & out[self._item_col].isin(keep_items)
            ]
            if len(out) == before:
                break

        return out.reset_index(drop=True), FilterReport(
            iterations=iterations,
            rows_before=rows_before,
            rows_after=len(out),
            users_before=users_before,
            users_after=out[self._user_col].nunique(),
            items_before=items_before,
            items_after=out[self._item_col].nunique(),
        )

    def _drop_consecutive_repeats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Within each user (ordered by timestamp), drop rows where the item
        equals the previous row's item."""
        assert self._timestamp_col is not None
        sorted_df = df.sort_values([self._user_col, self._timestamp_col], kind="stable")
        prev_item = sorted_df.groupby(self._user_col)[self._item_col].shift()
        return sorted_df[sorted_df[self._item_col] != prev_item]


__all__ = ["FilterReport", "IterativeNCoreFilter"]
