from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class LabelEncoder:
    """Bidirectional mapping between raw ids and dense integer ids.

    ``raw_to_id`` is the forward map; ``id_to_raw`` is its inverse.
    ``offset`` is the smallest assigned id (``1`` for items so ``0`` is the
    PAD slot; ``0`` for users).
    """

    raw_to_id: dict[object, int]
    id_to_raw: tuple[object, ...]
    offset: int

    @property
    def vocab_size(self) -> int:
        """Number of distinct ids (excluding the offset slots below it)."""
        return len(self.id_to_raw)

    @classmethod
    def fit(cls, values: Iterable[object], *, offset: int = 0) -> LabelEncoder:
        """Fit an encoder over the unique values.

        Sorted unique values map to consecutive ids in lexicographic order,
        so re-running on the same data gives the same mapping.
        """
        unique = sorted(set(values))
        raw_to_id = {raw: i + offset for i, raw in enumerate(unique)}
        return cls(raw_to_id=raw_to_id, id_to_raw=tuple(unique), offset=offset)

    @classmethod
    def fit_tiered(
        cls, tiers: Iterable[Iterable[object]], *, offset: int = 0
    ) -> LabelEncoder:
        """Fit so earlier tiers occupy the lowest ids, contiguously.

        Each value is assigned to the FIRST tier it appears in; within a tier
        values are sorted, so the mapping is reproducible across runs.

        Used to order item ids by the earliest split an item appears in
        (train, then val-only, then test-only). That makes each phase's *warm*
        vocabulary a contiguous prefix (``1..n_train`` for sweeps,
        ``1..n_trainval`` for final runs), so a recommender is built over the
        warm items by passing the prefix length, with no id remapping.

        Args:
            tiers: Ordered groups of raw values, most-important first.
            offset: Smallest assigned id.
        """
        ordered: list[object] = []
        seen: set[object] = set()
        for tier in tiers:
            fresh = sorted(set(tier) - seen)
            ordered.extend(fresh)
            seen.update(fresh)
        raw_to_id = {raw: i + offset for i, raw in enumerate(ordered)}
        return cls(raw_to_id=raw_to_id, id_to_raw=tuple(ordered), offset=offset)

    def transform(self, values: pd.Series | Iterable[object]) -> np.ndarray:
        """Map raw values to encoded ids."""
        if isinstance(values, pd.Series):
            return values.map(self.raw_to_id).to_numpy()
        return np.asarray([self.raw_to_id[v] for v in values])

    def to_json(self, path: Path) -> None:
        """Serialize the mapping to JSON."""
        path.write_text(
            json.dumps(
                {
                    "offset": self.offset,
                    "id_to_raw": [str(v) for v in self.id_to_raw],
                },
                indent=2,
            )
        )


__all__ = ["LabelEncoder"]
