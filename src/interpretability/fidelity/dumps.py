from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from src.interpretability.fidelity.monotonic import PerUserRecord, SliceGroup


def write_results_csv(
    records: Iterable[PerUserRecord],
    *,
    path: Path,
    monotonic_values: dict[tuple[SliceGroup, int, int], float] | None = None,
) -> Path:
    """Write per-user fidelity records to a CSV.

    Schema (one row per ``(user_id, metric, explainer, item_rank, mode, kr, ke)``):

    ``user_id, explainer, item_rank, mode, ke, kr, metric, raw_value, monotonic_value``

    The monotonic correction is embedded, so no separate recalculation
    step is needed.

    Args:
        records: Flat list of per-user records (already gathered to
            rank 0 in the DDP case).
        path: Destination CSV path. Parent directories are created.
        monotonic_values: Optional ``{(group, ke, user_id): value}``
            map produced alongside the streaming records. When
            provided, the ``monotonic_value`` column is populated;
            otherwise it is left empty (raw-only dump).

    Returns:
        The CSV path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "user_id",
        "explainer",
        "item_rank",
        "mode",
        "ke",
        "kr",
        "metric",
        "raw_value",
        "monotonic_value",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            monotonic = (
                monotonic_values.get((rec.group, rec.ke, rec.user_id))
                if monotonic_values is not None
                else None
            )
            writer.writerow(
                {
                    "user_id": rec.user_id,
                    "explainer": rec.group.explainer,
                    "item_rank": rec.group.item_rank,
                    "mode": rec.group.mode.value,
                    "ke": rec.ke,
                    "kr": "" if rec.group.kr is None else rec.group.kr,
                    "metric": rec.group.metric,
                    "raw_value": rec.value,
                    "monotonic_value": "" if monotonic is None else monotonic,
                }
            )
    return path


__all__ = ["write_results_csv"]
