from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from src.interpretability.fidelity.metric import PerturbationMode


@dataclass(frozen=True)
class SliceGroup:
    """Identifier for a group whose ``ke`` axis we monotonize together.

    Everything *except* ``ke`` — that is the axis the monotonic post-pass
    reduces over. One :class:`PerUserRecord` belongs to exactly one
    ``(SliceGroup, ke, user_id)``.

    Attributes:
        metric: Registered metric name (e.g. ``"del_ke"``).
        explainer: Explainer name (e.g. ``"ig"``).
        item_rank: Index in the ``item_ranks`` config (0 = top-1).
        mode: Perturbation mode.
        kr: Recommendation-cutoff cap. ``None`` for metrics with
            ``depends_on_kr=False``.
    """

    metric: str
    explainer: str
    item_rank: int
    mode: PerturbationMode
    kr: int | None


@dataclass(frozen=True)
class PerUserRecord:
    """One per-user value emitted by a metric for one ``(slice, ke, user)``.

    The evaluator captures these from every ``metric.update`` call when
    monotonic correction or artifact dumps are enabled. Records are
    gathered to rank 0 via ``dist.gather_object`` (in-memory, no
    ``/tmp``); only records from ``supports_monotonic=True`` metrics are
    fed to :func:`monotonic_aggregate` — the rest keep their raw
    streaming mean as the headline (see ``FidelityEvaluator.evaluate``).

    Attributes:
        group: The :class:`SliceGroup` this record belongs to.
        ke: Explanation budget at which the value was computed.
        user_id: Long-valued user identifier from the dataloader batch.
        value: The per-user scalar: a target-probability ratio vs. the
            original prediction (DEL@Ke, INS@Ke).
    """

    group: SliceGroup
    ke: int
    user_id: int
    value: float


def slice_key(group: SliceGroup, ke: int) -> str:
    """Render a :class:`SliceGroup` + ``ke`` into the evaluator's scalar key.

    Format matches :meth:`FidelityEvaluator._slice_key` exactly so the
    monotonic post-pass overwrites / extends the same W&B namespace the
    streaming raw scalars use.
    """
    suffix = f"/kr={group.kr}" if group.kr is not None else ""
    return (
        f"{group.explainer}/{group.metric}/rank={group.item_rank}"
        f"/mode={group.mode.value}/ke={ke}{suffix}"
    )


def monotonic_per_user_values(
    records: Sequence[PerUserRecord],
    *,
    ke_grid: Sequence[int],
    reduction: Literal["cummin", "cummax"] = "cummin",
) -> dict[tuple[SliceGroup, int, int], float]:
    """Per-user, per-(group, ke) monotonic-corrected value table.

    Same algorithm as :func:`monotonic_aggregate` but returns the
    full per-user matrix instead of the slice means. Used by
    :func:`write_results_csv` to populate the ``monotonic_value`` column
    of the per-user details dump.

    Args:
        records: Flat list of per-user records.
        ke_grid: ``ke`` values sorted ascending (matrix columns align here).
        reduction: ``"cummin"`` or ``"cummax"``.

    Returns:
        Mapping ``(group, ke, user_id) -> monotonic_value``. Keys cover
        every ``(group, ke, user_id)`` triple present in ``records``;
        carry-forward fills any ``(user, ke)`` hole.
    """
    sorted_ke = sorted(ke_grid)
    ke_index = {ke: i for i, ke in enumerate(sorted_ke)}

    grouped: dict[SliceGroup, list[PerUserRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.group, []).append(rec)

    out: dict[tuple[SliceGroup, int, int], float] = {}
    for group, recs in grouped.items():
        per_user: dict[int, dict[int, float]] = {}
        for rec in recs:
            per_user.setdefault(rec.user_id, {})[rec.ke] = rec.value
        if not per_user:
            continue
        user_ids = list(per_user.keys())
        matrix = torch.full(
            (len(user_ids), len(sorted_ke)), float("nan"), dtype=torch.float64
        )
        for row, uid in enumerate(user_ids):
            for ke, val in per_user[uid].items():
                if ke in ke_index:
                    matrix[row, ke_index[ke]] = val

        # Defensive carry-forward (NaN -> prior cell on the same row).
        for row in range(matrix.size(0)):
            last = float("nan")
            for col in range(matrix.size(1)):
                v = float(matrix[row, col].item())
                if v != v:
                    matrix[row, col] = last
                else:
                    last = v

        if reduction == "cummin":
            corrected = matrix.cummin(dim=1).values
        elif reduction == "cummax":
            corrected = matrix.cummax(dim=1).values
        else:
            raise ValueError(f"unknown monotonic_reduction: {reduction!r}")

        for row, uid in enumerate(user_ids):
            for col, ke in enumerate(sorted_ke):
                out[(group, ke, uid)] = float(corrected[row, col].item())
    return out


def monotonic_aggregate(
    records: Sequence[PerUserRecord],
    *,
    ke_grid: Sequence[int],
    reduction: Literal["cummin", "cummax"] = "cummin",
    raw_suffix: str = "_raw",
) -> dict[str, float]:
    """Post-pass: cummin (or cummax) per user along ``ke``, then mean.

    Algorithm per :class:`SliceGroup`:

    1. Build a ``{user_id: {ke: value}}`` map from the input records.
    2. Materialise a ``[N_users × len(ke_grid)]`` matrix sorted by ``ke``
       ascending. Missing ``(user, ke)`` cells are filled with the user's
       most recent earlier value (carry-forward). With ``shuffle=False``
       sampling and one record per ``(user, group, ke)`` this never
       triggers; the fill is defensive.
    3. ``torch.cummin`` (or ``torch.cummax``) along the ``ke`` axis per
       row. Standard semantics: the value at column ``k`` is the running
       min / max up to and including ``k``.
    4. Mean across users → one scalar per ``ke``.

    Both the monotonic-corrected and the raw streaming scalars are
    returned (raw under the ``raw_suffix``-suffixed key) so the caller
    can log both with a single dict merge.

    Args:
        records: Flat list of per-user records, typically the
            rank-0-gathered concatenation of every rank's local buffer.
        ke_grid: The ``ke`` values the evaluator iterated, sorted
            ascending. Used to align the matrix columns.
        reduction: ``"cummin"`` or ``"cummax"``, per the metric's own
            ``monotonic_reduction``. Lower-is-better metrics use cummin;
            higher-is-better metrics (e.g. INS@Ke) use cummax. Callers
            with a mix of both must group records by reduction and call
            this once per group — one call takes only one reduction.
        raw_suffix: Suffix appended to the slice key for the raw mean.

    Returns:
        Flat ``{slice_key: scalar}`` mapping. Each ``SliceGroup × ke``
        yields one monotonic-corrected key and one raw-suffixed key.
    """
    sorted_ke = sorted(ke_grid)
    ke_index = {ke: i for i, ke in enumerate(sorted_ke)}

    # Group records by SliceGroup.
    grouped: dict[SliceGroup, list[PerUserRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.group, []).append(rec)

    out: dict[str, float] = {}
    for group, recs in grouped.items():
        # Build per-user dense rows aligned to ke_grid.
        per_user: dict[int, dict[int, float]] = {}
        for rec in recs:
            per_user.setdefault(rec.user_id, {})[rec.ke] = rec.value

        n_users = len(per_user)
        n_ke = len(sorted_ke)
        if n_users == 0 or n_ke == 0:
            continue

        matrix = torch.full((n_users, n_ke), float("nan"), dtype=torch.float64)
        for row, (_uid, ke_to_val) in enumerate(per_user.items()):
            for ke, val in ke_to_val.items():
                if ke in ke_index:
                    matrix[row, ke_index[ke]] = val

        # Defensive carry-forward for any holes; should be a no-op in normal use.
        for row in range(n_users):
            last = float("nan")
            for col in range(n_ke):
                v = float(matrix[row, col].item())
                if v != v:  # NaN check
                    matrix[row, col] = last
                else:
                    last = v

        raw_per_ke = matrix.mean(dim=0)  # [n_ke]
        if reduction == "cummin":
            monotonic_matrix = matrix.cummin(dim=1).values
        elif reduction == "cummax":
            monotonic_matrix = matrix.cummax(dim=1).values
        else:
            raise ValueError(f"unknown monotonic_reduction: {reduction!r}")
        monotonic_per_ke = monotonic_matrix.mean(dim=0)  # [n_ke]

        for col, ke in enumerate(sorted_ke):
            key = slice_key(group, ke)
            out[key] = float(monotonic_per_ke[col].item())
            out[key + raw_suffix] = float(raw_per_ke[col].item())

    return out


__all__ = [
    "PerUserRecord",
    "SliceGroup",
    "monotonic_aggregate",
    "monotonic_per_user_values",
    "slice_key",
]
