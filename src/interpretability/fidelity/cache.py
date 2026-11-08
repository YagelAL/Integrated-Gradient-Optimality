from __future__ import annotations

import torch

from src.interpretability.fidelity.metric import PerturbationMode
from src.interpretability.inputs import Attributions


class AttributionCache:
    """Per-batch cache of explainer attributions.

    Static explainers (``is_dynamic=False``) are keyed by explainer alone,
    since one attribution serves every ``ke``; dynamic explainers are keyed
    by ``(explainer, ke)``. Cleared by the evaluator at the top of each batch.
    """

    def __init__(self) -> None:
        self._static: dict[str, Attributions] = {}
        self._dynamic: dict[tuple[str, int], Attributions] = {}

    def put_static(self, explainer: str, attrs: Attributions) -> None:
        self._static[explainer] = attrs

    def get_static(self, explainer: str) -> Attributions | None:
        return self._static.get(explainer)

    def put_dynamic(self, explainer: str, ke: int, attrs: Attributions) -> None:
        self._dynamic[(explainer, ke)] = attrs

    def get_dynamic(self, explainer: str, ke: int) -> Attributions | None:
        return self._dynamic.get((explainer, ke))

    def clear(self) -> None:
        self._static.clear()
        self._dynamic.clear()


class ScoreCache:
    """Per-batch cache of perturbed-recommender scores, keyed by ``(explainer, ke, mode)``.

    The perturbed scores depend only on the explainer, ``ke`` and the
    perturbation mode, so ``full_sort_predict`` runs once per triple and
    every metric reading that triple shares the cached tensor.
    """

    def __init__(self) -> None:
        self._scores: dict[tuple[str, int, PerturbationMode], torch.Tensor] = {}

    def put(
        self,
        *,
        explainer: str,
        ke: int,
        mode: PerturbationMode,
        scores: torch.Tensor,
    ) -> None:
        self._scores[(explainer, ke, mode)] = scores

    def get(
        self, *, explainer: str, ke: int, mode: PerturbationMode
    ) -> torch.Tensor:
        """Return cached scores for the triple. Raises ``KeyError`` on a miss."""
        return self._scores[(explainer, ke, mode)]

    def clear(self) -> None:
        self._scores.clear()


__all__ = ["AttributionCache", "ScoreCache"]
