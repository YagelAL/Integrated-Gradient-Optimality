from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace as dc_replace
from typing import ClassVar

import torch

from src.datasets.history_vector import PAD_IDX
from src.interpretability.fidelity.metric import target_probability
from src.interpretability.fidelity.perturbations import (
    RetainPerturbation,
    TruncatePerturbation,
)
from src.interpretability.inputs import Attributions, ExplainInput
from src.recommenders.base import BaseRecommender


class UnknownSelectionObjective(KeyError):
    """Raised when a selection-objective name is not in the registry."""

    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(name)
        self.name = name
        self.available = available

    def __str__(self) -> str:
        return (
            f"Unknown selection objective {self.name!r}. "
            f"Available: {sorted(self.available)}"
        )


class SelectionObjective(ABC):
    """Criterion the PLS subset search optimizes over candidate sets.

    Attributes:
        name: Registry key (``del`` / ``ins`` / ``del_ke`` / ``ins_ke``).
        needs_truncate: Whether scoring reads the TRUNCATE probe.
        needs_retain: Whether scoring reads the RETAIN probe.
        readout: Which quantity a probe returns: ``"cdcg"`` (from the
            target's rank) or ``"prob"`` (the target's probability, what
            del_ke / ins_ke grade). Same perturbation either way.
    """

    name: ClassVar[str]
    needs_truncate: ClassVar[bool]
    needs_retain: ClassVar[bool]
    readout: ClassVar[str] = "cdcg"

    @abstractmethod
    def score(
        self,
        cdcg_trunc: torch.Tensor | None,
        cdcg_retain: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-candidate value, higher = better. Unneeded probes arrive as None."""


class DeletionObjective(SelectionObjective):
    """Necessity by rank: the explained item should fall furthest when removed."""

    name = "del"
    needs_truncate = True
    needs_retain = False

    def score(self, cdcg_trunc, cdcg_retain):
        return -cdcg_trunc


class InsertionObjective(SelectionObjective):
    """Sufficiency by rank: the retained-only input should rank the item highest."""

    name = "ins"
    needs_truncate = False
    needs_retain = True

    def score(self, cdcg_trunc, cdcg_retain):
        return cdcg_retain


class DelKeObjective(SelectionObjective):
    """Necessity measured the way del_ke grades it: the target's probability.

    Same TRUNCATE probe as ``del``, probability readout. The graded metric is
    p(perturbed)/p(original); the denominator is constant across candidates
    for a row, so only the perturbed probability is needed.
    """

    name = "del_ke"
    needs_truncate = True
    needs_retain = False
    readout = "prob"

    def score(self, cdcg_trunc, cdcg_retain):
        return -cdcg_trunc          # holds the probability under readout="prob"


class InsKeObjective(SelectionObjective):
    """Sufficiency measured the way ins_ke grades it: the target's probability.

    Same RETAIN probe as ``ins``, probability readout. As for del_ke, the
    constant denominator of p(retained)/p(original) cancels in the argmax.
    """

    name = "ins_ke"
    needs_truncate = False
    needs_retain = True
    readout = "prob"

    def score(self, cdcg_trunc, cdcg_retain):
        # Holds the probability under readout="prob"; higher retained
        # probability = more sufficient = better.
        return cdcg_retain


_REGISTRY: dict[str, type[SelectionObjective]] = {
    cls.name: cls
    for cls in (
        DeletionObjective,
        InsertionObjective,
        DelKeObjective,
        InsKeObjective,
    )
}


def build_selection_objective(name: str) -> SelectionObjective:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise UnknownSelectionObjective(name, list(_REGISTRY)) from None


def row_input(inp: ExplainInput, b: int) -> ExplainInput:
    """Single-row ``[1, ...]`` view of a batch ExplainInput."""
    sl = slice(b, b + 1)
    return dc_replace(
        inp,
        items_to_explain=inp.items_to_explain[sl],
        user_ids=inp.user_ids[sl],
        history=None if inp.history is None else inp.history[sl],
    )


def tile_input(inp: ExplainInput, n: int) -> ExplainInput:
    """Tile a B=1 ExplainInput to B=n."""

    def rep(t: torch.Tensor | None, dims: int) -> torch.Tensor | None:
        if t is None:
            return None
        return t.expand(n, *t.shape[1:]).contiguous() if dims > 1 else t.expand(n).contiguous()

    return dc_replace(
        inp,
        items_to_explain=rep(inp.items_to_explain, 1),
        user_ids=rep(inp.user_ids, 1),
        history=rep(inp.history, 2),
    )


class CandidateAuditor:
    """Scores candidate explanations through the evaluator's own probe path.

    Both probes use the evaluator's Truncate/Retain perturbations,
    ``full_sort_predict`` and its original-history re-mask (items the
    perturbation removed must not re-enter the ranking), so the readouts see
    exactly what the evaluator grades.

    Probes call the recommender directly, never the evaluator's counted
    ``_predict``.
    """

    def __init__(
        self, recommender: BaseRecommender, objective: SelectionObjective
    ) -> None:
        self.recommender = (
            recommender.module if hasattr(recommender, "module") else recommender
        )
        self.objective = objective
        self._truncate = TruncatePerturbation(self.recommender)
        self._retain = RetainPerturbation(self.recommender)

    @torch.no_grad()
    def cdcg_probes(
        self, inp: ExplainInput, importance: torch.Tensor, ke: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Probe value after truncating / retaining each row's top-``ke``.

        Args:
            inp: Batch input; each row is scored independently.
            importance: ``[B, num_items + 1]`` per-row candidate scores.
            ke: Number of top positions the perturbations act on.

        Returns:
            ``(trunc, retain)``, each ``[B]`` or ``None`` when the objective
            does not need that probe.
        """
        attrs = Attributions(importance=importance)
        ct = self._probe(inp, self._truncate, attrs, ke) if self.objective.needs_truncate else None
        cr = self._probe(inp, self._retain, attrs, ke) if self.objective.needs_retain else None
        return ct, cr

    def _probe(
        self,
        inp: ExplainInput,
        perturbation: TruncatePerturbation | RetainPerturbation,
        attrs: Attributions,
        ke: int,
    ) -> torch.Tensor:
        perturbed = perturbation.apply(inp, attrs, ke)
        scores, _ = self.recommender.full_sort_predict(perturbed.to_recommender_batch())
        # Original-history re-mask, as in the evaluator, before either readout.
        assert inp.history is not None
        scores = scores.masked_fill(
            inp.history.to(scores.device).bool(), float("-inf")
        )
        if self.objective.readout == "prob":
            return self._target_prob(inp, scores)
        return self._cdcg(inp, scores)

    def _target_prob(self, inp: ExplainInput, scores: torch.Tensor) -> torch.Tensor:
        """Target probability, matching del_ke / ins_ke grading (PAD not masked)."""
        return target_probability(
            scores,
            inp.items_to_explain.to(scores.device),
            self.recommender.score_activation,
        )

    def _cdcg(self, inp: ExplainInput, scores: torch.Tensor) -> torch.Tensor:
        """``1 / log2(rank + 1)`` of the target, PAD excluded from the ranking."""
        scores = scores.clone()
        scores[:, PAD_IDX] = float("-inf")
        targets = inp.items_to_explain.to(scores.device)
        target_scores = scores.gather(1, targets.view(-1, 1))
        dtype = torch.float32 if scores.device.type == "mps" else torch.float64
        ranks = (scores > target_scores).sum(dim=1).to(dtype) + 1.0
        return 1.0 / torch.log2(ranks + 1.0)


__all__ = [
    "CandidateAuditor",
    "DelKeObjective",
    "DeletionObjective",
    "InsKeObjective",
    "InsertionObjective",
    "SelectionObjective",
    "UnknownSelectionObjective",
    "build_selection_objective",
    "row_input",
    "tile_input",
]
