from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch

from src.exceptions import InvalidPerturbation
from src.interpretability.fidelity.metric import PerturbationMode
from src.interpretability.inputs import Attributions, ExplainInput
from src.recommenders.base import BaseRecommender


class Perturbation(ABC):
    """Strategy for perturbing a history by its top-``ke`` important items.

    Attributes:
        mode: The :class:`PerturbationMode` this strategy implements.
    """

    mode: ClassVar[PerturbationMode]

    def __init__(self, recommender: BaseRecommender) -> None:
        self.recommender = (
            recommender.module if hasattr(recommender, "module") else recommender
        )

    @abstractmethod
    def apply(self, inp: ExplainInput, attrs: Attributions, ke: int) -> ExplainInput:
        """Return a new :class:`ExplainInput` with the perturbation applied."""

    def _check_shape(self, history: torch.Tensor, importance: torch.Tensor) -> None:
        if importance.shape != history.shape:
            raise InvalidPerturbation(
                self.mode.value,
                f"importance shape {tuple(importance.shape)} does not match "
                f"history shape {tuple(history.shape)}",
            )

    @staticmethod
    def _select_positions(
            reference: torch.Tensor,
            importance: torch.Tensor,
            explainable: torch.Tensor,
            ke: int,
    ) -> torch.Tensor:
        """Return a ``[B, D]`` bool mask over the ``ke`` most important
        explainable positions per row.

        ``reference`` is only used for its shape/device. Non-explainable
        positions are set to ``-inf``, so they are only reached when a row
        has fewer than ``ke`` explainable positions; the ``valid`` check then
        excludes them, so a row never selects a non-explainable position.
        """
        importance = importance.masked_fill(~explainable, float("-inf"))
        idx = importance.topk(ke, dim=-1).indices  # [B, ke]
        valid = importance.gather(1, idx) > float("-inf")  # [B, ke]
        mask = torch.zeros_like(reference, dtype=torch.bool)
        mask.scatter_(1, idx, valid)
        return mask


class TruncatePerturbation(Perturbation):
    """Zero the top-``ke`` important columns of the multi-hot history."""

    mode = PerturbationMode.TRUNCATE

    def apply(self, inp: ExplainInput, attrs: Attributions, ke: int) -> ExplainInput:
        assert inp.history is not None
        history = inp.history
        self._check_shape(history, attrs.importance)
        zero_mask = self._select_positions(
            history, attrs.importance, inp.explainable_mask, ke
        )
        return inp.replace_history(history=history.masked_fill(zero_mask, 0.0))


class RetainPerturbation(Perturbation):
    """Keep only the top-``ke`` important columns of the history, zero the rest."""

    mode = PerturbationMode.RETAIN

    def apply(self, inp: ExplainInput, attrs: Attributions, ke: int) -> ExplainInput:
        assert inp.history is not None
        history = inp.history
        self._check_shape(history, attrs.importance)
        keep_mask = self._select_positions(
            history, attrs.importance, inp.explainable_mask, ke
        )
        return inp.replace_history(history=history.masked_fill(~keep_mask, 0.0))


_PERTURBATION_CLASSES: dict[PerturbationMode, type[Perturbation]] = {
    PerturbationMode.TRUNCATE: TruncatePerturbation,
    PerturbationMode.RETAIN: RetainPerturbation,
}


def build_perturbation(
        mode: PerturbationMode | str, recommender: BaseRecommender
) -> Perturbation:
    mode_enum = PerturbationMode(mode) if isinstance(mode, str) else mode
    return _PERTURBATION_CLASSES[mode_enum](recommender)


__all__ = [
    "Perturbation",
    "RetainPerturbation",
    "TruncatePerturbation",
    "build_perturbation",
]
