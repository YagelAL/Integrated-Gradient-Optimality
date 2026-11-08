from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch

from src.capabilities import Family
from src.exceptions import IncompatibleExplainer
from src.interpretability.inputs import Attributions, ExplainInput
from src.recommenders.base import BaseRecommender


class BaseExplainer(ABC):
    """ABC for every explanation method.

    Attributes:
        name: Registry key; matches the ``name`` field of a budget-config entry.
        supported_families: Recommender families this explainer can run
            against; checked once at construction.
        is_dynamic: ``True`` => the explanation depends on ``ke`` (e.g. a
            search for the best ``ke``-subset), so ``explain()`` is called
            once per ``(user, item_rank, ke)``. ``False`` => the ranking is
            ``ke``-independent, computed once per ``(user, item_rank)`` and
            reused across the ``ke`` grid.
    """

    name: ClassVar[str]
    supported_families: ClassVar[frozenset[Family]]
    is_dynamic: ClassVar[bool] = False

    def __init__(
        self,
        recommender: BaseRecommender,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        """Bind a frozen recommender (DDP wrappers are unwrapped) and validate it.

        Raises:
            IncompatibleExplainer: ``recommender.family`` is not in
                ``cls.supported_families``.
        """
        unwrapped = recommender.module if hasattr(recommender, "module") else recommender
        if unwrapped.family not in self.supported_families:
            raise IncompatibleExplainer(
                explainer_type=type(self).__name__,
                recommender_type=type(unwrapped).__name__,
                family_mismatch=(unwrapped.family, self.supported_families),
            )
        self.recommender: BaseRecommender = unwrapped
        self.device = torch.device(device)

    @abstractmethod
    def explain(self, inp: ExplainInput, *, ke: int | None = None) -> Attributions:
        """Produce per-item importance scores ``[B, num_items + 1]`` for one batch.

        ``ke`` is only consulted by dynamic explainers.
        """


__all__ = ["BaseExplainer"]
