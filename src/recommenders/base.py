from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar, Literal

import torch
import torch.nn as nn

from src.capabilities import Family


class BaseRecommender(nn.Module):
    """ABC for every recommender.

    The model owns its training loss (``calculate_loss``) and its eval
    scoring (``full_sort_predict``); the trainer just calls these methods.
    """

    family: ClassVar[Family]
    # How the raw logits from forward() map to a probability: "sigmoid" for
    # per-item pointwise objectives, "softmax" for catalog-competitive ones.
    score_activation: ClassVar[Literal["sigmoid", "softmax"]]

    def __init__(self, num_items: int, **kwargs: Any) -> None:
        super().__init__()
        self.num_items = num_items
        if not hasattr(type(self), "family"):
            raise TypeError(
                f"{type(self).__name__} must declare a class-level `family: Family`"
            )
        if getattr(type(self), "score_activation", None) not in ("sigmoid", "softmax"):
            raise TypeError(
                f"{type(self).__name__} must declare a class-level "
                "`score_activation: Literal['sigmoid', 'softmax']`"
            )

    @abstractmethod
    def calculate_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a scalar training loss for one batch ``{history, target, ...}``."""

    @abstractmethod
    def full_sort_predict(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score the catalog for evaluation. Returns ``(scores [B, V], labels [B])``."""

    def freeze(self) -> None:
        """Disable gradients and switch to eval mode."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()


__all__ = ["BaseRecommender"]
