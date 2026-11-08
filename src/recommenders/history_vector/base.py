from __future__ import annotations

from abc import abstractmethod

import torch

from src.capabilities import Family
from src.recommenders.base import BaseRecommender


class BaseHistoryVectorRecommender(BaseRecommender):
    """Family base for history-vector recommenders.

    Shape contract: ``forward(history: [B, num_items+1]) -> logits: [B, num_items+1]``.
    """

    family = Family.HISTORY_VECTOR

    def __init__(self, num_items: int) -> None:
        super().__init__(num_items=num_items)
        # +1 for the PAD slot at index 0 (item ids live at 1..num_items).
        self.input_dim = num_items + 1

    @abstractmethod
    def forward(self, history: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def calculate_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor: ...

    def full_sort_predict(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(scores, target)``; items already in the history are set to ``-inf``."""
        scores = self.forward(batch["history"])
        scores = scores.masked_fill(batch["history"].bool(), float("-inf"))
        return scores, batch["target"]


__all__ = ["BaseHistoryVectorRecommender"]
