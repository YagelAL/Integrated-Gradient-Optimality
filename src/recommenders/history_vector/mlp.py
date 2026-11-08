from __future__ import annotations

import torch
import torch.nn as nn

from src.config.schemas import RecommenderCfg
from src.recommenders.factory import register_recommender
from src.recommenders.history_vector.base import BaseHistoryVectorRecommender
from src.recommenders.losses import sampled_pop_mse


class MLP(BaseHistoryVectorRecommender):
    """Two-tower MLP, affine in the history.

    Separate Linear projections for user and item into a shared hidden
    space; the score for ``(user, item v)`` is the inner product
    ``users_fc(history) · items_fc(one_hot(v))``, computed for the whole
    catalog in one matmul.
    """

    score_activation = "sigmoid"

    def __init__(
        self, *, num_items: int, hidden_dim: int, dropout: float, beta: float = 0.5
    ) -> None:
        super().__init__(num_items=num_items)
        self.beta = beta
        self.users_fc = nn.Linear(self.input_dim, hidden_dim)
        self.items_fc = nn.Linear(self.input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Per-item negative-sampling weights for sampled_pop_mse. Defaults to
        # uniform; scripts/train.py overwrites with the training-split
        # popularity via set_popularity. persistent=False keeps it out of the
        # state_dict while still moving with .to(device).
        self.register_buffer(
            "item_pop", torch.ones(self.input_dim), persistent=False
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Algebraic identity with ``user_vec @ items_fc(eye(V)).T``:

            items_fc(e_v) = W[:, v] + b
            score[b, v]   = sum_h user_vec[b, h] * (W[h, v] + b[h])
                          = (user_vec @ W)[b, v] + (user_vec @ b)[b]

        The bias contributes a per-user scalar offset shared across items.
        Emits raw logits; ``calculate_loss`` applies the sigmoid.
        """
        if history.dim() == 1:
            history = history.unsqueeze(0)

        user_vec = self.users_fc(self.input_dropout(history))
        W = self.items_fc.weight
        b = self.items_fc.bias
        return user_vec @ W + (user_vec @ b).unsqueeze(-1)

    def set_popularity(self, pop: torch.Tensor) -> None:
        """Install the per-item popularity used for negative sampling.

        ``pop`` must be length ``input_dim`` (``num_items + 1``, PAD at 0),
        indexed the same as the multi-hot columns. Called once before
        training; until then the buffer is uniform (1s).
        """
        if pop.shape != self.item_pop.shape:
            raise ValueError(
                f"popularity shape {tuple(pop.shape)} != expected "
                f"{tuple(self.item_pop.shape)}"
            )
        self.item_pop = pop.to(dtype=torch.float32, device=self.item_pop.device)

    def calculate_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history = batch["history"]
        target = batch["target"]
        return sampled_pop_mse(
            torch.sigmoid(self.forward(history)),
            target,
            history,
            self.item_pop,
            beta=self.beta,
        )


@register_recommender("mlp")
def _build_mlp(cfg: RecommenderCfg, num_items: int) -> MLP:
    beta = cfg.beta if cfg.beta is not None else 0.5
    return MLP(
        num_items=num_items, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout, beta=beta
    )


__all__ = ["MLP"]
