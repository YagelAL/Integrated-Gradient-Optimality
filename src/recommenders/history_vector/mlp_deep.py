from __future__ import annotations

import torch
import torch.nn as nn

from src.config.schemas import RecommenderCfg
from src.recommenders.factory import register_recommender
from src.recommenders.history_vector.base import BaseHistoryVectorRecommender
from src.recommenders.losses import sampled_pop_mse


class MLPDeep(BaseHistoryVectorRecommender):
    """Two-tower MLP with a real non-linear, deep per-side network.

    Unlike ``MLP`` (one ``nn.Linear`` per tower, exactly affine), each tower
    here is a fixed-depth ``_NUM_LAYERS``-Linear / ReLU stack, so the score
    is not affine in the history.

    Item embeddings are computed by a forward pass over ``eye(V)`` on every
    call (``O(V^2)`` memory).
    """

    score_activation = "sigmoid"

    # Fixed depth, not a config field.
    _NUM_LAYERS = 5

    def __init__(
        self, *, num_items: int, hidden_dim: int, dropout: float, beta: float = 0.5
    ) -> None:
        super().__init__(num_items=num_items)
        self.beta = beta
        self.input_dropout = nn.Dropout(dropout)
        self.user_tower = self._build_tower(hidden_dim)
        self.item_tower = self._build_tower(hidden_dim)

        # Per-item negative-sampling weights for sampled_pop_mse. Defaults to
        # uniform; scripts/train.py overwrites with the training-split
        # popularity via set_popularity. persistent=False keeps it out of the
        # state_dict while still moving with .to(device).
        self.register_buffer(
            "item_pop", torch.ones(self.input_dim), persistent=False
        )

        self._init_weights()

    def _build_tower(self, hidden_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
        for _ in range(self._NUM_LAYERS - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        # He/Kaiming init for the ReLU stacks (no normalization layers).
        for tower in (self.user_tower, self.item_tower):
            for m in tower:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() == 1:
            history = history.unsqueeze(0)

        user_emb = self.user_tower(self.input_dropout(history))  # [B, H]
        eye = torch.eye(self.input_dim, device=history.device, dtype=history.dtype)
        item_emb = self.item_tower(eye)  # [V, H]
        return user_emb @ item_emb.T

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


@register_recommender("mlp_deep")
def _build_mlp_deep(cfg: RecommenderCfg, num_items: int) -> MLPDeep:
    beta = cfg.beta if cfg.beta is not None else 0.5
    return MLPDeep(
        num_items=num_items, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout, beta=beta
    )


__all__ = ["MLPDeep"]
