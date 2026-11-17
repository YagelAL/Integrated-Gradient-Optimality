from __future__ import annotations

import torch
import torch.nn as nn

from src.config.schemas import RecommenderCfg
from src.recommenders.factory import register_recommender
from src.recommenders.history_vector.base import BaseHistoryVectorRecommender
from src.recommenders.losses import sampled_pop_mse


class NCFGMF(BaseHistoryVectorRecommender):
    """NCF with the nonlinear MLP branch removed: GMF only.

    NCF fuses two branches through a final linear ``predict_layer``: a GMF branch that is
    exactly affine in ``history`` (``user_gmf`` has ``bias=False``; ``item_gmf`` embeddings
    are fixed parameters) and an MLP branch whose ReLU pyramid is the model's only source
    of nonlinearity. This class keeps only the GMF branch, so ``forward()`` is exactly
    affine in ``history`` for a fixed target item. It is a separate recommender (not a
    flag on ``NCF``), with the same init scheme and loss as NCF's GMF branch.
    """

    score_activation = "sigmoid"

    def __init__(
        self,
        *,
        num_items: int,
        factor_num: int,
        dropout: float,
        beta: float = 0.5,
    ) -> None:
        super().__init__(num_items=num_items)
        self.factor_num = factor_num
        self.beta = beta
        # dropout is accepted (RecommenderCfg always supplies one) but unused:
        # there is no MLP tower left to regularize.
        self.dropout = dropout

        # bias=False: matches NCF's GMF branch exactly (Linear-on-multi-hot ==
        # summed embedding lookup; a bias would add a constant item-independent
        # shift and conflate user-side/item-side signals).
        self.user_gmf = nn.Linear(self.input_dim, factor_num, bias=False)
        self.item_gmf = nn.Linear(self.input_dim, factor_num, bias=False)
        self.predict_layer = nn.Linear(factor_num, 1)

        self.register_buffer(
            "item_pop", torch.ones(self.input_dim), persistent=False
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Same scheme as NCF's GMF-branch init.
        for proj in (self.user_gmf, self.item_gmf):
            nn.init.normal_(proj.weight, std=0.01)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity="sigmoid")
        if self.predict_layer.bias is not None:
            nn.init.zeros_(self.predict_layer.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        user_g = self.user_gmf(history)             # [B, F]
        item_g = self.item_gmf.weight.T             # [V, F]
        gmf = user_g.unsqueeze(1) * item_g.unsqueeze(0)  # [B, V, F]
        return self.predict_layer(gmf).squeeze(-1)   # [B, V]

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


@register_recommender("ncf_gmf")
def _build_ncf_gmf(cfg: RecommenderCfg, num_items: int) -> NCFGMF:
    beta = cfg.beta if cfg.beta is not None else 0.5
    return NCFGMF(
        num_items=num_items,
        factor_num=cfg.hidden_dim,
        dropout=cfg.dropout,
        beta=beta,
    )


__all__ = ["NCFGMF"]
