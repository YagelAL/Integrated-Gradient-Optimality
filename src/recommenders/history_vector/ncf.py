from __future__ import annotations

import torch
import torch.nn as nn

from src.config.schemas import RecommenderCfg
from src.recommenders.factory import register_recommender
from src.recommenders.history_vector.base import BaseHistoryVectorRecommender
from src.recommenders.losses import sampled_pop_mse


class NCF(BaseHistoryVectorRecommender):
    """NeuMF (He et al., 2017): a GMF branch and an MLP branch fused by a linear head.

    * **MLP tower**: fixed depth (``_NUM_LAYERS = 3``) with pyramid widths. The MLP-side
      user/item projections have width ``factor_num * 2**(_NUM_LAYERS-1)``; the concatenated
      interaction (``factor_num * 2**_NUM_LAYERS``) is halved at every layer down to
      ``factor_num``.
    * **GMF branch**: exactly affine in the history. The ReLU tower is the model's only
      nonlinear part.

    Training uses ``sampled_pop_mse`` (shared with the MLP): the held-out target is the
    positive and one unseen item sampled proportionally to popularity is the negative,
    loss = ``mean((1-sigma_pos)^2) + beta * mean(sigma_neg^2)``. The popularity vector is
    installed once at train time via :meth:`set_popularity` (uniform until then).

    ``forward(history)`` scores the whole catalog in one pass and returns raw logits
    ``[B, V]``; the loss gathers the positive/negative columns it needs.

    Memory: ``O(B * V * M)`` from the broadcast interaction tensor, where
    ``M = factor_num * 2**(_NUM_LAYERS-1)``.
    """

    score_activation = "sigmoid"

    # MLP tower depth (fixed).
    _NUM_LAYERS = 3

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
        num_layers = self._NUM_LAYERS

        # bias=False: Linear-on-multi-hot == summed embedding lookup; a bias
        # would add a constant item-independent shift and conflate the
        # user-side and item-side signals.
        mlp_embed_dim = factor_num * (2 ** (num_layers - 1))
        self.user_gmf = nn.Linear(self.input_dim, factor_num, bias=False)
        self.item_gmf = nn.Linear(self.input_dim, factor_num, bias=False)
        self.user_mlp = nn.Linear(self.input_dim, mlp_embed_dim, bias=False)
        self.item_mlp = nn.Linear(self.input_dim, mlp_embed_dim, bias=False)

        # Pyramid tower: interaction width factor_num*2**num_layers, halved
        # each layer down to factor_num.
        mlp_modules: list[nn.Module] = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_modules)

        # GMF (factor_num) ⊕ MLP tail (factor_num) -> scalar.
        self.predict_layer = nn.Linear(2 * factor_num, 1)

        # Per-item sampling weights for the negative in ``sampled_pop_mse``.
        # Defaults to uniform; scripts/train.py overwrites it with the
        # training-split popularity via ``set_popularity``. persistent=False
        # keeps it out of the state_dict (no checkpoint coupling) while still
        # moving with ``.to(device)``.
        self.register_buffer(
            "item_pop", torch.ones(self.input_dim), persistent=False
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Small-normal embeddings, xavier MLP, kaiming predict head.
        for proj in (self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp):
            nn.init.normal_(proj.weight, std=0.01)
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity="sigmoid")
        if self.predict_layer.bias is not None:
            nn.init.zeros_(self.predict_layer.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        batch_size = history.size(0)
        # column v of W.T is item v's embedding (Linear weight is [out, in]).
        user_g = self.user_gmf(history)            # [B, F]
        item_g = self.item_gmf.weight.T            # [V, F]
        gmf = user_g.unsqueeze(1) * item_g.unsqueeze(0)  # [B, V, F]

        user_m = self.user_mlp(history)            # [B, M]
        item_m = self.item_mlp.weight.T            # [V, M]
        n_items = item_m.size(0)
        user_m_exp = user_m.unsqueeze(1).expand(batch_size, n_items, -1)
        items_m_exp = item_m.unsqueeze(0).expand(batch_size, -1, -1)
        interaction = torch.cat([user_m_exp, items_m_exp], dim=-1)  # [B, V, 2M]
        mlp_out = self.mlp_layers(interaction)     # [B, V, F]

        fused = torch.cat([gmf, mlp_out], dim=-1)  # [B, V, 2F]
        return self.predict_layer(fused).squeeze(-1)  # [B, V]

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


@register_recommender("ncf")
def _build_ncf(cfg: RecommenderCfg, num_items: int) -> NCF:
    # hidden_dim -> factor_num.
    beta = cfg.beta if cfg.beta is not None else 0.5
    return NCF(
        num_items=num_items,
        factor_num=cfg.hidden_dim,
        dropout=cfg.dropout,
        beta=beta,
    )


__all__ = ["NCF"]
