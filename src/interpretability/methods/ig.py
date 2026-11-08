from __future__ import annotations

import torch

from src.capabilities import Family
from src.interpretability.base import BaseExplainer
from src.interpretability.inputs import Attributions, ExplainInput
from src.interpretability.registry import register_explainer
from src.recommenders.base import BaseRecommender


class IGExplainer(BaseExplainer):
    """Integrated Gradients (Sundararajan et al., 2017) with an all-zeros baseline.

    IG in input space over the multi-hot history: gradients of the target
    item's logit are summed on the grid ``linspace(0, 1, num_steps + 1)``
    (endpoints included) and divided by ``num_steps``. Attribution:
    ``history * avg_grad * (history - baseline)``. The attribution does not
    depend on ``ke``.
    """

    name = "ig"
    supported_families = frozenset({Family.HISTORY_VECTOR})
    is_dynamic = False

    def __init__(
        self,
        recommender: BaseRecommender,
        *,
        device: torch.device | str = "cpu",
        num_steps: int = 10,
    ) -> None:
        super().__init__(recommender, device=device)
        self.num_steps = num_steps

    def explain(self, inp: ExplainInput, *, ke: int | None = None) -> Attributions:
        self.recommender.eval()
        history = inp.history.to(self.device).clone()  # [B, D]
        targets = inp.items_to_explain.to(self.device)  # [B]
        with torch.enable_grad():
            ig = self._compute_ig(history, torch.zeros_like(history), targets)
        importance = ig.masked_fill(~inp.explainable_mask.to(ig.device), float("-inf"))
        return Attributions(importance=importance)

    def _compute_ig(
        self, history: torch.Tensor, baseline: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """One IG pass for one baseline. Returns ``[B, D]``."""
        B, D = history.shape
        alphas = torch.linspace(0.0, 1.0, self.num_steps + 1, device=self.device)
        arange_B = torch.arange(B, device=self.device)
        grad_sum = torch.zeros(B, D, device=self.device)

        for alpha in alphas:
            x = (baseline + alpha * (history - baseline)).detach().requires_grad_(True)
            scores = self.recommender(x)
            scores[arange_B, targets].sum().backward()
            grad_sum = grad_sum + x.grad.detach()

        avg_grad = grad_sum / self.num_steps
        return history * avg_grad * (history - baseline)  # [B, D]


@register_explainer("ig")
def _build_ig(
    recommender: BaseRecommender,
    params: dict,
    device: torch.device,
) -> IGExplainer:
    return IGExplainer(recommender, device=device, num_steps=params["num_steps"])


__all__ = ["IGExplainer"]
