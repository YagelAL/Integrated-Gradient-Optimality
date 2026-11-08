from __future__ import annotations

import torch

from src.interpretability.fidelity.metric import (
    FidelityMetric,
    PerturbationMode,
    register_fidelity_metric,
    target_probability,
)


class INSAtKe(FidelityMetric):
    """INS@``ke`` — target-probability ratio after retaining only the top-``ke`` items.

    Per user, the value is ``p(x_u^Ke)_y / p(x_u)_y``: the target item's
    probability on the RETAIN-perturbed input (only the top-``ke``
    most-explaining history items kept), normalized by
    its probability on the original input. Probabilities come from the
    recommender's raw scores under its declared ``score_activation``, so the
    value is comparable across users and models. Higher means the explainer
    found a small subset that, on its own, already restores most of the
    model's original confidence — i.e. higher fidelity.
    """

    name = "ins_ke"
    depends_on_kr = False
    requires_modes = frozenset({PerturbationMode.RETAIN})
    supports_monotonic = True
    monotonic_reduction = "cummax"

    def __init__(self, *, score_mode: str) -> None:
        if score_mode not in ("sigmoid", "softmax"):
            raise ValueError(
                f"ins_ke requires score_mode 'sigmoid' or 'softmax', got "
                f"{score_mode!r}; the fidelity evaluator injects it from the "
                "recommender's score_activation"
            )
        self.score_mode = score_mode
        super().__init__()

    def per_user_values(
        self,
        *,
        orig_scores: torch.Tensor,
        perturbed_scores: torch.Tensor,
        items_to_explain: torch.Tensor,
        ke: int,  # noqa: ARG002
        kr: int | None,  # noqa: ARG002 (depends_on_kr=False; the evaluator passes None)
    ) -> torch.Tensor:
        orig_p = target_probability(orig_scores, items_to_explain, self.score_mode)
        pert_p = target_probability(perturbed_scores, items_to_explain, self.score_mode)
        return pert_p.to(torch.float64) / orig_p.to(torch.float64).clamp_min(1e-9)


@register_fidelity_metric("ins_ke")
def _build_ins_ke(params: dict) -> INSAtKe:
    return INSAtKe(score_mode=params.get("score_mode", ""))


__all__ = ["INSAtKe"]
