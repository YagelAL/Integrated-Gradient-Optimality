from __future__ import annotations

import torch


def _sample_pop_neg(
    scores: torch.Tensor,
    target: torch.Tensor,
    history: torch.Tensor,
    pop: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pos, neg)``: the target score and one popularity-sampled negative score per user.

    The negative pool is the user's unseen items (``history < 0.5``) minus the
    target; one item is drawn per user with probability proportional to ``pop``.

    Shapes: scores [B, V], target [B], history [B, V], pop [V].
    """
    pos = scores.gather(1, target.unsqueeze(1)).squeeze(1)  # [B]
    weights = pop.unsqueeze(0) * (history < 0.5).to(pop.dtype)  # [B, V]
    weights.scatter_(1, target.unsqueeze(1), 0.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
    neg_idx = torch.multinomial(weights, 1).squeeze(1)  # [B]
    neg = scores.gather(1, neg_idx.unsqueeze(1)).squeeze(1)  # [B]
    return pos, neg


def sampled_pop_mse(
    scores: torch.Tensor,
    target: torch.Tensor,
    history: torch.Tensor,
    pop: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """MSE on 1 positive + 1 popularity-sampled negative.

        pos_loss = mean((1 - p_pos) ** 2)   # regress positive prob -> 1
        neg_loss = mean(p_neg ** 2)         # regress sampled negative prob -> 0
        loss     = pos_loss + beta * neg_loss

    Shapes: scores [B, V] in [0, 1], target [B], history [B, V], pop [V].
    """
    pos, neg = _sample_pop_neg(scores, target, history, pop)
    return ((1.0 - pos) ** 2).mean() + beta * (neg ** 2).mean()


__all__ = ["sampled_pop_mse"]
