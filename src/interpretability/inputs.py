from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from src.capabilities import Family


@dataclass(frozen=True)
class ExplainInput:
    """One batch of explanation inputs.

    The evaluator builds one ``ExplainInput`` per batch and feeds the same
    frozen object to every explainer. Perturbations derive new inputs via
    :meth:`replace_history` rather than mutating in place.

    Attributes:
        family: Recommender family the input is shaped for.
        items_to_explain: ``[B]`` item ids whose recommendation is attributed.
        user_ids: ``[B]`` user ids.
        history: ``[B, num_items + 1]`` multi-hot interaction vector.
    """

    family: Family
    items_to_explain: torch.Tensor
    user_ids: torch.Tensor
    history: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.history is None:
            raise ValueError("HISTORY_VECTOR ExplainInput requires history")

    @property
    def batch_size(self) -> int:
        return int(self.user_ids.shape[0])

    @property
    def explainable_mask(self) -> torch.Tensor:
        """Bool ``[B, num_items + 1]`` of attributable positions (``history > 0``).

        Explainers ``masked_fill(~explainable_mask, -inf)`` so the
        perturbation top-k can never select a non-history item.
        """
        assert self.history is not None
        return self.history > 0

    @classmethod
    def from_batch(
        cls,
        batch: dict[str, torch.Tensor],
        *,
        family: Family,
        items_to_explain: torch.Tensor,
    ) -> ExplainInput:
        """Adapt a dataloader batch; ``items_to_explain`` is resolved by the caller."""
        return cls(
            family=family,
            items_to_explain=items_to_explain,
            user_ids=batch["user_id"],
            history=batch["history"],
        )

    def replace_history(self, *, history: torch.Tensor) -> ExplainInput:
        """Return a new input with a perturbed history."""
        return replace(self, history=history)

    def to_recommender_batch(self) -> dict[str, torch.Tensor]:
        """Convert back to the dict shape that ``full_sort_predict`` consumes."""
        return {
            "user_id": self.user_ids,
            "history": self.history,  # type: ignore[dict-item]
            "target": self.items_to_explain,
        }


@dataclass(frozen=True)
class Attributions:
    """Per-item importance scores ``[B, num_items + 1]`` produced by an explainer.

    Larger = more important. Non-history positions must be ``-inf`` so the
    perturbation top-k can never select them.
    """

    importance: torch.Tensor


__all__ = ["Attributions", "ExplainInput"]
