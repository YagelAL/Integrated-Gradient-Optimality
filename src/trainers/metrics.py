"""Top-k recommendation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Metric(ABC):
    """A stateful, streaming metric.

    Subclasses accumulate per-batch state in update() and return a
    final scalar from compute(). Inputs are the uniform output of
    model.full_sort_predict(batch):

    * scores: [N, num_items + 1] — one row per prediction.
    * labels: [N] — target item id per row, always >= 1.

    The PAD slot lives at column 0; metrics mask it to -inf before ranking
    so PAD can never win the top-k.
    """

    name: str

    def __init__(self) -> None:
        """Initialize accumulators by calling reset()."""
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state for a fresh evaluation pass."""

    @abstractmethod
    def update(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate one batch into the running state.

        Args:
            scores: [N, num_items + 1] float tensor of per-item scores.
            labels: [N] long tensor of target item ids (>= 1).
        """

    @abstractmethod
    def compute(self) -> float:
        """Compute the final metric value over everything accumulated so far.

        Returns:
            Scalar metric value.
        """


def _mask_pad(scores: torch.Tensor) -> torch.Tensor:
    """Set the PAD column (index 0) to -inf so it can't win the top-k.

    Args:
        scores: [N, num_items + 1] score tensor.

    Returns:
        A clone of scores with column 0 replaced by -inf.
    """
    out = scores.clone()
    out[:, 0] = float("-inf")
    return out


class HRAtK(Metric):
    """Hit Rate @ K — fraction of cases where the target is in the top-k scored items."""

    def __init__(self, k: int = 10) -> None:
        """Configure the cutoff.

        Args:
            k: Top-k cutoff (must be >= 1).
        """
        self.k = k
        self.name = f"hr@{k}"
        super().__init__()

    def reset(self) -> None:
        self._hits = 0
        self._total = 0

    def update(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        scores = _mask_pad(scores)
        topk = scores.topk(self.k, dim=-1).indices  # [N, k]
        hits = (topk == labels.unsqueeze(-1)).any(dim=-1)
        self._hits += int(hits.sum().item())
        self._total += int(labels.numel())

    def compute(self) -> float:
        return self._hits / max(self._total, 1)


class MRR(Metric):
    """Mean Reciprocal Rank — averaged 1 / rank of the target."""

    name = "mrr"

    def reset(self) -> None:
        self._sum = 0.0
        self._total = 0

    def update(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        scores = _mask_pad(scores)
        target_scores = scores.gather(1, labels.unsqueeze(1))
        # Rank (1-indexed) = number of strictly higher scores + 1.
        ranks = (scores > target_scores).sum(dim=1).float() + 1.0
        self._sum += float((1.0 / ranks).sum().item())
        self._total += int(labels.numel())

    def compute(self) -> float:
        return self._sum / max(self._total, 1)


class NDCGAtK(Metric):
    """NDCG @ K with a single relevant item: 1 / log2(rank + 1) if in top-k, else 0."""

    def __init__(self, k: int = 10) -> None:
        """Configure the cutoff.

        Args:
            k: Top-k cutoff (must be >= 1).
        """
        self.k = k
        self.name = f"ndcg@{k}"
        super().__init__()

    def reset(self) -> None:
        self._sum = 0.0
        self._total = 0

    def update(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        scores = _mask_pad(scores)
        topk = scores.topk(self.k, dim=-1).indices  # [N, k]
        hits = (topk == labels.unsqueeze(-1))  # [N, k] one-hot per row at the matching column
        # Position of the hit (1-indexed); rows with no hit get 0.
        positions = torch.arange(1, self.k + 1, device=scores.device, dtype=torch.float)
        hit_positions = (hits.float() * positions).sum(dim=-1)
        ndcg = torch.where(
            hit_positions > 0,
            1.0 / torch.log2(hit_positions + 1.0),
            torch.zeros_like(hit_positions),
        )
        self._sum += float(ndcg.sum().item())
        self._total += int(labels.numel())

    def compute(self) -> float:
        return self._sum / max(self._total, 1)


class MPR(Metric):
    """Mean Percentile Rank — average percentile of the target item.

    Lower is better (target should rank near the top). Range is [0, 100].
    """

    name = "mpr"

    def reset(self) -> None:
        self._sum = 0.0
        self._total = 0
        self._catalog_size = 0

    def update(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        scores = _mask_pad(scores)
        target_scores = scores.gather(1, labels.unsqueeze(1))
        ranks = (scores > target_scores).sum(dim=1).float()  # 0-indexed
        # Catalog size excludes the PAD slot (column 0).
        catalog = scores.size(1) - 1
        self._catalog_size = catalog
        self._sum += float((ranks / max(catalog - 1, 1) * 100).sum().item())
        self._total += int(labels.numel())

    def compute(self) -> float:
        return self._sum / max(self._total, 1)


def default_metrics() -> list[Metric]:
    """Build the standard set of evaluation metrics.

    Returns:
        A fresh list of metric instances: HR@10, HR@20, MRR, NDCG@10, MPR.
    """
    return [HRAtK(k=10), HRAtK(k=20), MRR(), NDCGAtK(k=10), MPR()]


__all__ = ["HRAtK", "MPR", "MRR", "Metric", "NDCGAtK", "default_metrics"]
