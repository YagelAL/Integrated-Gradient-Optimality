from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum, unique
from typing import ClassVar, Literal, TypeAlias

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.exceptions import XaiRecsysError


def target_probability(
    scores: torch.Tensor, items: torch.Tensor, score_mode: str
) -> torch.Tensor:
    """Target-item probability from raw recommender scores.

    ``full_sort_predict`` emits raw logits (family contract); ``score_mode``
    is the recommender's declared ``score_activation``. ``-inf`` entries
    (masked candidates) are safe under both squashes.

    Args:
        scores: ``[B, num_items+1]`` raw scores.
        items: ``[B]`` long tensor of target item ids.
        score_mode: ``"sigmoid"`` (per-item pointwise models) or
            ``"softmax"`` (catalog-competitive models).

    Returns:
        ``[B]`` float tensor of target probabilities.
    """
    if score_mode == "sigmoid":
        return torch.sigmoid(scores.gather(1, items.unsqueeze(1)).squeeze(1))
    if score_mode == "softmax":
        probs = F.softmax(scores, dim=-1)
        return probs.gather(1, items.unsqueeze(1)).squeeze(1)
    raise ValueError(
        f"unknown score_mode: {score_mode!r}; expected 'sigmoid' or 'softmax'"
    )


@unique
class PerturbationMode(str, Enum):
    """Hard input perturbations used by the fidelity metrics.

    Attributes:
        TRUNCATE: Zero the top-``ke`` important history columns (DEL@Ke).
        RETAIN: Keep only the top-``ke`` important history columns,
            zeroing the rest (INS@Ke).
    """

    TRUNCATE = "truncate"
    RETAIN = "retain"


class FidelityMetric(ABC):
    """Per-user fidelity metric with default streaming aggregation.

    Concrete subclasses implement exactly one method:
    :meth:`per_user_values`, which returns the per-user scalar signal for
    a single ``(batch, ke[, kr])`` slice. The default ``update`` and
    ``compute`` accumulate the streaming raw mean from those values; the
    evaluator captures the same per-user vector and feeds it to
    :func:`monotonic_aggregate` (post-pass on rank 0) when monotonic
    correction or dumps are enabled.

    Class attributes:
        name: Registry key + display name.
        requires_modes: Perturbation modes whose perturbed-scores tensor
            ``per_user_values()`` consumes. The evaluator only computes
            modes that appear in at least one active metric's set.
        depends_on_kr: ``True`` ⇒ ``per_user_values`` is called once per
            ``kr`` in the configured grid; ``False`` ⇒ ``kr=None`` and the
            metric ignores ``kr`` entirely.
        supports_monotonic: ``True`` ⇒ this metric's per-user values can
            be aggregated by :func:`monotonic_aggregate` across the ``ke``
            axis via :attr:`monotonic_reduction`. ``False`` ⇒ only the
            raw streaming mean is reported.
        monotonic_reduction: ``"cummin"`` or ``"cummax"``. Applied per
            user along the ``ke`` axis after the run, then averaged
            across users.

    Streaming buffers live as scalar ``torch.float64`` tensors registered
    via :meth:`_register_buffer`; the default :meth:`all_reduce`
    SUM-reduces them across DDP ranks once at end-of-eval.
    """

    name: ClassVar[str]
    requires_modes: ClassVar[frozenset[PerturbationMode]]
    depends_on_kr: ClassVar[bool]
    supports_monotonic: ClassVar[bool] = False
    monotonic_reduction: ClassVar[Literal["cummin", "cummax"]] = "cummin"

    def __init__(self) -> None:
        self._buffers: dict[str, torch.Tensor] = {}
        self.reset()

    # ------------------------------------------------------------------
    # Buffer helpers (used by the default reset/compute).
    # ------------------------------------------------------------------

    def _register_buffer(self, key: str, value: float = 0.0) -> None:
        """Declare a scalar float buffer to be SUM-reduced across ranks."""
        self._buffers[key] = torch.tensor(value, dtype=torch.float64)

    def _add(self, key: str, value: float | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            value = float(value.detach().to(dtype=torch.float64).item())
        self._buffers[key] += value

    def _get(self, key: str) -> float:
        return float(self._buffers[key].item())

    # ------------------------------------------------------------------
    # The one method a subclass must implement.
    # ------------------------------------------------------------------

    @abstractmethod
    def per_user_values(
        self,
        *,
        orig_scores: torch.Tensor,
        perturbed_scores: torch.Tensor,
        items_to_explain: torch.Tensor,
        ke: int,
        kr: int | None,
    ) -> torch.Tensor:
        """Return the ``[B]`` per-user signal for one slice.

        Args:
            orig_scores: ``[B, num_items+1]`` recommender scores on the
                original input. Provided for metrics that need the
                pre-perturbation rank.
            perturbed_scores: ``[B, num_items+1]`` recommender scores on
                the perturbed input for the current ``(explainer, ke, mode)``.
            items_to_explain: ``[B]`` long tensor of the items whose
                recommendation we are attributing.
            ke: Current explanation budget.
            kr: Current recommendation-cutoff cap. ``None`` iff
                ``depends_on_kr=False``.

        Returns:
            ``[B]`` float tensor — one scalar per user in the batch.
        """

    # ------------------------------------------------------------------
    # Default reset/update/compute. Concrete metrics inherit these.
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear streaming accumulators.

        The default registers two buffers — ``sum`` (Σ per-user values)
        and ``total`` (count of users) — sufficient for any
        per-user-mean metric. Override only if you need extra reduction
        state.
        """
        self._buffers = {}
        self._register_buffer("sum")
        self._register_buffer("total")

    def update(
        self,
        *,
        orig_scores: torch.Tensor,
        perturbed_scores: torch.Tensor,
        items_to_explain: torch.Tensor,
        ke: int,
        kr: int | None,
        user_ids: torch.Tensor,  # noqa: ARG002 (passed by evaluator; metric default ignores it)
    ) -> torch.Tensor:
        """Compute and accumulate one slice; return the per-user values.

        The evaluator captures the returned tensor and stashes
        ``(user_ids, values)`` into its per-user record buffer when
        monotonic correction or dumps are enabled. The metric itself
        never owns the record table — keeping it in the evaluator means
        DDP gather and the monotonic post-pass have one place to live.

        Returns:
            ``[B]`` per-user values (the same tensor
            :meth:`per_user_values` produced).
        """
        v = self.per_user_values(
            orig_scores=orig_scores,
            perturbed_scores=perturbed_scores,
            items_to_explain=items_to_explain,
            ke=ke,
            kr=kr,
        )
        self._add("sum", float(v.sum().item()))
        self._add("total", float(v.numel()))
        return v

    def compute(self) -> float:
        """Streaming raw mean across all users seen so far on this rank."""
        return self._get("sum") / max(self._get("total"), 1.0)

    def all_reduce(self) -> None:
        """SUM-reduce every registered float buffer across DDP ranks.

        Buffers live on CPU, but the NCCL backend can only reduce CUDA
        tensors. The collective therefore runs on the current CUDA device
        when the backend requires it, and the result is moved back to CPU
        so downstream :meth:`compute` / :meth:`_get` stays device-agnostic.

        Subclasses that need a non-SUM reduction (e.g., MAX) should
        override this method.
        """
        if not dist.is_available() or not dist.is_initialized():
            return
        reduce_device = (
            torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend() == "nccl"
            else torch.device("cpu")
        )
        for key in self._buffers:
            buf = self._buffers[key].to(reduce_device)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            self._buffers[key] = buf.cpu()


# ---------------------------------------------------------------------------
# Registry — keeps the metric set extensible without evaluator surgery.
# ---------------------------------------------------------------------------

FidelityMetricBuilder: TypeAlias = Callable[[dict], FidelityMetric]


class UnknownFidelityMetric(XaiRecsysError):
    """Raised when a fidelity-metric name is not in the registry."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = tuple(available)
        super().__init__(
            f"unknown fidelity metric {name!r}; registered: {sorted(self.available)}"
        )


_METRIC_REGISTRY: dict[str, FidelityMetricBuilder] = {}


def register_fidelity_metric(
    name: str,
) -> Callable[[FidelityMetricBuilder], FidelityMetricBuilder]:
    """Decorator: register a builder under ``name``."""

    def _decorator(builder: FidelityMetricBuilder) -> FidelityMetricBuilder:
        if name in _METRIC_REGISTRY:
            raise ValueError(f"fidelity metric {name!r} already registered")
        _METRIC_REGISTRY[name] = builder
        return builder

    return _decorator


def build_fidelity_metric(name: str, params: dict | None = None) -> FidelityMetric:
    if name not in _METRIC_REGISTRY:
        raise UnknownFidelityMetric(name, list(_METRIC_REGISTRY))
    return _METRIC_REGISTRY[name](params or {})


def list_fidelity_metrics() -> list[str]:
    return sorted(_METRIC_REGISTRY)


__all__ = [
    "FidelityMetric",
    "FidelityMetricBuilder",
    "PerturbationMode",
    "UnknownFidelityMetric",
    "build_fidelity_metric",
    "list_fidelity_metrics",
    "register_fidelity_metric",
    "target_probability",
]
