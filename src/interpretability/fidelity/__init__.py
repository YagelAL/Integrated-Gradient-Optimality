"""Fidelity-evaluation subpackage.

* :class:`FidelityMetric` is the streaming ABC; subclasses declare
  ``requires_modes`` + ``depends_on_kr`` so the evaluator computes exactly
  the perturbed scores each metric needs.
* :class:`Perturbation` is the strategy ABC for input perturbation.
* :class:`FidelityEvaluator` runs the single-pass loop: each
  ``(batch, explainer, ke, mode)`` invokes the recommender exactly once and
  feeds the cached scores to every active metric.
"""

from __future__ import annotations

from src.interpretability.fidelity.cache import AttributionCache, ScoreCache
from src.interpretability.fidelity.dumps import write_results_csv
from src.interpretability.fidelity.evaluator import (
    DumpSpec,
    FidelityEvalResult,
    FidelityEvaluator,
    MetricSpec,
)
from src.interpretability.fidelity.metric import (
    FidelityMetric,
    PerturbationMode,
    UnknownFidelityMetric,
    build_fidelity_metric,
    list_fidelity_metrics,
    register_fidelity_metric,
)
from src.interpretability.fidelity.monotonic import (
    PerUserRecord,
    SliceGroup,
    monotonic_aggregate,
)
from src.interpretability.fidelity.perturbations import (
    Perturbation,
    RetainPerturbation,
    TruncatePerturbation,
    build_perturbation,
)

# Side-effect imports populate the metric registry.
from src.interpretability.fidelity import metrics  # noqa: F401

__all__ = [
    "AttributionCache",
    "DumpSpec",
    "FidelityEvalResult",
    "FidelityEvaluator",
    "FidelityMetric",
    "MetricSpec",
    "PerUserRecord",
    "Perturbation",
    "PerturbationMode",
    "RetainPerturbation",
    "ScoreCache",
    "SliceGroup",
    "TruncatePerturbation",
    "UnknownFidelityMetric",
    "build_fidelity_metric",
    "build_perturbation",
    "list_fidelity_metrics",
    "monotonic_aggregate",
    "register_fidelity_metric",
    "write_results_csv",
]
