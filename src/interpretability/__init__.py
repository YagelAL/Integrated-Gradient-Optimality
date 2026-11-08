from __future__ import annotations

from src.interpretability.base import BaseExplainer
from src.interpretability.inputs import Attributions, ExplainInput
from src.interpretability.registry import (
    UnknownExplainer,
    build_explainer,
    list_explainers,
    register_explainer,
)


# Importing ``methods`` registers the ``ig`` and ``pls`` explainers.
from src.interpretability import methods  # noqa: F401

__all__ = [
    "Attributions",
    "BaseExplainer",
    "ExplainInput",
    "UnknownExplainer",
    "build_explainer",
    "list_explainers",
    "register_explainer",
]
