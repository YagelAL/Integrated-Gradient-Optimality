"""Each module here registers its explainer via ``@register_explainer("<name>")``."""

from __future__ import annotations

from src.interpretability.methods import ig  # noqa: F401
from src.interpretability.methods import pls  # noqa: F401

__all__: list[str] = []
