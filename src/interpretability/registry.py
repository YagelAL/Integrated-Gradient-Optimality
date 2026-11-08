from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import torch

from src.exceptions import XaiRecsysError
from src.interpretability.base import BaseExplainer
from src.recommenders.base import BaseRecommender

ExplainerBuilder: TypeAlias = Callable[[BaseRecommender, dict, torch.device], BaseExplainer]


class UnknownExplainer(XaiRecsysError):
    """Raised when an explainer name is not in the registry."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = tuple(available)
        super().__init__(
            f"unknown explainer {name!r}; registered: {sorted(self.available)}"
        )


_REGISTRY: dict[str, ExplainerBuilder] = {}


def register_explainer(name: str) -> Callable[[ExplainerBuilder], ExplainerBuilder]:
    """Decorator: register a builder under ``name``.

    A concrete explainer module exposes a tiny ``_build_<name>(recommender,
    params, device) -> BaseExplainer`` function decorated with
    ``@register_explainer("<name>")``. The Hydra explainer config's
    ``name`` field selects which builder runs.

    Mirrors :func:`src.recommenders.factory.register_recommender` so the
    two factories feel the same to callers.
    """

    def _decorator(builder: ExplainerBuilder) -> ExplainerBuilder:
        if name in _REGISTRY:
            raise ValueError(f"explainer {name!r} already registered")
        _REGISTRY[name] = builder
        return builder

    return _decorator


def build_explainer(
    name: str,
    recommender: BaseRecommender,
    *,
    params: dict | None = None,
    device: torch.device | str = "cpu",
) -> BaseExplainer:
    """Resolve a registered explainer and instantiate it.

    Args:
        name: Registry key (matches ``ExplainerCfg.name``).
        recommender: Frozen, weights-loaded recommender to attribute.
        params: Free-form per-explainer hyperparameters (e.g.
            ``num_steps`` for IG). Builders pull what they need.
        device: Forwarded to the explainer's ``__init__``.

    Raises:
        UnknownExplainer: No builder is registered under ``name``.
    """
    if name not in _REGISTRY:
        raise UnknownExplainer(name, list(_REGISTRY))
    return _REGISTRY[name](recommender, params or {}, torch.device(device))


def list_explainers() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "ExplainerBuilder",
    "UnknownExplainer",
    "build_explainer",
    "list_explainers",
    "register_explainer",
]
