from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from src.config.schemas import RecommenderCfg
from src.exceptions import XaiRecsysError
from src.recommenders.base import BaseRecommender

RecommenderBuilder: TypeAlias = Callable[[RecommenderCfg, int], BaseRecommender]


class UnknownRecommender(XaiRecsysError):
    """Raised when a recommender name is not in the registry."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = tuple(available)
        super().__init__(
            f"unknown recommender {name!r}; registered: {sorted(self.available)}"
        )


_REGISTRY: dict[str, RecommenderBuilder] = {}


def register_recommender(name: str) -> Callable[[RecommenderBuilder], RecommenderBuilder]:
    """Decorator-registry for recommenders.

    A concrete recommender module registers itself with
    @register_recommender("<name>"), where <name> matches RecommenderCfg.name.
    build_recommender dispatches on that name. Builders take (cfg, num_items)
    and return a constructed BaseRecommender — keep them thin, just unpack
    cfg and forward to the model's __init__.

    Datasets are declarative YAML, so they don't need a Python registry;
    recommenders have nontrivial per-model code and benefit from one.
    """

    def _decorator(builder: RecommenderBuilder) -> RecommenderBuilder:
        if name in _REGISTRY:
            raise ValueError(f"recommender {name!r} already registered")
        _REGISTRY[name] = builder
        return builder

    return _decorator


def build_recommender(cfg: RecommenderCfg, num_items: int) -> BaseRecommender:
    """Resolve a registered recommender and instantiate it."""
    if cfg.name not in _REGISTRY:
        raise UnknownRecommender(cfg.name, list(_REGISTRY))
    model = _REGISTRY[cfg.name](cfg, num_items)
    if model.family is not cfg.family:
        raise ValueError(
            f"recommender {cfg.name!r} declares family {model.family.value!r} "
            f"but config requested family {cfg.family.value!r}"
        )
    return model


def list_recommenders() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "RecommenderBuilder",
    "UnknownRecommender",
    "build_recommender",
    "list_recommenders",
    "register_recommender",
]
