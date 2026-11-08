"""Recommender registry; importing this package registers mlp and mlp_deep."""

from __future__ import annotations

from src.recommenders.factory import (
    RecommenderBuilder,
    UnknownRecommender,
    build_recommender,
    list_recommenders,
    register_recommender,
)
from src.recommenders import history_vector as _history_vector  # noqa: F401  (side-effect imports)

__all__ = [
    "RecommenderBuilder",
    "UnknownRecommender",
    "build_recommender",
    "list_recommenders",
    "register_recommender",
]
