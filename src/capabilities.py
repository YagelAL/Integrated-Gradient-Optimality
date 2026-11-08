from __future__ import annotations

from enum import Enum, unique


@unique
class Family(str, Enum):
    """The class of recommender, defined by the shape of its input.

    Attributes:
        HISTORY_VECTOR: Takes a multi-hot vector ``[B, num_items + 1]`` over
            the item catalog (1 where the user interacted, 0 otherwise).
    """

    HISTORY_VECTOR = "history_vector"


__all__ = ["Family"]
