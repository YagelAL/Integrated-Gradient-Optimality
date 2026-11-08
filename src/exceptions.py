from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.capabilities import Family


class XaiRecsysError(Exception):
    """Base class for all domain errors raised by this package."""


class IncompatibleExplainer(XaiRecsysError):
    """An explainer cannot operate on the given recommender's family."""

    def __init__(
        self,
        explainer_type: str,
        recommender_type: str,
        *,
        family_mismatch: tuple[Family, Iterable[Family]],
    ) -> None:
        self.explainer_type = explainer_type
        self.recommender_type = recommender_type
        self.family_mismatch = family_mismatch
        got, supported = family_mismatch
        allowed = ", ".join(f.value for f in supported)
        super().__init__(
            f"{explainer_type!r} is incompatible with {recommender_type!r}; "
            f"family {got.value!r} not in supported families {{{allowed}}}"
        )


class InvalidPerturbation(XaiRecsysError):
    """A perturbation cannot be applied to the given input."""

    def __init__(self, perturbation_type: str, reason: str) -> None:
        self.perturbation_type = perturbation_type
        self.reason = reason
        super().__init__(f"perturbation {perturbation_type!r} cannot be applied: {reason}")


class ArtifactNotFound(XaiRecsysError):
    """A W&B artifact could not be resolved."""

    def __init__(self, artifact_path: str, reason: str = "") -> None:
        self.artifact_path = artifact_path
        self.reason = reason
        msg = f"artifact not found: {artifact_path!r}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


__all__ = [
    "ArtifactNotFound",
    "IncompatibleExplainer",
    "InvalidPerturbation",
    "XaiRecsysError",
]
