"""Concrete fidelity metrics.

Each module side-effect-registers a builder with the central metric
registry via ``@register_fidelity_metric("<name>")``. Adding a metric is
one new file in this directory.
"""

from __future__ import annotations

from src.interpretability.fidelity.metrics import (
    del_ke,
    ins_ke,
)

__all__ = ["del_ke", "ins_ke"]
