"""Raw target-probability probes for `synth_gt_audit.py` (same path as the PLS objective)."""

from __future__ import annotations

import torch

from src.interpretability.fidelity.metric import target_probability


def set_probability(auditor, row_inp, positions, ke: int) -> float:
    """p(target) after Truncate (del_ke) / Retain (ins_ke) of `positions` on a [1, D] row."""
    imp = torch.full(row_inp.history.shape, float("-inf"), device=row_inp.history.device)
    imp[0, sorted(positions)] = 1.0
    ct, cr = auditor.cdcg_probes(row_inp, imp, ke)
    return float((ct if ct is not None else cr)[0])


def original_probability(model, scores: torch.Tensor, target: int) -> float:
    """p(target) on the unperturbed history, from `full_sort_predict` scores [1, num_items+1]."""
    items = torch.tensor([target], device=scores.device)
    return float(target_probability(scores, items, model.score_activation)[0])
