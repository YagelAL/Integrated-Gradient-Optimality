from __future__ import annotations

from dataclasses import replace as dc_replace
from itertools import combinations
from math import ceil, comb

import numpy as np
import torch

from src.capabilities import Family
from src.interpretability.base import BaseExplainer
from src.interpretability.inputs import Attributions, ExplainInput
from src.interpretability.methods._adaptive_batch import run_adaptive
from src.interpretability.registry import register_explainer
from src.interpretability.selection import (
    CandidateAuditor,
    build_selection_objective,
    row_input,
    tile_input,
)
from src.recommenders.base import BaseRecommender


def _assign_importance(
    importance: torch.Tensor, b: int, subset: tuple[int, ...], device: torch.device
) -> None:
    """Mark selected positions with uniform score 1.0; all others stay -inf."""
    idx = torch.tensor(subset, dtype=torch.long, device=device)
    importance[b, idx] = 1.0


def _parallel_local_search(
    objective,
    d: int,
    ke: int,
    *,
    num_restarts: int,
    eval_budget: int,
    temp_decay: float,
    temp_init: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    space = comb(d, ke)

    # Small space ⇒ enumerate exactly (cheap, and avoids SA's coverage gaps).
    if space <= eval_budget:
        combos = list(combinations(range(d), ke))
        masks = np.zeros((len(combos), d), dtype=np.float64)
        for i, c in enumerate(combos):
            masks[i, list(c)] = 1.0
        return masks[int(np.argmin(objective(masks)))]

    restarts = max(1, min(num_restarts, space))
    n_iters = max(1, ceil(eval_budget / restarts))

    # Distinct initial exact-ke masks (one per chain) for good coverage.
    seen: set[tuple[int, ...]] = set()
    X = np.zeros((restarts, d), dtype=np.float64)
    for i in range(restarts):
        while True:
            c = tuple(sorted(rng.choice(d, size=ke, replace=False).tolist()))
            if c not in seen:
                seen.add(c)
                break
        X[i, list(c)] = 1.0

    obj = objective(X)  # [restarts]
    best_i = int(np.argmin(obj))
    best_x = X[best_i].copy()
    best_obj = float(obj[best_i])

    evals = restarts
    T = temp_init
    with np.errstate(all="ignore"):  # exp overflow / divide-by-zero at T=0 is fine
        for _ in range(1, n_iters):
            T *= temp_decay
            # Trim the final round so evals lands exactly on eval_budget
            # (matters when eval_budget is not a multiple of restarts).
            m = min(restarts, eval_budget - evals)
            # Propose: flip one 1->0 and one 0->1 per chain (stays exact-ke).
            Xn = X[:m].copy()
            for i in range(m):
                ones = np.flatnonzero(X[i])
                zeros = np.flatnonzero(X[i] == 0.0)
                Xn[i, rng.choice(ones)] = 0.0
                Xn[i, rng.choice(zeros)] = 1.0

            objn = objective(Xn)
            for i in range(m):
                if objn[i] < obj[i]:
                    X[i], obj[i] = Xn[i], objn[i]
                elif T > 0.0 and rng.random() < np.exp((obj[i] - objn[i]) / T):
                    X[i], obj[i] = Xn[i], objn[i]

            bi = int(np.argmin(objn))
            if objn[bi] < best_obj:
                best_obj, best_x = float(objn[bi]), Xn[bi].copy()

            evals += m
            if evals >= eval_budget:
                break
    return best_x


class ParallelLocalSearchExplainer(BaseExplainer):
    """PLS: search for the best ``ke``-subset of the history under a selection objective.

    Parallel simulated-annealing chains minimize the negated
    :class:`SelectionObjective` score, probed through :class:`CandidateAuditor`
    (the evaluator's own perturbations). When ``comb(d, ke) <= eval_budget``
    the search enumerates every subset exactly. SA acceptance at T>0 is not
    scale-invariant, so ``temp_init`` is given in objective units.
    """

    name = "pls"
    supported_families = frozenset({Family.HISTORY_VECTOR})
    is_dynamic = True

    def __init__(
        self,
        recommender: BaseRecommender,
        *,
        device: torch.device | str = "cpu",
        selection_objective: str = "del",
        num_restarts: int = 10,
        eval_budget: int = 200,
        temp_decay: float = 0.0,
        temp_init: float = 0.0,
        max_forward_batch: int = 512,
        seed: int | None = 42,
    ) -> None:
        super().__init__(recommender, device=device)
        self._auditor = CandidateAuditor(
            self.recommender, build_selection_objective(selection_objective)
        )
        self.num_restarts = num_restarts
        self.eval_budget = eval_budget
        if temp_decay > 0.0 and temp_init <= 0.0:
            raise ValueError("temp_init must be > 0 when temp_decay > 0 (annealing on)")
        self.temp_decay = temp_decay
        self.temp_init = temp_init
        self.max_forward_batch = max_forward_batch
        self.seed = seed
        self._call_count = 0
        # Adaptive chunk size and success streak (see _adaptive_batch.run_adaptive).
        self._batch_size = max_forward_batch
        self._streak = 0

    def explain(self, inp: ExplainInput, *, ke: int | None = None) -> Attributions:
        if ke is None:
            raise ValueError("pls is dynamic; the evaluator must pass ke")
        if self.seed is not None:
            rng = np.random.default_rng(self.seed + self._call_count)
            self._call_count += 1
        else:
            rng = np.random.default_rng()

        self.recommender.eval()
        return self._explain_hv(inp, ke, rng)

    @torch.no_grad()
    def _explain_hv(self, inp: ExplainInput, ke: int, rng: np.random.Generator) -> Attributions:
        assert inp.history is not None
        history = inp.history.to(self.device)
        explainable = inp.explainable_mask.to(self.device)
        importance = torch.full(
            explainable.shape, float("-inf"), dtype=torch.float32, device=self.device
        )

        for b in range(inp.batch_size):
            target = int(inp.items_to_explain[b])
            cands = explainable[b].nonzero(as_tuple=False).squeeze(1).tolist()
            cands = [c for c in cands if c != target]  # target is never removable
            row = row_input(inp, b)
            hist_b = history[b : b + 1].clone()
            hist_b[0, target] = 0.0  # target zeroed before the search
            row = dc_replace(row, history=hist_b)
            subset = self._search_row_hv(row, cands, ke, rng)
            if subset:
                _assign_importance(importance, b, subset, self.device)
        return Attributions(importance=importance)

    def _search_row_hv(
        self,
        row: ExplainInput,
        cands: list[int],
        ke: int,
        rng: np.random.Generator,
    ) -> tuple[int, ...]:
        d = len(cands)
        ke_eff = min(ke, d)
        if ke_eff <= 0:
            return ()
        if ke_eff == d:  # trivial: remove every candidate, no search
            return tuple(cands)

        cand_t = torch.tensor(cands, dtype=torch.long, device=self.device)

        def objective(masks: np.ndarray) -> np.ndarray:
            return self._objective_row(masks, row, cand_t, ke_eff)

        best = _parallel_local_search(
            objective, d, ke_eff,
            num_restarts=self.num_restarts, eval_budget=self.eval_budget,
            temp_decay=self.temp_decay, temp_init=self.temp_init, rng=rng,
        )
        removed = np.flatnonzero(best)
        return tuple(cands[j] for j in removed)

    @torch.no_grad()
    def _objective_row(
        self,
        masks: np.ndarray,
        row: ExplainInput,
        cand_t: torch.Tensor,
        ke_eff: int,
    ) -> np.ndarray:
        """Evaluate ``[r, d]`` removal masks via the shared audit probes.

        Marks each mask's chosen columns as the importance top-``ke``
        and scores through :class:`CandidateAuditor`; returns ``[r]`` negated
        objective values (the SA loop minimizes).
        """
        r = masks.shape[0]
        m = torch.from_numpy(masks).to(self.device) > 0  # [r, d] bool
        width = row.history.shape[1]

        def run_chunk(start: int, stop: int) -> np.ndarray:
            chunk = m[start:stop]  # [c, d]
            c = chunk.shape[0]
            rep = tile_input(row, c)
            imp = torch.full((c, width), float("-inf"), device=self.device)
            imp[:, cand_t] = torch.where(
                chunk,
                torch.ones((), device=self.device),
                torch.full((), float("-inf"), device=self.device),
            )
            ct, cr = self._auditor.cdcg_probes(rep, imp, ke_eff)
            val = self._auditor.objective.score(ct, cr)
            return (-val).cpu().numpy()

        return run_adaptive(
            r,
            lambda: self._batch_size,
            lambda bs: setattr(self, "_batch_size", bs),
            self.max_forward_batch,
            run_chunk,
            lambda: self._streak,
            lambda s: setattr(self, "_streak", s),
        )


@register_explainer("pls")
def _build_pls(
    recommender: BaseRecommender,
    params: dict,
    device: torch.device,
) -> ParallelLocalSearchExplainer:
    return ParallelLocalSearchExplainer(
        recommender,
        device=device,
        selection_objective=params.get("selection_objective", "del"),
        num_restarts=params.get("num_restarts", 10),
        eval_budget=params.get("eval_budget", 200),
        temp_decay=params.get("temp_decay", 0.0),
        temp_init=params.get("temp_init", 0.0),
        max_forward_batch=params.get("max_forward_batch", 512),
        seed=params.get("seed", 42),
    )


__all__ = ["ParallelLocalSearchExplainer"]
