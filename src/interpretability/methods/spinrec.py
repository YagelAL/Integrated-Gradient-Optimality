from __future__ import annotations

from typing import List

import torch
import torch.utils.data

from src.capabilities import Family
from src.interpretability.base import BaseExplainer
from src.interpretability.inputs import Attributions, ExplainInput
from src.interpretability.registry import register_explainer
from src.interpretability.selection import CandidateAuditor, build_selection_objective
from src.recommenders.base import BaseRecommender

# ── Baseline method constants ─────────────────────────────────────────────────
_ZEROS = "zeros"
_RANDOM_USER = "sample_random_user"
_VALID_METHODS = frozenset({_ZEROS,_RANDOM_USER, })


class SPINRecExplainer(BaseExplainer):
    """SpinRec: Integrated Gradients with best-of-N baseline selection (history-vector recommenders).

    Implements Integrated Gradients (Sundararajan et al., 2017) with best-of-N
    baseline selection per explanation length (SPINRec, Eq. 3): for each
    call, ``num_baseline_samples`` random baselines are sampled from the
    training pool, plus one zero baseline, plus (stochastic method only) a
    mean candidate, the average of the sampled candidates' attribution maps,
    which costs no extra IG pass: N+2 candidates in total. The candidate that
    best satisfies the configured :class:`SelectionObjective` (default
    ``"del"``: lowest CDCG after truncating its top-``ke``) is returned.

    Dynamic with a cached candidate stage: the IG attributions are independent
    of ``ke``, so they are computed once per ``(user_ids, items_to_explain)``
    batch and reused across the evaluator's per-``ke`` calls; only the
    audition (one or two counterfactual probes per candidate, per the
    configured :class:`SelectionObjective`) re-runs per ``ke``. Probes score
    through the evaluator's own perturbations (see :class:`CandidateAuditor`),
    so the selection objective is identical to the metric that later grades
    the chosen attribution.

    IG runs in input space: it interpolates the multi-hot history vector, and the
    attribution is ``history * avg_grad * (history - baseline)``. The integration
    grid is ``linspace(0, 1, num_steps+1)`` (endpoints included; the gradient sum
    is divided by ``num_steps``).

    Args:
        recommender: Frozen history-vector recommender.
        device: Device for intermediate tensors.
        num_steps: Riemann-approximation steps along the integration path.
        method: Baseline strategy: ``"zeros"`` or ``"sample_random_user"``.
        num_baseline_samples: Number of random baselines per call. Zero baseline
            is always added on top. ``None`` auto-resolves: 0 for ``"zeros"``,
            10 for stochastic methods.
        baseline_pool_size: Batches pre-sampled from ``data_loader`` into the pool.
        data_loader: **Training** dataloader, required for ``"sample_random_user"``.
            Must contain the ``"history"`` key.
        selection_objective: Audition criterion (``del`` / ``ins`` / ``del_ke`` /
            ``ins_ke``, see :mod:`src.interpretability.selection`).
        seed: Seeds the baseline pool AND the per-batch baseline draws (via
            RNG forks keyed on the batch's users/items). Same-seed variants
            of one cell audit identical candidate streams; ``None`` keeps the
            unseeded global-RNG behavior.
    """

    name = "spinrec"
    supported_families = frozenset({Family.HISTORY_VECTOR})
    is_dynamic: bool = True  # per-ke selection; candidates cached across the ke grid

    def __init__(
        self,
        recommender: BaseRecommender,
        *,
        device: torch.device | str = "cpu",
        num_steps: int = 10,
        method: str = _RANDOM_USER,
        num_baseline_samples: int | None = None,
        baseline_pool_size: int = 10,
        data_loader: torch.utils.data.DataLoader | None = None,
        selection_objective: str = "del",
        seed: int | None = None,
    ) -> None:
        super().__init__(recommender, device=device)

        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown SPINRec baseline {method!r}. "
                f"Valid options: {sorted(_VALID_METHODS)}"
            )
        if method == _RANDOM_USER and data_loader is None:
            raise ValueError(
                "method='sample_random_user' requires data_loader (training dataloader) "
                "to be injected at runtime."
            )

        self.num_steps = num_steps
        self.method = method

        # For zeros: only zero baseline is used (1 candidate total).
        if method == _ZEROS:
            self.num_baseline_samples = 0
        elif num_baseline_samples is None:
            self.num_baseline_samples = 10
        else:
            self.num_baseline_samples = max(0, num_baseline_samples)

        self.seed = seed
        self._baseline_pool: List[torch.Tensor] | None = None

        if method == _RANDOM_USER:
            # Seeded fork: same-seed instances (e.g. the variants of
            # one cell) build identical pools, and the global RNG other
            # explainers consume is left untouched.
            if seed is not None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    self._baseline_pool = self._initialize_baseline_pool(
                        data_loader, baseline_pool_size
                    )
            else:
                self._baseline_pool = self._initialize_baseline_pool(
                    data_loader, baseline_pool_size
                )

        self._auditor = CandidateAuditor(
            self.recommender, build_selection_objective(selection_objective)
        )
        # One-slot candidate cache: the evaluator iterates ke innermost with a
        # frozen ExplainInput, so consecutive per-ke calls share one input.
        self._cand_key: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cand_cache: List[torch.Tensor] | None = None

    # ── Baseline pool ─────────────────────────────────────────────────────────

    def _initialize_baseline_pool(
        self,
        data_loader: torch.utils.data.DataLoader,
        pool_size: int,
    ) -> List[torch.Tensor]:
        """Pre-sample ``pool_size`` batches of ``batch["history"]`` (float) from the training dataloader."""
        pool: List[torch.Tensor] = []
        loader_iter = iter(data_loader)
        for _ in range(pool_size):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(data_loader)
                batch = next(loader_iter)
            pool.append(batch["history"].cpu().float())
        return pool

    # ── Core explain method ───────────────────────────────────────────────────

    def explain(self, inp: ExplainInput, *, ke: int | None = None) -> Attributions:
        """Select the best cached IG candidate for this ``ke``.

        Args:
            inp: Batch input.
            ke: Explanation length the selection is optimized for. Required
                (``is_dynamic=True``); the candidate attributions themselves
                are ``ke``-independent and cached.

        Returns:
            :class:`Attributions` with the candidate that is best under the
            selection objective for this ``ke``.
        """
        if ke is None:
            raise ValueError("spinrec is dynamic; the evaluator must pass ke")
        candidates = self._cached_candidates(inp)
        best_ig = self._auditor.select(inp, candidates, ke)

        importance = best_ig.masked_fill(
            ~inp.explainable_mask.to(best_ig.device), float("-inf")
        )
        return Attributions(importance=importance)

    def _cached_candidates(self, inp: ExplainInput) -> List[torch.Tensor]:
        """Return the N+2 candidate attributions, computing them once per input.

        The IG passes are independent of ``ke``; the cache key is the batch's
        ``(user_ids, items_to_explain)`` so a new batch or a different
        explained rank recomputes, while the evaluator's per-``ke`` calls hit.
        """
        if (
            self._cand_key is not None
            and torch.equal(self._cand_key[0], inp.user_ids)
            and torch.equal(self._cand_key[1], inp.items_to_explain)
        ):
            assert self._cand_cache is not None
            return self._cand_cache

        self.recommender.eval()
        # Seeded fork keyed on (seed, batch identity): same-seed variants draw
        # IDENTICAL baseline streams for the same users, so an iso-budget variant
        # with a smaller Nb audits a strict prefix of a larger variant's draws,
        # and candidate generation stops perturbing the global RNG (draws no
        # longer depend on explainer interleaving order).
        if self.seed is not None:
            batch_key = hash(
                (self.seed, tuple(inp.user_ids.tolist()), tuple(inp.items_to_explain.tolist()))
            ) & 0x7FFFFFFF
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(batch_key)
                candidates = self._candidates_hv(inp)
        else:
            candidates = self._candidates_hv(inp)

        self._cand_key = (inp.user_ids.clone(), inp.items_to_explain.clone())
        self._cand_cache = candidates
        return candidates

    # ── Candidate generation ──────────────────────────────────────────────────

    def _append_mean_candidate(self, candidates: List[torch.Tensor]) -> None:
        """Append the mean of the sampled candidates' attribution maps, in place.

        Costs no extra IG pass, so generation stays at N_b+1 model passes while
        selection sees N_b+2 candidates. Assumes the ``num_baseline_samples``
        sampled maps are the *leading* entries, so the zero baseline is left out
        of the average. No-op for ``method="zeros"`` (which stays a single pure-IG
        candidate) and when nothing was sampled.
        """
        if self.method == _RANDOM_USER and self.num_baseline_samples > 0:
            candidates.append(
                torch.stack(candidates[: self.num_baseline_samples]).mean(0)
            )

    def _candidates_hv(self, inp: ExplainInput) -> List[torch.Tensor]:
        """IG candidates in input space for multi-hot history vectors."""
        history = inp.history.to(self.device).clone()        # [B, D]
        targets = inp.items_to_explain.to(self.device)       # [B]
        B, D = history.shape

        candidates: List[torch.Tensor] = []
        with torch.enable_grad():
            # N random baselines from training pool
            for _ in range(self.num_baseline_samples):
                baseline = self._build_baseline(history)
                candidates.append(
                    self._compute_ig_hv(history, baseline, targets, B, D)
                )
            # Zero baseline  always included
            candidates.append(
                self._compute_ig_hv(
                    history, torch.zeros_like(history), targets, B, D
                )
            )
            self._append_mean_candidate(candidates)
        return candidates

    def _compute_ig_hv(
        self,
        history: torch.Tensor,
        baseline: torch.Tensor,
        targets: torch.Tensor,
        B: int,
        D: int,
    ) -> torch.Tensor:
        """One IG pass in input space for a single baseline. Returns [B, D].
        Uses linspace(0, 1, num_steps+1), endpoints included; the gradient
        sum is divided by num_steps.
        Attribution formula: history * avg_grad * (history - baseline).
        """
        alphas = torch.linspace(0.0, 1.0, self.num_steps + 1, device=self.device)
        arange_B = torch.arange(B, device=self.device)
        grad_sum = torch.zeros(B, D, device=self.device)

        for alpha in alphas:
            x = (baseline + alpha * (history - baseline)).detach().requires_grad_(True)
            scores = self.recommender(x)
            scores[arange_B, targets].sum().backward()
            grad_sum = grad_sum + x.grad.detach()

        avg_grad = grad_sum / self.num_steps
        return history * avg_grad * (history - baseline)   # [B, D]

    # ── Baseline construction ─────────────────────────────────────────────────

    def _build_baseline(self, history: torch.Tensor) -> torch.Tensor:
        """Build one random baseline with shape [B, D]."""
        B, D = history.shape

        if self.method == _ZEROS:
            return torch.zeros(B, D, dtype=torch.float32, device=history.device)

        # _RANDOM_USER
        assert self._baseline_pool is not None
        pool_idx = int(torch.randint(len(self._baseline_pool), (1,)).item())
        pool_batch = self._baseline_pool[pool_idx]          # [pool_B, D]
        pool_B = pool_batch.size(0)
        if pool_B >= B:
            baseline = pool_batch[:B]
        else:
            repeats = (B // pool_B) + 1
            baseline = pool_batch.repeat(repeats, 1)[:B]
        return baseline.to(history.device)


# ── Registry builder ──────────────────────────────────────────────────────────

@register_explainer("spinrec")
def _build_spinrec(
    recommender: BaseRecommender,
    params: dict,
    device: torch.device,
) -> SPINRecExplainer:
    return SPINRecExplainer(
        recommender,
        device=device,
        num_steps=params.get("num_steps", 10),
        method=params.get("method", _RANDOM_USER),
        num_baseline_samples=params.get("num_baseline_samples"),
        baseline_pool_size=params.get("baseline_pool_size", 10),
        data_loader=params.get("data_loader"),
        selection_objective=params.get("selection_objective", "del"),
        seed=params.get("seed"),
    )


__all__ = ["SPINRecExplainer"]
