from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.capabilities import Family
from src.interpretability.base import BaseExplainer
from src.interpretability.fidelity.cache import AttributionCache, ScoreCache
from src.interpretability.fidelity.dumps import write_results_csv
from src.interpretability.fidelity.metric import (
    FidelityMetric,
    PerturbationMode,
    build_fidelity_metric,
)
from src.interpretability.fidelity.monotonic import (
    PerUserRecord,
    SliceGroup,
    monotonic_aggregate,
    monotonic_per_user_values,
)
from src.interpretability.fidelity.perturbations import Perturbation, build_perturbation
from src.interpretability.inputs import ExplainInput
from src.recommenders.base import BaseRecommender
from src.trainers.metrics import _mask_pad


@dataclass(frozen=True)
class MetricSpec:
    """One line item from ``configs/fidelity/*.yaml`` ``metrics:`` block.

    Carries the registry name + free-form params dict the metric's builder
    consumes. The evaluator clones one metric instance per slice from this
    spec, so the same spec drives every ``(explainer, ke, kr, mode, rank)``
    cell.

    Attributes:
        name: Registry key (matches ``@register_fidelity_metric``).
        params: Builder kwargs, threaded to ``build_fidelity_metric``.
    """

    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class DumpSpec:
    """Where to dump the per-user results CSV when artifact saving is enabled.

    Attributes:
        save_artifacts: Write the per-user results CSV under
            ``out_dir / details``.
        out_dir: Root directory for the dumps (typically
            ``hydra.run.dir``). One run = one out_dir.
        dataset: Dataset name (filename component for the dumps).
        recommender: Recommender name (filename component).
    """

    save_artifacts: bool = False
    out_dir: Path = field(default_factory=lambda: Path.cwd())
    dataset: str = "unknown"
    recommender: str = "unknown"


@dataclass(frozen=True)
class FidelityEvalResult:
    """Aggregated outputs from one :meth:`FidelityEvaluator.evaluate` call.

    Attributes:
        scalars: Flat ``"<explainer>/<metric>/rank=<r>/mode=<m>/ke=<k>[/kr=<kr>]"``
            map of every metric × explainer × rank × ke × mode (× kr)
            combination. When ``monotonic=True``, each slice key for a
            ``supports_monotonic=True`` metric carries the monotonic-
            corrected value plus a sibling ``<key>_raw`` key with the
            uncorrected streaming mean; a ``supports_monotonic=False``
            metric's key carries the raw streaming mean directly (no
            cummin/cummax, no ``_raw`` sibling).
            Suitable for direct W&B logging on rank 0.
        recommender_call_count: Number of ``full_sort_predict`` invocations
            during evaluation. The evaluator asserts this equals
            ``n_batches * (1 + |explainers| * |item_ranks| * |ke_grid| * |active_modes|)``
            — the locked-in invariant of the design.
        dump_paths: Paths of the files written by the dump pass; empty when
            ``save_artifacts`` is ``False``.
    """

    scalars: dict[str, float]
    recommender_call_count: int
    dump_paths: list[Path] = field(default_factory=list)


class FidelityEvaluator:
    """Single-pass, DDP-native fidelity evaluator.

    For every batch from ``test_loader`` the evaluator:

    1. Calls ``recommender.full_sort_predict`` **once** to get
       ``orig_scores`` and the dataloader labels.
    2. Resolves ``items_to_explain`` per ``item_rank`` from ``orig_scores``.
    3. Computes static-explainer attributions **once** per
       ``(batch, item_rank)``.
    4. For each ``ke``, computes dynamic-explainer attributions per
       ``(batch, item_rank, ke)``; for each active perturbation mode,
       builds the perturbed input and runs ``full_sort_predict`` **once**
       per ``(explainer, ke, mode)``.
    5. Feeds the cached perturbed scores to every metric *slice* whose
       ``requires_modes`` includes that mode, iterating over ``kr_grid``
       only for metric specs with ``depends_on_kr=True``. The per-user
       value tensor each metric returns is captured into the per-user
       record buffer when monotonic correction or artifact dumps are on.

    The recommender is never invoked twice for the same input. The
    evaluator records its own call count and asserts the invariant at
    end-of-eval so any regression surfaces immediately.

    Monotonic correction + opt-in dumps:

    * When ``monotonic=True``, each per-user vector from
      :meth:`FidelityMetric.update` is appended (with its slice group
      and ``ke``) into ``self._records``. After streaming finishes the
      records are gathered to rank 0 via ``dist.gather_object``; only
      records from ``supports_monotonic=True`` metrics are fed to
      :func:`monotonic_aggregate` (grouped by each metric's
      ``monotonic_reduction`` — cummin or cummax), which produces the
      headline (monotonic) and ``_raw`` (uncorrected) scalars together.
      Metrics that opt out keep the raw streaming mean as their headline.
    * When ``dump.save_artifacts=True``, the same gathered records are
      written to a CSV under ``dump.out_dir / details``.

    DDP: streaming per-rank state syncs via ``metric.all_reduce()`` (sum
    buffers); per-user records sync via ``dist.gather_object``.
    """

    def __init__(
        self,
        *,
        recommender: BaseRecommender,
        explainers: Sequence[BaseExplainer],
        metric_specs: Sequence[MetricSpec],
        family: Family,
        ke_grid: Sequence[int],
        kr_grid: Sequence[int],
        item_ranks: Sequence[int] = (0,),
        device: torch.device | str = "cpu",
        monotonic: bool = True,
        dump: DumpSpec | None = None,
    ) -> None:
        self.recommender = (
            recommender.module if hasattr(recommender, "module") else recommender
        )
        self.explainers = list(explainers)
        self.metric_specs = list(metric_specs)
        self.family = family
        self.ke_grid = list(ke_grid)
        self.kr_grid = list(kr_grid)
        self.item_ranks = list(item_ranks)
        self.device = torch.device(device)
        self.monotonic = monotonic
        self.dump = dump or DumpSpec()

        # Validation: every explainer must support the recommender's family.
        for e in self.explainers:
            if family not in e.supported_families:
                raise ValueError(
                    f"explainer {e.name!r} does not support family {family.value!r}"
                )

        # Resolve the union of perturbation modes any spec requires by
        # instantiating one probe per spec. `score_mode` is injected from the
        # recommender so del_ke / ins_ke know its probability semantics; an
        # explicit config value still wins.
        self._metric_params: dict[str, dict] = {
            spec.name: {
                "score_mode": self.recommender.score_activation,
                **spec.params,
            }
            for spec in self.metric_specs
        }
        probes: dict[str, FidelityMetric] = {
            spec.name: build_fidelity_metric(spec.name, self._metric_params[spec.name])
            for spec in self.metric_specs
        }
        active_mode_set: set[PerturbationMode] = set()
        for probe in probes.values():
            active_mode_set |= set(probe.requires_modes)
        self.active_modes: list[PerturbationMode] = sorted(
            active_mode_set, key=lambda m: m.value
        )
        self._perturbations: dict[PerturbationMode, Perturbation] = {
            mode: build_perturbation(mode, self.recommender)
            for mode in self.active_modes
        }

        # Per-spec defaults — captured here so the slice loop can read
        # supports_monotonic / monotonic_reduction without re-probing.
        self._spec_supports_monotonic: dict[str, bool] = {
            spec.name: probes[spec.name].supports_monotonic
            for spec in self.metric_specs
        }
        self._spec_reduction: dict[str, str] = {
            spec.name: probes[spec.name].monotonic_reduction
            for spec in self.metric_specs
        }

        # Pre-build one metric instance per slice. Slice key encodes
        # everything that distinguishes a per-W&B-scalar cell.
        self._slices: dict[str, FidelityMetric] = {}
        self._slice_groups: dict[str, SliceGroup] = {}
        self._slice_ke: dict[str, int] = {}
        self._slice_explainer: dict[str, str] = {}
        for spec in self.metric_specs:
            probe = probes[spec.name]
            for explainer in self.explainers:
                for item_rank in self.item_ranks:
                    for ke in self.ke_grid:
                        for mode in probe.requires_modes:
                            if mode not in self._perturbations:
                                continue
                            kr_iter = self.kr_grid if probe.depends_on_kr else (None,)
                            for kr in kr_iter:
                                key = self._slice_key(
                                    explainer.name, spec.name, item_rank, mode, ke, kr
                                )
                                self._slices[key] = build_fidelity_metric(
                                    spec.name, self._metric_params[spec.name]
                                )
                                self._slice_groups[key] = SliceGroup(
                                    metric=spec.name,
                                    explainer=explainer.name,
                                    item_rank=item_rank,
                                    mode=mode,
                                    kr=kr,
                                )
                                self._slice_ke[key] = ke
                                self._slice_explainer[key] = explainer.name

        # Per-user records: list of PerUserRecord; only populated when
        # monotonic correction or artifact dumps are enabled.
        self._records: list[PerUserRecord] = []
        self._call_count = 0

    @property
    def _collect_records(self) -> bool:
        """Whether the per-user record buffer should be populated."""
        return self.monotonic or self.dump.save_artifacts

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> FidelityEvalResult:
        """Run the single-pass loop over ``test_loader``."""
        attr_cache = AttributionCache()
        score_cache = ScoreCache()
        self._records = []
        self._call_count = 0

        n_batches = 0
        for batch in tqdm(
            test_loader,
            total=len(test_loader),
            desc="fidelity eval",
            disable=not self._is_rank_zero(),
            dynamic_ncols=True,
            mininterval=30,
        ):
            batch = self._to_device(batch)
            attr_cache.clear()
            score_cache.clear()
            n_batches += 1

            # 1) One recommender call per batch on the original input.
            orig_scores, _ = self._predict(batch)

            # 2) Resolve items_to_explain per item_rank.
            ranks_to_items = self._resolve_items_to_explain(orig_scores)

            # 3) For each item_rank, do the static-attribution pass, then
            #    the ke × mode loop.
            for item_rank in self.item_ranks:
                items_to_explain = ranks_to_items[item_rank]
                inp = ExplainInput.from_batch(
                    batch,
                    family=self.family,
                    items_to_explain=items_to_explain,
                )

                # Static-explainer attributions once per (batch, item_rank).
                for explainer in self.explainers:
                    if explainer.is_dynamic:
                        continue
                    attr_key = self._attr_key(explainer.name, item_rank)
                    if attr_cache.get_static(attr_key) is None:
                        attr_cache.put_static(attr_key, explainer.explain(inp))

                for ke in self.ke_grid:
                    # Dynamic-explainer attributions per (batch, item_rank, ke).
                    for explainer in self.explainers:
                        if not explainer.is_dynamic:
                            continue
                        attrs = explainer.explain(inp, ke=ke)
                        attr_cache.put_dynamic(
                            self._attr_key(explainer.name, item_rank), ke, attrs
                        )

                    # One predict per (explainer, ke, mode).
                    for mode in self.active_modes:
                        for explainer in self.explainers:
                            attrs = (
                                attr_cache.get_dynamic(
                                    self._attr_key(explainer.name, item_rank), ke
                                )
                                if explainer.is_dynamic
                                else attr_cache.get_static(
                                    self._attr_key(explainer.name, item_rank)
                                )
                            )
                            assert attrs is not None
                            perturbed_inp = self._perturbations[mode].apply(
                                inp, attrs, ke
                            )
                            ps, _ = self._predict(perturbed_inp.to_recommender_batch())
                            # Rank against the ORIGINAL candidate set: items the
                            # perturbation removed must not re-enter the
                            # ranking. The target is never in the history, so
                            # del_ke / ins_ke are unaffected.
                            ps = ps.masked_fill(
                                batch["history"].bool(), float("-inf")
                            )
                            score_cache.put(
                                explainer=self._score_key(
                                    explainer.name, item_rank
                                ),
                                ke=ke,
                                mode=mode,
                                scores=ps,
                            )

                    # Feed every slice metric for this (item_rank, ke).
                    for slice_key, metric in self._slices.items():
                        if self._slice_ke[slice_key] != ke:
                            continue
                        group = self._slice_groups[slice_key]
                        if group.item_rank != item_rank:
                            continue
                        ps = score_cache.get(
                            explainer=self._score_key(group.explainer, item_rank),
                            ke=ke,
                            mode=group.mode,
                        )
                        v = metric.update(
                            orig_scores=orig_scores,
                            perturbed_scores=ps,
                            items_to_explain=items_to_explain,
                            ke=ke,
                            kr=group.kr,
                            user_ids=inp.user_ids,
                        )
                        if self._collect_records:
                            self._append_records(group, ke, inp.user_ids, v)

        # 4) DDP sync per slice (raw streaming scalars).
        for metric in self._slices.values():
            metric.all_reduce()

        # 5) Build the raw scalar map (always).
        raw_scalars: dict[str, float] = {
            key: metric.compute() for key, metric in self._slices.items()
        }

        # 6) Monotonic post-pass on rank 0 (if enabled).
        scalars: dict[str, float] = {}
        if self.monotonic:
            all_records = self._gather_records()
            if self._is_rank_zero():
                for reduction, recs in self._records_by_reduction(all_records).items():
                    scalars.update(
                        monotonic_aggregate(recs, ke_grid=self.ke_grid, reduction=reduction)
                    )
                # Metrics that don't support monotonic correction: headline
                # is the raw streaming mean directly, no cummin/cummax and
                # no separate _raw sibling (it would just duplicate the headline).
                for key, group in self._slice_groups.items():
                    if not self._spec_supports_monotonic[group.metric]:
                        scalars[key] = raw_scalars[key]
            else:
                all_records = []
        else:
            all_records = []
            if self._is_rank_zero():
                # Without monotonic, the raw scalar is the headline (no _raw suffix).
                scalars.update(raw_scalars)

        # 7) Optional disk dumps (rank 0 only).
        dump_paths: list[Path] = []
        if self._is_rank_zero() and self.dump.save_artifacts:
            details_path = (
                self.dump.out_dir
                / "details"
                / f"{self.dump.dataset}_{self.dump.recommender}.csv"
            )
            # Provide the per-user monotonic table so the CSV's
            # monotonic_value column is populated; the per-slice means
            # already in `scalars` won't help row-level dumps.
            mono_table: dict[tuple[SliceGroup, int, int], float] | None = None
            if self.monotonic:
                mono_table = {}
                for reduction, recs in self._records_by_reduction(all_records).items():
                    mono_table.update(
                        monotonic_per_user_values(
                            recs, ke_grid=self.ke_grid, reduction=reduction
                        )
                    )
            written = write_results_csv(
                all_records, path=details_path, monotonic_values=mono_table
            )
            dump_paths.append(written)

        # 8) Recommender-call invariant.
        expected_per_batch = 1 + (
            len(self.explainers)
            * len(self.item_ranks)
            * len(self.ke_grid)
            * len(self.active_modes)
        )
        expected = expected_per_batch * n_batches
        if self._call_count != expected:
            raise RuntimeError(
                "recommender-call invariant violated: "
                f"got {self._call_count} calls, expected {expected} "
                f"({expected_per_batch} per batch × {n_batches} batches)"
            )

        return FidelityEvalResult(
            scalars=scalars,
            recommender_call_count=self._call_count,
            dump_paths=dump_paths,
        )

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One call to ``full_sort_predict``; bumps the call counter.

        Going through this helper is what makes the per-batch invariant
        observable. Direct calls to ``full_sort_predict`` from outside
        this method would silently break the invariant assertion.
        """
        scores, labels = self.recommender.full_sort_predict(batch)
        self._call_count += 1
        return scores, labels

    def _resolve_items_to_explain(
        self, orig_scores: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        scores = _mask_pad(orig_scores)
        max_rank = max(self.item_ranks) + 1
        topk = scores.topk(max_rank, dim=-1).indices  # [B, max_rank]
        return {r: topk[:, r] for r in self.item_ranks}

    def _append_records(
        self,
        group: SliceGroup,
        ke: int,
        user_ids: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Append one batch's per-user values into the records buffer."""
        uids = user_ids.detach().cpu().tolist()
        vals = values.detach().to(torch.float64).cpu().tolist()
        for uid, val in zip(uids, vals, strict=True):
            self._records.append(
                PerUserRecord(group=group, ke=ke, user_id=int(uid), value=float(val))
            )

    def _gather_records(self) -> list[PerUserRecord]:
        """Gather per-user records to rank 0 via ``dist.gather_object``."""
        if not (dist.is_available() and dist.is_initialized()):
            return self._records
        world = dist.get_world_size()
        gathered: list[list[PerUserRecord]] | None = (
            [[] for _ in range(world)] if dist.get_rank() == 0 else None
        )
        dist.gather_object(self._records, gathered, dst=0)
        if dist.get_rank() != 0:
            return []
        assert gathered is not None
        flat: list[PerUserRecord] = []
        for shard in gathered:
            flat.extend(shard)
        return flat

    def _records_by_reduction(
        self, records: list[PerUserRecord]
    ) -> dict[str, list[PerUserRecord]]:
        """Group monotonic-eligible records by their metric's reduction.

        ``monotonic_aggregate``/``monotonic_per_user_values`` apply one
        reduction to every record in a single call. Most metrics want
        ``cummin`` (removing more should only hurt), but ins_ke wants
        ``cummax`` (retaining more should only help) — mixing them in one
        call would cummin ins_ke in the wrong direction. Split by
        reduction here and call the aggregator once per group instead.
        Records from ``supports_monotonic=False`` metrics are dropped.
        """
        grouped: dict[str, list[PerUserRecord]] = {}
        for rec in records:
            if not self._spec_supports_monotonic[rec.group.metric]:
                continue
            reduction = self._spec_reduction[rec.group.metric]
            grouped.setdefault(reduction, []).append(rec)
        return grouped

    @staticmethod
    def _is_rank_zero() -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return True
        return dist.get_rank() == 0

    @staticmethod
    def _attr_key(explainer: str, item_rank: int) -> str:
        return f"{explainer}@rank={item_rank}"

    @staticmethod
    def _score_key(explainer: str, item_rank: int) -> str:
        return f"{explainer}@rank={item_rank}"

    @staticmethod
    def _slice_key(
        explainer: str,
        metric: str,
        item_rank: int,
        mode: PerturbationMode,
        ke: int,
        kr: int | None,
    ) -> str:
        suffix = f"/kr={kr}" if kr is not None else ""
        return (
            f"{explainer}/{metric}/rank={item_rank}/mode={mode.value}/ke={ke}{suffix}"
        )

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }


__all__ = ["DumpSpec", "FidelityEvalResult", "FidelityEvaluator", "MetricSpec"]
