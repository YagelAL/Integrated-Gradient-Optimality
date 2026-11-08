"""Ground-truth / set-recovery audit for the synthetic experiment.

For each test user's rank-0 explained recommendation `v`, computes the
planted ground truth GT_5 (top-5 of v's planted drivers present in the
user's eval-time history, ranked by weight; all planted weights are
positive) and scores each requested explainer's top-ke removal set against
it: precision@ke, recall@ke, Jaccard@ke, exact_hit@ke, for ke=1..5.

`precision_model` scores the same set against the model's optimal k-subset
(GT_model): for `mlp`, the top-k of its closed-form effective item weights;
otherwise the set chosen by an exact-enumeration `pls` in the same pass.

Each explainer's ke-selection is extracted with
`Perturbation._select_positions`, i.e. exactly the set the fidelity
evaluator perturbs. Headline subset: rank 0, users with |G|>=5 (the
exclusion rate is printed).

Example::

    python scripts/synth_gt_audit.py \\
        +final=mlp_synth budget=q50 \\
        ++audit_ground_truth_path=Data/Raw_Data/synth_ground_truth.json \\
        ++audit_explainers=ig,pls
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import src.interpretability  # noqa: E402  (registers explainers + metrics)
import src.recommenders  # noqa: E402  (registers recommender classes, incl. mlp_deep)
from scripts._audit_setup import load_model_and_context  # noqa: E402
from src.capabilities import Family  # noqa: E402
from src.datasets.history_vector import PAD_IDX  # noqa: E402
from src.interpretability.fidelity.perturbations import Perturbation  # noqa: E402
from src.interpretability.inputs import ExplainInput  # noqa: E402
from src.interpretability.registry import build_explainer  # noqa: E402
from src.trainers.dataloaders import build_dataloader  # noqa: E402

assert src.interpretability and src.recommenders

KE_GRID = [1, 2, 3, 4, 5]


def _load_ground_truth(path: Path, id_to_raw: list[int], offset: int) -> dict[int, dict[int, float]]:
    """Remap the raw-id-space planted weights into encoded-id space: W[v_enc] = {j_enc: w}."""
    payload = json.loads(path.read_text())
    raw_to_enc = {raw: i + offset for i, raw in enumerate(id_to_raw)}
    weights_raw = {int(k): v for k, v in payload["weights"].items()}

    weights_enc: dict[int, dict[int, float]] = {}
    for v_raw, drivers_raw in weights_raw.items():
        if v_raw not in raw_to_enc:
            continue
        v = raw_to_enc[v_raw]
        enc_drivers = {}
        for j_raw_str, w in drivers_raw.items():
            j_raw = int(j_raw_str)
            if j_raw not in raw_to_enc:
                continue
            enc_drivers[raw_to_enc[j_raw]] = w
        weights_enc[v] = enc_drivers
    return weights_enc


def _gt5(weights_for_v: dict[int, float], hist_items: set[int]) -> list[tuple[int, float]]:
    present = [(j, w) for j, w in weights_for_v.items() if j in hist_items]
    present.sort(key=lambda kv: -kv[1])
    return present[:5]


def _set_metrics(s_ke: set[int], gt_ke: set[int], ke: int) -> tuple[float, float, float, float]:
    if not gt_ke:
        return float("nan"), float("nan"), float("nan"), float("nan")
    inter = len(s_ke & gt_ke)
    precision = inter / ke if ke else float("nan")
    recall = inter / len(gt_ke)
    union = len(s_ke | gt_ke)
    jaccard = inter / union if union else float("nan")
    exact_hit = 1.0 if s_ke == gt_ke else 0.0
    return precision, recall, jaccard, exact_hit


def _bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    vals = np.array([v for v in values if not np.isnan(v)], dtype=np.float64)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means[i] = sample.mean()
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Manual Spearman rank correlation (avoids a hard scipy dependency)."""
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _mlp_effective_weight(model, v: int) -> np.ndarray:
    """Closed-form effective per-item weight of a trained `mlp`.

    The linear part of the map from `history` to `score[:, v]` is
    `history @ Uw.T @ (W[:, v] + b)`, with Uw = users_fc.weight,
    W = items_fc.weight, b = items_fc.bias (see MLP.forward).
    """
    with torch.no_grad():
        w_col = model.items_fc.weight[:, v] + model.items_fc.bias  # [hidden_dim]
        eff = model.users_fc.weight.T @ w_col  # [input_dim]
    return eff.cpu().numpy()


@hydra.main(version_base=None, config_path="../configs", config_name="explainer_eval")
def main(cfg: DictConfig) -> None:
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    container.pop("hydra", None)

    dataset_cfg, recommender_cfg, artifact_cfg, device, splits_dir, num_items, warm_items, model = (
        load_model_and_context(container)
    )

    # ── Ground truth ────────────────────────────────────────────────────────
    gt_path = Path(container.get("audit_ground_truth_path") or (_ROOT / "Data" / "Raw_Data" / "synth_ground_truth.json"))
    encoder_payload = json.loads((splits_dir / "item_encoder.json").read_text())
    offset = int(encoder_payload["offset"])
    id_to_raw = [int(x) for x in encoder_payload["id_to_raw"]]
    weights_enc = _load_ground_truth(gt_path, id_to_raw, offset)
    print(f"ground truth loaded: {len(weights_enc)} items with a planted driver set (post-encoding)")

    test_loader = build_dataloader(
        splits_dir / "test.csv", num_items=num_items, phase="test",
        batch_size=int(container["eval"]["batch_size"]), num_workers=int(container["eval"]["num_workers"]),
        warm_items=warm_items,
    )

    # ── Explainers ───────────────────────────────────────────────────────────
    all_expl = container["all_explainers"]
    requested = container.get("audit_explainers")
    explainer_names = [s.strip() for s in requested.split(",")] if requested else list(all_expl.keys())
    explainers = {}
    for name in explainer_names:
        if name not in all_expl:
            print(f"WARNING: explainer {name!r} not in the active budget catalog, skipping")
            continue
        params = dict(all_expl[name]["params"])
        explainers[name] = build_explainer(name, model, params=params, device=device)
    print(f"auditing explainers: {list(explainers)}")

    item_rank = int(container.get("audit_item_rank") or 0)
    max_users = container.get("audit_max_users")

    hydra_run_dir = Path(HydraConfig.get().run.dir)
    out_csv = hydra_run_dir / f"synth_gt_audit_{recommender_cfg.name}_{dataset_cfg.name}_q{container['budget_q']}.csv"
    fieldnames = [
        "user_id", "target_item", "hist_len", "g_size", "gt5_items", "gt5_weights",
        "explainer", "ke", "precision", "recall", "jaccard", "exact_hit",
        "precision_model", "pls_selection_objective",
    ]
    # Record which selection_objective `pls` ran with, so each CSV is self-describing.
    pls_selection_objective = (
        all_expl.get("pls", {}).get("params", {}).get("selection_objective", "") if "pls" in explainers else ""
    )

    # Per-(explainer, ke) accumulators for the headline (|G|>=5) subset.
    headline = defaultdict(lambda: defaultdict(list))  # explainer -> ke -> [precision,...]
    n_users_total = 0
    n_users_headline = 0
    spearman_vals = defaultdict(list)  # explainer -> [rho,...]

    import csv as csv_mod
    with out_csv.open("w", newline="") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for batch in test_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad():
                orig_scores, _ = model.full_sort_predict(batch)
            pad_masked = orig_scores.clone()
            pad_masked[:, PAD_IDX] = float("-inf")
            top = pad_masked.topk(item_rank + 1, dim=-1)
            items_to_explain = top.indices[:, item_rank]
            valid = top.values[:, item_rank] > float("-inf")

            inp = ExplainInput.from_batch(batch, family=Family.HISTORY_VECTOR, items_to_explain=items_to_explain)
            history = batch["history"]

            for b in range(inp.batch_size):
                if not bool(valid[b]):
                    continue
                n_users_total += 1
                target = int(items_to_explain[b].item())
                uid = int(batch["user_id"][b].item())
                hist_items = set(history[b].nonzero(as_tuple=False).squeeze(-1).tolist())

                weights_v = weights_enc.get(target, {})
                gt5 = _gt5(weights_v, hist_items)
                g_size = len([j for j in weights_v if j in hist_items])
                is_headline = g_size >= 5
                if is_headline:
                    n_users_headline += 1

                row_hist = history[b : b + 1]
                row_inp = ExplainInput(
                    family=Family.HISTORY_VECTOR, items_to_explain=items_to_explain[b : b + 1],
                    user_ids=inp.user_ids[b : b + 1], history=row_hist,
                )
                explainable = row_hist.bool()

                if "ig" in explainers:
                    attrs0 = explainers["ig"].explain(row_inp, ke=1)
                    imp = attrs0.importance[0]
                    present_j = [j for j in weights_v if j in hist_items]
                    if len(present_j) >= 2:
                        true_w = np.array([weights_v[j] for j in present_j])
                        attr_w = np.array([imp[j].item() for j in present_j])
                        spearman_vals["ig"].append(_spearman(true_w, attr_w))

                for ke in KE_GRID:
                    gt_ke = {j for j, _ in gt5[:ke]}

                    # All explainers' selections at this ke, computed first so an
                    # exact-enumeration `pls` in the same pass can serve as the
                    # GT_model reference for non-affine models.
                    selected: dict[str, set[int]] = {}
                    for expl_name, explainer in explainers.items():
                        attrs = explainer.explain(row_inp, ke=ke)
                        mask = Perturbation._select_positions(row_hist, attrs.importance, explainable, ke)
                        selected[expl_name] = set(mask[0].nonzero(as_tuple=False).squeeze(-1).tolist())

                    oracle_set = selected.get("pls")

                    for expl_name, s_ke in selected.items():
                        precision, recall, jaccard, exact_hit = _set_metrics(s_ke, gt_ke, ke)

                        precision_model = ""
                        if recommender_cfg.name == "mlp":
                            eff = _mlp_effective_weight(model, target)
                            present = [(j, eff[j]) for j in hist_items]
                            present.sort(key=lambda kv: -kv[1])
                            gt_model_ke = {j for j, _ in present[:ke]}
                            precision_model, *_ = _set_metrics(s_ke, gt_model_ke, ke)
                        elif oracle_set is not None:
                            precision_model, *_ = _set_metrics(s_ke, oracle_set, ke)

                        if is_headline:
                            headline[expl_name][ke].append(precision)

                        writer.writerow({
                            "user_id": uid, "target_item": target, "hist_len": len(hist_items),
                            "g_size": g_size, "gt5_items": "|".join(str(j) for j, _ in gt5),
                            "gt5_weights": "|".join(f"{w:.4f}" for _, w in gt5),
                            "explainer": expl_name, "ke": ke,
                            "precision": precision, "recall": recall, "jaccard": jaccard, "exact_hit": exact_hit,
                            "precision_model": precision_model,
                            "pls_selection_objective": pls_selection_objective,
                        })

            if max_users is not None and n_users_total >= int(max_users):
                break

    excl_rate = 1.0 - (n_users_headline / n_users_total if n_users_total else 0.0)
    print(f"\nWrote {out_csv}")
    print(f"n_users_total={n_users_total}  n_users_headline(|G|>=5)={n_users_headline}  exclusion_rate={excl_rate:.4f}")

    print("\n=== precision@ke vs GT_data, headline subset (|G|>=5), percentile-bootstrap 95% CI ===")
    for expl_name in explainers:
        for ke in KE_GRID:
            vals = headline[expl_name][ke]
            mean, lo, hi = _bootstrap_ci(vals)
            print(f"  {expl_name:16s} ke={ke}  precision={mean:.4f}  CI=[{lo:.4f},{hi:.4f}]  n={len(vals)}")

    if spearman_vals:
        print("\n=== attribution-vs-planted-weight Spearman rho (ig only) ===")
        for expl_name, vals in spearman_vals.items():
            mean, lo, hi = _bootstrap_ci(vals)
            print(f"  {expl_name:16s} rho={mean:.4f}  CI=[{lo:.4f},{hi:.4f}]  n={len(vals)}")

    if recommender_cfg.name not in ("mlp",) and "pls" not in explainers:
        print(f"\nNOTE: GT_model scoring only implemented for mlp's closed-form effective "
              f"weight; {recommender_cfg.name} needs an exact-oracle reference (include "
              f"`pls` configured for exact enumeration in ++audit_explainers) to populate "
              f"precision_model.")


if __name__ == "__main__":
    main()
