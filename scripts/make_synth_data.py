"""Generate the synthetic dataset with a known causal ground truth.

Plants a generative model ``p*(x | h) = sigmoid(sum_{j in h} w_{x,j} + b_x)``
over a genre-structured catalog, then samples user histories from that model,
so the observed co-occurrence structure reflects the planted weights.

Catalog: N_GENRES genres, each with a CORE_SIZE-item core + a PERIPHERY_SIZE-
item periphery. Per item x, weights are two-tiered and positive-only (the
optimal removal set is ranked by signed weight, so positive weights keep the
ground-truth ranking unambiguous):

  * 5 "primary" drivers per item, sampled from the item's own genre core,
    weight in PRIMARY_W_RANGE, distinct values (no ranking ties).
  * "secondary" drivers: the item's remaining core-mates plus a capped subset
    of its genre's periphery, weight in SECONDARY_W_RANGE (never overlapping
    the primary range, so top-5-by-weight stays unambiguous).
  * everything outside the item's own genre: weight exactly 0.

History length is hard-capped at N_MAX so the exact PLS enumeration over
comb(d, ke) subsets stays within budget for every user (d = eval-time history
length = generated length - 1, since eval holds out one item).

Usage::

    python scripts/make_synth_data.py --seed 0
    python scripts/make_synth_data.py --seed 0 --skip-preprocess
"""

from __future__ import annotations

import argparse
import os
import json
import math
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.schemas import (  # noqa: E402
    DatasetCfg,
    FilterCfg,
    PreprocessCfg,
    RandomFractionSplit,
)
from src.config.paths import SplitPaths  # noqa: E402
from src.preprocess.pipeline import run_preprocess_pipeline  # noqa: E402

N_GENRES = 20
CORE_SIZE = 8
PERIPHERY_SIZE = 20
N_USERS = 8000
P_CORE = 0.85
N_MAX = 24  # hard cap on generated history length
N_AUTO_MIN, N_AUTO_MAX = 10, 15  # target autoregressive-addition count (before cap)
N_FILLER_MIN, N_FILLER_MAX = 0, 4
MIN_HIST_LEN = 5  # floor, matching min_items_per_user (N-core filter)
PRIMARY_W_RANGE = (1.0, 3.0)
SECONDARY_W_RANGE = (0.1, 0.5)
N_SECONDARY_PERIPHERY = 7  # capped periphery items that also get a secondary weight, per item
BIAS = -3.0

# Recommended PLS exact-enumeration budget for the eval-time (post-holdout)
# history length N_MAX - 1, with margin.
_EVAL_D_MAX = N_MAX - 1
RECOMMENDED_EVAL_BUDGET = int(math.comb(_EVAL_D_MAX, 5) * 1.5)


def build_catalog() -> tuple[list[int], dict[int, int], dict[int, list[int]], dict[int, list[int]]]:
    items: list[int] = []
    genre_of: dict[int, int] = {}
    core_of_genre: dict[int, list[int]] = {g: [] for g in range(N_GENRES)}
    periphery_of_genre: dict[int, list[int]] = {g: [] for g in range(N_GENRES)}
    next_id = 1
    for g in range(N_GENRES):
        for _ in range(CORE_SIZE):
            items.append(next_id)
            genre_of[next_id] = g
            core_of_genre[g].append(next_id)
            next_id += 1
        for _ in range(PERIPHERY_SIZE):
            items.append(next_id)
            genre_of[next_id] = g
            periphery_of_genre[g].append(next_id)
            next_id += 1
    return items, genre_of, core_of_genre, periphery_of_genre


def plant_weights(
    rng: random.Random,
    items: list[int],
    genre_of: dict[int, int],
    core_of_genre: dict[int, list[int]],
    periphery_of_genre: dict[int, list[int]],
) -> tuple[dict[int, dict[int, float]], dict[int, str], dict[int, float]]:
    """Returns (W[x] = {driver: weight}, tier[(x,driver) encoded as f"{x}:{driver}"], bias[x])."""
    W: dict[int, dict[int, float]] = {}
    tier: dict[str, str] = {}
    bias: dict[int, float] = {}
    for x in items:
        g = genre_of[x]
        core_others = [j for j in core_of_genre[g] if j != x]
        primary = rng.sample(core_others, 5)
        # 101 evenly spaced candidate values in the primary range -> sampling
        # 5 distinct ones keeps weights unambiguous (never ties for "5th").
        candidate_ws = [
            round(PRIMARY_W_RANGE[0] + (PRIMARY_W_RANGE[1] - PRIMARY_W_RANGE[0]) * i / 100, 4)
            for i in range(101)
        ]
        primary_ws = rng.sample(candidate_ws, 5)
        w_x: dict[int, float] = {}
        for j, w in zip(primary, primary_ws):
            w_x[j] = w
            tier[f"{x}:{j}"] = "primary"

        remaining_core = [j for j in core_others if j not in primary]
        periphery_pool = periphery_of_genre[g]
        n_periph = min(N_SECONDARY_PERIPHERY, len(periphery_pool))
        secondary_periph = rng.sample(periphery_pool, n_periph)
        secondary_items = remaining_core + secondary_periph
        # Distinct secondary weights too, so GT_5 has no ties at the rank-5
        # boundary when a user lacks a primary driver.
        secondary_candidate_ws = [
            round(SECONDARY_W_RANGE[0] + (SECONDARY_W_RANGE[1] - SECONDARY_W_RANGE[0]) * i / 100, 4)
            for i in range(101)
        ]
        secondary_ws = rng.sample(secondary_candidate_ws, len(secondary_items))
        for j, w in zip(secondary_items, secondary_ws):
            w_x[j] = w
            tier[f"{x}:{j}"] = "secondary"

        W[x] = w_x
        bias[x] = BIAS
    return W, tier, bias


def _weighted_choice(rng: random.Random, pool: list[int], weights: list[float]) -> int:
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(pool) - 1


def generate_user_history(
    rng: random.Random,
    g: int,
    W: dict[int, dict[int, float]],
    bias: dict[int, float],
    core_of_genre: dict[int, list[int]],
    periphery_of_genre: dict[int, list[int]],
    genre_of: dict[int, int],
    all_items: list[int],
) -> set[int]:
    hist: set[int] = set()

    # 1. Seed each core item independently at P_CORE (< 1, so core items are
    #    not perfectly collinear in the observed data).
    for c in core_of_genre[g]:
        if rng.random() < P_CORE:
            hist.add(c)

    # 2. Autoregressive additions, sampled proportional to p*(x|h_so_far)
    #    from the user's own genre pool, budget-capped so N_MAX holds exactly.
    filler_budget = rng.randint(N_FILLER_MIN, N_FILLER_MAX)
    genre_pool = [it for it in core_of_genre[g] + periphery_of_genre[g] if it not in hist]
    n_auto_target = rng.randint(N_AUTO_MIN, N_AUTO_MAX)
    max_from_auto = max(0, N_MAX - len(hist) - filler_budget)
    n_auto = min(n_auto_target, max_from_auto, len(genre_pool))
    for _ in range(n_auto):
        if not genre_pool:
            break
        scores = []
        for x in genre_pool:
            s = sum(W[x].get(j, 0.0) for j in hist)
            p = 1.0 / (1.0 + math.exp(-(s + bias[x])))
            scores.append(max(p, 1e-6))
        idx = _weighted_choice(rng, genre_pool, scores)
        hist.add(genre_pool.pop(idx))

    # 3. Filler items from other genres (distractor noise).
    other_items = [it for it in all_items if genre_of[it] != g and it not in hist]
    rng.shuffle(other_items)
    remaining_budget = N_MAX - len(hist)
    n_filler = min(filler_budget, remaining_budget, len(other_items))
    hist.update(other_items[:n_filler])

    # 4. Top up to the N-core minimum so preprocessing does not drop the user.
    if len(hist) < MIN_HIST_LEN:
        topup_pool = [it for it in all_items if it not in hist]
        rng.shuffle(topup_pool)
        need = min(MIN_HIST_LEN - len(hist), N_MAX - len(hist), len(topup_pool))
        hist.update(topup_pool[:need])

    assert len(hist) <= N_MAX, f"history length {len(hist)} exceeds hard cap {N_MAX}"
    return hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-users", type=int, default=N_USERS)
    ap.add_argument("--output-dir", type=Path, default=os.environ.get("RAW_DATA_DIR", _ROOT / "Data" / "Raw_Data"))
    ap.add_argument("--preprocess-output-root", type=Path, default=os.environ.get("DATA_OUTPUT_ROOT", _ROOT / "processed_data"))
    ap.add_argument("--skip-preprocess", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    items, genre_of, core_of_genre, periphery_of_genre = build_catalog()
    W, tier, bias = plant_weights(rng, items, genre_of, core_of_genre, periphery_of_genre)

    print(f"catalog: {N_GENRES} genres x ({CORE_SIZE} core + {PERIPHERY_SIZE} periphery) = {len(items)} items")
    print(f"N_MAX={N_MAX} (eval-time d<=<{_EVAL_D_MAX}>, comb({_EVAL_D_MAX},5)={math.comb(_EVAL_D_MAX, 5)}, "
          f"recommended eval_budget={RECOMMENDED_EVAL_BUDGET})")

    rows: list[tuple[int, int]] = []
    hist_lens: list[int] = []
    for uid in range(1, args.n_users + 1):
        g = uid % N_GENRES
        hist = generate_user_history(rng, g, W, bias, core_of_genre, periphery_of_genre, genre_of, items)
        hist_lens.append(len(hist))
        for it in hist:
            rows.append((uid, it))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.output_dir / "synth.csv"
    with raw_csv.open("w") as fh:
        fh.write("user_id,item_id\n")
        for uid, it in rows:
            fh.write(f"{uid},{it}\n")
    print(f"wrote {len(rows)} interactions, {args.n_users} users -> {raw_csv}")
    print(f"history length: min={min(hist_lens)} max={max(hist_lens)} "
          f"mean={sum(hist_lens) / len(hist_lens):.2f}")

    gt_path = args.output_dir / "synth_ground_truth.json"
    gt_payload = {
        "generator_knobs": {
            "n_genres": N_GENRES, "core_size": CORE_SIZE, "periphery_size": PERIPHERY_SIZE,
            "n_users": args.n_users, "p_core": P_CORE, "n_max": N_MAX,
            "n_auto_range": [N_AUTO_MIN, N_AUTO_MAX], "n_filler_range": [N_FILLER_MIN, N_FILLER_MAX],
            "min_hist_len": MIN_HIST_LEN, "primary_w_range": list(PRIMARY_W_RANGE),
            "secondary_w_range": list(SECONDARY_W_RANGE), "n_secondary_periphery": N_SECONDARY_PERIPHERY,
            "bias": BIAS, "seed": args.seed,
        },
        "recommended_eval_budget": RECOMMENDED_EVAL_BUDGET,
        "eval_d_max": _EVAL_D_MAX,
        "genre_of": {str(k): v for k, v in genre_of.items()},
        "core_of_genre": {str(k): v for k, v in core_of_genre.items()},
        "periphery_of_genre": {str(k): v for k, v in periphery_of_genre.items()},
        "weights": {str(x): w for x, w in W.items()},  # W[x] = {driver_item: weight}
        "tier": tier,  # "{x}:{driver}" -> "primary"|"secondary"
        "bias": {str(x): b for x, b in bias.items()},
    }
    gt_path.write_text(json.dumps(gt_payload, indent=2))
    print(f"wrote ground truth -> {gt_path}")

    if args.skip_preprocess:
        return

    # Preprocessing + N-core survival check.
    dataset_cfg = DatasetCfg(
        name="synth", raw_path=raw_csv, user_col="user_id", item_col="item_id",
        timestamp_col=None, rating_col=None, rating_threshold=None, format="csv", csv_sep=",",
    )
    filters_cfg = FilterCfg(min_items_per_user=5, min_users_per_item=5, drop_consecutive_repeats=False)
    split_cfg = RandomFractionSplit(test_fraction=0.2, val_fraction=0.1, seed=42)
    pre_cfg = PreprocessCfg(
        dataset=dataset_cfg, filters=filters_cfg, split=split_cfg, output_root=args.preprocess_output_root,
    )
    result = run_preprocess_pipeline(pre_cfg)
    print(f"\npreprocess: num_users={result.num_users} num_items={result.num_items} filter={result.filter_report}")

    paths = SplitPaths(output_root=args.preprocess_output_root, dataset_name="synth", split=split_cfg)
    encoder_payload = json.loads((paths.split_dir / "item_encoder.json").read_text())
    id_to_raw = {int(x) for x in encoder_payload["id_to_raw"]}
    n_dropped = len(items) - len(id_to_raw)
    print(f"catalog survival: {len(id_to_raw)}/{len(items)} planted items survived N-core filtering")
    if n_dropped:
        dropped = sorted(set(items) - id_to_raw)
        print(f"WARNING: {n_dropped} items dropped: {dropped}")
        # Surviving items whose planted driver set references a dropped item.
        affected = []
        for x in items:
            if x not in id_to_raw:
                continue
            bad_drivers = [j for j in W[x] if j not in id_to_raw]
            if bad_drivers:
                affected.append((x, bad_drivers))
        if affected:
            print(f"WARNING: {len(affected)} surviving items reference dropped drivers "
                  f"(fewer available drivers in their GT rows): e.g. {affected[:5]}")


if __name__ == "__main__":
    main()
