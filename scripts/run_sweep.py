"""Launch a W&B sweep across all visible GPUs and retrain the best run.

Workflow::

    1. Register a sweep from ``configs/sweep/<sweep>.yaml`` with W&B.
    2. Launch one ``wandb agent`` subprocess per visible GPU, each pinned
       to a single device via ``CUDA_VISIBLE_DEVICES``. ``--count`` is
       split across the agents (one experiment at a time per GPU).
    3. Wait for all agents to finish.
    4. Query the W&B API for the sweep's best run (by the sweep YAML's
       ``metric``).
    5. Launch a final ``scripts/train.py`` run with the best
       hyperparameters, but with ``final_run=true`` — train on
       train+val combined, evaluate on test, tag the W&B run as
       ``final_optimized``.

Usage::

    python scripts/run_sweep.py --sweep mlp_deep_synth --project mlp_deep_synth --dataset synth --count 1
    python scripts/run_sweep.py --sweep mlp    --project mlp_yahoo  --count 30 --gpus 4

If no CUDA GPUs are visible, falls back to a single CPU agent.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml  # noqa: E402

import wandb  # noqa: E402

FINAL_RUN_TAG = "final_optimized"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a multi-GPU W&B sweep and retrain the best config on the full data."
    )
    parser.add_argument(
        "--sweep",
        required=True,
        help="sweep name → configs/sweep/<name>.yaml",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="W&B project (defaults to the sweep name)",
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity (defaults to $WANDB_ENTITY)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="total number of sweep runs to execute across all GPUs",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="number of GPUs to use (defaults to all visible CUDA devices)",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="skip the post-sweep retrain-on-full-data + eval-on-test step",
    )
    parser.add_argument(
        "--final-tag",
        default=FINAL_RUN_TAG,
        help=f"W&B tag for the final optimized run (default: {FINAL_RUN_TAG!r})",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Hydra dataset name (e.g. ml1m, yahoo). When set, ``dataset=<name>`` "
            "is injected into the sweep YAML's command so every trial — and the "
            "final retrain — target that dataset. Without it, the Hydra default "
            "in configs/train.yaml is used."
        ),
    )
    parser.add_argument(
        "--min-final-epochs",
        type=int,
        default=None,
        help="floor for the final retrain's trainer.num_epochs: max(best_epoch + 1, this); unset = best_epoch",
    )
    args = parser.parse_args()

    sweep_yaml = _ROOT / "configs" / "sweep" / f"{args.sweep}.yaml"
    if not sweep_yaml.is_file():
        available = sorted(p.stem for p in (_ROOT / "configs" / "sweep").glob("*.yaml"))
        raise SystemExit(f"unknown sweep {args.sweep!r}; available: {available}")

    # Plain YAML load — do NOT use OmegaConf here. Sweep YAMLs contain W&B
    # macros like ``${env}`` / ``${interpreter}`` / ``${args_no_hyphens}``
    # which W&B's agent interprets at run time.
    with sweep_yaml.open() as f:
        sweep_cfg = yaml.safe_load(f)
    assert isinstance(sweep_cfg, dict)

    if args.dataset:
        # Inject ``dataset=<name>`` into the sweep YAML's command, immediately
        # before ``${args_no_hyphens}`` so per-trial overrides still take
        # precedence. This pushes the same dataset through every wandb agent
        # trial and (via _launch_final_run's fixed-arg extraction) through
        # the final retrain.
        cmd_list = list(sweep_cfg.get("command", []))
        dataset_arg = f"dataset={args.dataset}"
        if dataset_arg not in cmd_list:
            try:
                idx = cmd_list.index("${args_no_hyphens}")
            except ValueError:
                idx = len(cmd_list)
            cmd_list.insert(idx, dataset_arg)
        sweep_cfg["command"] = cmd_list

    project = args.project or args.sweep
    sweep_id = wandb.sweep(sweep_cfg, entity=args.entity, project=project)
    print(f"Sweep registered: id={sweep_id} project={project}")

    n_gpus = _resolve_gpu_count(args.gpus)
    per_agent_counts = _split_count(args.count, n_gpus)
    print(
        f"Launching {n_gpus} parallel agent(s) "
        f"(counts per agent: {per_agent_counts})"
    )
    _run_parallel_agents(
        sweep_id=sweep_id,
        project=project,
        entity=args.entity,
        per_agent_counts=per_agent_counts,
    )

    if args.skip_final:
        print("--skip-final set; not launching the final optimized retrain.")
        return

    best_run, best_overrides = _select_best_run(
        sweep_yaml_cfg=sweep_cfg,
        sweep_id=sweep_id,
        project=project,
        entity=args.entity,
    )
    print(f"Best sweep run: id={best_run.id} name={best_run.name!r}")

    # Pin the final retrain to the same epoch budget the best sweep trial
    # used (where its primary metric peaked). This avoids the
    # final-stage overtraining that would otherwise happen because the
    # final run has early stopping disabled.
    metric_cfg = sweep_cfg.get("metric") or {}
    metric_name = metric_cfg.get("name", "val/hr@10")
    metric_goal = metric_cfg.get("goal", "maximize")
    best_epoch = _best_epoch_from_run(
        best_run, metric_name=metric_name, metric_goal=metric_goal
    )
    if best_epoch is not None and best_epoch > 0:
        final_epochs = (
            best_epoch
            if args.min_final_epochs is None
            else max(best_epoch + 1, args.min_final_epochs)
        )
        print(
            f"Best epoch in sweep history: {best_epoch} (peak {metric_name}); "
            f"final run trainer.num_epochs = {final_epochs} "
            f"(min_final_epochs={args.min_final_epochs})."
        )
        best_overrides["trainer.num_epochs"] = final_epochs
    else:
        print(
            "Could not infer best epoch from run history; final run will "
            "use the configured trainer.num_epochs default."
        )

    print("Final-run overrides:")
    for k, v in best_overrides.items():
        print(f"  {k}={v}")

    _launch_final_run(
        sweep_yaml_cfg=sweep_cfg,
        overrides=best_overrides,
        project=project,
        entity=args.entity,
        final_tag=args.final_tag,
    )


# ---------------------------------------------------------------------------
# Parallel agent launching.
# ---------------------------------------------------------------------------


def _resolve_gpu_count(requested: int | None) -> int:
    """Determine how many agents to launch.

    Honors ``CUDA_VISIBLE_DEVICES`` if set; otherwise asks torch. Returns
    at least 1 (we still launch one CPU agent if no GPUs are available).
    """
    if requested is not None:
        return max(1, requested)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = [x for x in visible.split(",") if x.strip()]
        if ids:
            return len(ids)
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
    except ImportError:
        pass
    return 1


def _split_count(total: int, n_agents: int) -> list[int]:
    """Distribute ``total`` runs as evenly as possible across agents."""
    base = total // n_agents
    remainder = total % n_agents
    return [base + (1 if i < remainder else 0) for i in range(n_agents)]


def _run_parallel_agents(
    *,
    sweep_id: str,
    project: str,
    entity: str | None,
    per_agent_counts: list[int],
) -> None:
    """Spawn one ``wandb agent`` subprocess per GPU; wait for all to finish."""
    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids: list[str | None]
    if parent_visible is not None and parent_visible.strip():
        ids = [x.strip() for x in parent_visible.split(",") if x.strip()]
        # If user asked for fewer agents than ids, take the first N.
        gpu_ids = ids[: len(per_agent_counts)]
    else:
        gpu_ids = [str(i) for i in range(len(per_agent_counts))]

    if not gpu_ids:
        gpu_ids = [None]  # CPU fallback.

    sweep_path = f"{entity + '/' if entity else ''}{project}/{sweep_id}"
    processes: list[subprocess.Popen] = []
    for i, count in enumerate(per_agent_counts):
        if count <= 0:
            continue
        env = os.environ.copy()
        gpu = gpu_ids[i] if i < len(gpu_ids) else None
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            sys.executable,
            "-m",
            "wandb",
            "agent",
            "--count",
            str(count),
            sweep_path,
        ]
        print(
            f"[agent {i}] CUDA_VISIBLE_DEVICES={gpu!r} count={count} -> "
            f"{' '.join(cmd)}"
        )
        processes.append(subprocess.Popen(cmd, env=env, cwd=str(_ROOT)))

    failed = []
    for i, p in enumerate(processes):
        rc = p.wait()
        if rc != 0:
            failed.append((i, rc))
    if failed:
        details = ", ".join(f"agent {i} -> rc={rc}" for i, rc in failed)
        raise SystemExit(f"one or more sweep agents failed: {details}")
    print("All sweep agents finished.")


# ---------------------------------------------------------------------------
# Best-run selection + final retrain.
# ---------------------------------------------------------------------------


def _select_best_run(
    *,
    sweep_yaml_cfg: dict,
    sweep_id: str,
    project: str,
    entity: str | None,
) -> tuple[Any, dict[str, Any]]:
    """Query the sweep, pick the best run, and extract its sweep hyperparams.

    Returns (best_run, overrides) where overrides is a dict of
    ``hydra.dotted.key -> value`` restricted to the sweep YAML's
    parameter keys.
    """
    metric_cfg = sweep_yaml_cfg.get("metric") or {}
    metric_name = metric_cfg.get("name", "val/hr@10")
    metric_goal = metric_cfg.get("goal", "maximize")
    param_keys: list[str] = list((sweep_yaml_cfg.get("parameters") or {}).keys())

    api = wandb.Api()
    sweep_path = f"{entity + '/' if entity else ''}{project}/{sweep_id}"
    sweep = api.sweep(sweep_path)
    runs = [r for r in sweep.runs if r.state == "finished"]
    if not runs:
        raise SystemExit(f"no finished runs in sweep {sweep_path}")

    def score(run: Any) -> float:
        v = run.summary.get(metric_name)
        if v is None:
            return float("-inf") if metric_goal == "maximize" else float("inf")
        return float(v)

    reverse = metric_goal == "maximize"
    best = sorted(runs, key=score, reverse=reverse)[0]

    overrides: dict[str, Any] = {}
    for k in param_keys:
        if k in best.config:
            overrides[k] = best.config[k]
    return best, overrides


def _best_epoch_from_run(
    run: Any,
    *,
    metric_name: str,
    metric_goal: str,
) -> int | None:
    """Return the epoch at which ``metric_name`` was best in ``run``'s history.

    The Trainer logs ``{"<prefix>/<metric>": v, "epoch": e}`` once per
    eval-epoch (see ``src/trainers/base.py``); we scan those rows and
    pick the epoch with the best metric value under ``metric_goal``.

    Returns ``None`` if the metric or epoch keys are not present in the
    run history — in that case the caller should fall back to the
    configured ``trainer.num_epochs`` default.
    """
    best_epoch: int | None = None
    best_value: float | None = None
    for row in run.scan_history(keys=["epoch", metric_name]):
        v = row.get(metric_name)
        e = row.get("epoch")
        if v is None or e is None:
            continue
        try:
            v_f = float(v)
            e_i = int(e)
        except (TypeError, ValueError):
            continue
        if best_value is None or (
            (metric_goal == "maximize" and v_f > best_value)
            or (metric_goal == "minimize" and v_f < best_value)
        ):
            best_value, best_epoch = v_f, e_i
    return best_epoch


def _launch_final_run(
    *,
    sweep_yaml_cfg: dict,
    overrides: dict[str, Any],
    project: str,
    entity: str | None,
    final_tag: str,
) -> None:
    """Run scripts/train.py once with the best hyperparams + final_run=true."""
    # Discover any fixed Hydra args baked into the sweep ``command`` (e.g.
    # ``recommender=mlp_deep``) so the final run targets the same recommender.
    fixed_args: list[str] = []
    for token in sweep_yaml_cfg.get("command", []):
        if isinstance(token, str) and "=" in token and not token.startswith("${"):
            fixed_args.append(token)

    program = sweep_yaml_cfg.get("program", "scripts/train.py")

    cmd = [sys.executable, program, *fixed_args]
    for k, v in overrides.items():
        cmd.append(f"{k}={v}")
    cmd.append("final_run=true")
    cmd.append(f"final_run_tag={final_tag}")
    cmd.append(f"wandb.project={project}")
    if entity:
        cmd.append(f"wandb.entity={entity}")
    # train.py appends ``final_run_tag`` to wandb.tags itself when
    # ``final_run=true``; no need to override the list via CLI.

    print("Launching final optimized retrain:")
    print("  " + " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(_ROOT))
    if rc != 0:
        raise SystemExit(f"final retrain failed with rc={rc}")
    print(f"Final optimized run complete (tag: {final_tag!r}).")


if __name__ == "__main__":
    main()
