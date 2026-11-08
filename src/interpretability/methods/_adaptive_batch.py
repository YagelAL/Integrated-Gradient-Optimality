"""Adaptive batch-size chunk runner: shrink on CUDA OOM, grow back after a stable streak."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

_GROWTH_PATIENCE = 20  # consecutive successful chunks before doubling the batch size back up


def run_adaptive(
    r: int,
    get_batch_size: Callable[[], int],
    set_batch_size: Callable[[int], None],
    max_batch_size: int,
    run_chunk: Callable[[int, int], np.ndarray],
    get_streak: Callable[[], int] | None = None,
    set_streak: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Run ``run_chunk(start, stop) -> np.ndarray[stop-start]`` over ``[0, r)``.

    Halves the batch size (via ``get_batch_size``/``set_batch_size``) and retries
    on CUDA OOM; grows it back after a streak of successes, capped at
    ``max_batch_size``. Callers own the batch-size state so it can persist across
    calls (e.g. as an instance attribute) instead of resetting to the ceiling
    every time. ``get_streak``/``set_streak`` let the streak persist the same way.
    """
    out = np.empty(r, dtype=np.float64)
    start = 0
    streak = get_streak() if get_streak is not None else 0
    while start < r:
        bs = get_batch_size()
        stop = min(start + bs, r)
        try:
            out[start:stop] = run_chunk(start, stop)
        except torch.cuda.OutOfMemoryError:
            if bs <= 1:
                raise
            torch.cuda.empty_cache()
            set_batch_size(max(1, bs // 2))
            streak = 0
            continue
        start = stop
        streak += 1
        if streak >= _GROWTH_PATIENCE and bs < max_batch_size:
            set_batch_size(min(max_batch_size, bs * 2))
            streak = 0
    if set_streak is not None:
        set_streak(streak)
    return out


__all__ = ["run_adaptive"]
