"""x-window picking and ref/test point generation.

Distributions calibrated against the dev set
(0518_test_liubw/data/task/dev/samples.jsonl, 300 samples).
"""
from __future__ import annotations

import warnings

import numpy as np

WINDOWS: dict[str, dict] = {
    "easy":    {"ref_span": 7.5, "test_span": 2.5, "ref_n_range": (6, 9)},
    "medium":  {"ref_span": 8.4, "test_span": 2.8, "ref_n_range": (8, 11)},
    "hard":    {"ref_span": 8.7, "test_span": 2.9, "ref_n_range": (10, 14)},
    "expert":  {"ref_span": 8.2, "test_span": 2.7, "ref_n_range": (12, 16)},
    "extreme": {"ref_span": 9.5, "test_span": 3.1, "ref_n_range": (15, 20)},
}

GLOBAL_REF_X_RANGE = (-6.3, 8.0)
GLOBAL_TEST_X_RANGE = (-6.3, 2.7)

TEST_N = 50

EVAL_GLOBALS = {"np": np, "__builtins__": {}}


def pick_ref_window(diff: str, rng: np.random.Generator) -> tuple[float, float]:
    span = WINDOWS[diff]["ref_span"] * rng.uniform(0.85, 1.15)
    center = rng.uniform(0.0, 1.5)
    lo, hi = GLOBAL_REF_X_RANGE
    x_min = max(lo, center - span / 2)
    x_max = min(hi, center + span / 2)
    if x_max - x_min < 1.0:
        x_min, x_max = lo, lo + max(span, 4.0)
    return x_min, x_max


def pick_test_window(
    diff: str,
    ref_window: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[float, float]:
    span = WINDOWS[diff]["test_span"] * rng.uniform(0.9, 1.1)
    lo, hi = GLOBAL_TEST_X_RANGE
    rmin, rmax = ref_window
    cmin = max(lo, rmin - 0.5)
    cmax = min(hi, rmax)
    if cmax - cmin < span + 0.1:
        x_min = max(lo, cmin)
        return x_min, x_min + span
    x_min = float(rng.uniform(cmin, cmax - span))
    return x_min, x_min + span


def make_ref_xs(
    x_min: float, x_max: float, n: int, rng: np.random.Generator,
) -> list[float]:
    xs = sorted(rng.uniform(x_min, x_max, size=n).tolist())
    return [float(round(v, 6)) for v in xs]


def make_test_xs(x_min: float, x_max: float, n: int = TEST_N) -> list[float]:
    return [float(v) for v in np.linspace(x_min, x_max, n)]


def eval_expr(expr_numpy: str, xs: list[float]) -> np.ndarray:
    """Evaluate via the same sandbox that eval.py uses."""
    x = np.asarray(xs, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            y = eval(expr_numpy, {"x": x, **EVAL_GLOBALS})
    y = np.asarray(y, dtype=float)
    if y.shape == ():
        y = np.full_like(x, float(y))
    return y
