"""IND v4 generator — single-file, round-number distributions.

Design goals (from user):
  1. Coordinates in `data_points_text` and `test_points` are stored as
     4-decimal floats, matching eval.py's `f"({x:.4f}, {y:.4f})"` prompt
     format. Generator output is byte-equivalent to what eval.py prints.
  2. Distribution constants are simple, round numbers (5%/10%/15%/20% ...).
     No Laplace smoothing, no dev metadata reverse-fitting. Match dev to
     within a few percentage points and stop there.

Schema matches dev (`0518_test_liubw/data/task/dev/samples.jsonl`) so the
output drops into eval.py / build_symreg_data.py without changes.

Usage:
    python in_domain/v4/generate_v4.py --n 2000 --out in_domain/output/ind_v4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ind-v4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIFFICULTIES = ("easy", "medium", "hard", "expert", "extreme")

# All 13 functions in the candidate pool (matches eval.py / dev).
ALL_FUNCS = (
    "np.sin", "np.cos", "np.exp", "np.sqrt", "np.log", "np.tanh",
    "np.arctan", "np.arcsin", "np.arccos", "np.sinh", "np.cosh", "np.tan",
    "np.abs",
)

# Functions that can be called on raw `b * x` without domain issues.
SAFE_FUNCS = ("np.sin", "np.cos", "np.tanh", "np.arctan", "np.sinh", "np.cosh")

# --- per-difficulty distributions (round numbers) --------------------------
#
# Reference (dev empirical, for sanity):
#   pow_pct        easy 7%   med 17%  hard 40%  expert 57%  extreme 58%
#   has_abs_pct    easy 0%   med 13%  hard 20%  expert 45%  extreme 38%
#   has_log_pct    easy 0%   med 13%  hard  0%  expert 28%  extreme 13%
#   has_sqrt_pct   easy 8%   med  0%  hard 20%  expert 17%  extreme 25%
#   ref_span_mean  easy 7.5  med 8.4  hard 8.7  expert 8.2  extreme 9.5
#   test_span_mean ≈2.5     ≈2.8     ≈2.9     ≈2.7        ≈3.1
#   ref_n_mean     easy 7    med 9    hard 12   expert 14   extreme 17
#   hints_n_mean   easy 0.9  med 2.1  hard 3.1  expert 4.0  extreme 4.9
#   avg_terms      easy 1.0  med 2.15 hard 1.0  expert 1.43 extreme 1.80
#
# v4 round-numbered targets. Each row sums to 1.0.
# Slot meanings:
#   plain      — 1 SAFE_FUNCS atom  (a*f(b*x) or +c phase)
#   poly       — atom with x ** 2 / x ** 3 inside
#   sqrt       — sqrt(abs(atom)+1)
#   log        — log(abs(atom)+1)
#   abs        — abs(safe_atom)
#   exp_pos    — a * np.exp(b*x)
#   exp_neg_x2 — a * np.exp(-c*x**2)
#   exp_atom   — a * np.exp(-c*x**2) * f(b*x)   (for hard/extreme)
#   nest_phase — a * f(b*x + d*g(c*x))          (extreme only)
STRUCT_WEIGHTS: dict[str, dict[str, float]] = {
    # easy: single atom, sometimes exp; almost no abs/log/sqrt.
    "easy":    {"plain": 0.70, "exp_pos": 0.20, "exp_neg_x2": 0.10},
    # medium: 2-term, head atom + linear/constant tail.
    "medium":  {"plain": 0.55, "abs": 0.15, "log": 0.15, "exp_neg_x2": 0.15},
    # hard: 1 term but richer atom — pow / sqrt / exp*sin product.
    "hard":    {"plain": 0.30, "poly": 0.25, "sqrt": 0.20, "abs": 0.10, "exp_atom": 0.15},
    # expert: 1-2 terms, more diverse heads.
    "expert":  {"plain": 0.20, "poly": 0.20, "sqrt": 0.20, "log": 0.20, "abs": 0.10, "exp_neg_x2": 0.10},
    # extreme: deepest nesting + product structure.
    "extreme": {"plain": 0.15, "poly": 0.20, "sqrt": 0.20, "exp_atom": 0.20, "log": 0.10, "nest_phase": 0.15},
}

# Additive tail (linear / constant) probability for medium / expert / extreme.
TAIL_PROB = {"easy": 0.0, "medium": 0.95, "hard": 0.0, "expert": 0.30, "extreme": 0.40}

# top-level term count (most rows are 1; medium is 2).
TERM_PMF = {
    "easy":    {1: 1.00},
    "medium":  {2: 0.85, 3: 0.15},
    "hard":    {1: 1.00},
    "expert":  {1: 0.60, 2: 0.40},
    "extreme": {1: 0.30, 2: 0.50, 3: 0.20},
}

# ref_n: how many points in data_points_text.
REF_N_RANGE = {
    "easy":    (6, 8),
    "medium":  (8, 11),
    "hard":    (10, 14),
    "expert":  (12, 16),
    "extreme": (15, 20),
}

# x-window spans (rounded numbers).
REF_SPAN = {"easy": 7.5, "medium": 8.5, "hard": 8.5, "expert": 8.5, "extreme": 9.5}
TEST_SPAN = {"easy": 2.5, "medium": 2.8, "hard": 2.9, "expert": 2.7, "extreme": 3.1}

# function_hints length (target). Round numbers per difficulty.
HINT_LEN = {"easy": (0, 2), "medium": (1, 3), "hard": (2, 4), "expert": (3, 5), "extreme": (4, 6)}

# Per-difficulty weights for the SAFE_FUNCS slot inside a `plain` atom.
# Tracks dev's expr_freq qualitatively (sin dominates, cos second).
SAFE_FUNC_WEIGHTS = {
    "easy":    {"np.sin": 0.40, "np.cos": 0.30, "np.tanh": 0.10, "np.arctan": 0.10, "np.sinh": 0.05, "np.cosh": 0.05},
    "medium":  {"np.sin": 0.45, "np.cos": 0.30, "np.tanh": 0.10, "np.arctan": 0.05, "np.sinh": 0.05, "np.cosh": 0.05},
    "hard":    {"np.sin": 0.55, "np.cos": 0.25, "np.tanh": 0.10, "np.arctan": 0.05, "np.sinh": 0.03, "np.cosh": 0.02},
    "expert":  {"np.sin": 0.55, "np.cos": 0.25, "np.tanh": 0.10, "np.arctan": 0.05, "np.sinh": 0.03, "np.cosh": 0.02},
    "extreme": {"np.sin": 0.65, "np.cos": 0.20, "np.tanh": 0.10, "np.arctan": 0.03, "np.sinh": 0.01, "np.cosh": 0.01},
}

# Distractor weights for function_hints (round). Same across difficulties to
# keep things simple — every non-true function has equal chance of being a
# distractor, modulo a small popularity bias for sin/cos/exp/log. np.abs is
# included so the prompt pool matches dev (where abs appears in ~14% of
# function_hints even though it's a structural wrapper, not a base function).
DISTRACTOR_WEIGHTS = {
    "np.sin": 0.18, "np.cos": 0.16, "np.exp": 0.16, "np.log": 0.10,
    "np.abs": 0.10, "np.sqrt": 0.08, "np.tanh": 0.06, "np.arctan": 0.05,
    "np.arcsin": 0.04, "np.arccos": 0.03, "np.sinh": 0.02, "np.cosh": 0.01,
    "np.tan": 0.01,
}

CONSTS_SMALL = (1, 2, 3)
CONSTS_FRAC = (0.3, 0.5, 0.8, 1.5)
CONSTS_BIG = (4, 5)
CONST_GROUP_W = (0.70, 0.20, 0.10)

Y_ABS_MAX = 1e6
Y_VAR_MIN = 1e-6
TEST_N = 50
DEFAULT_RETRIES = 50
DEFAULT_SEED = 20260520

EVAL_GLOBALS = {"np": np, "__builtins__": {}}

DEFAULT_PROMPT = (
    "You are a symbolic regression expert. Given a function curve plot and "
    "reference information, call submit_expression with the inferred numpy "
    "expression.\n\n{function_hints}{data_points}{axis_note}\n"
    "Use only numpy (np.) functions; the variable must be named x."
)


# --- low-level helpers -----------------------------------------------------

def _fmt_const(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return f"{v:g}"


def _signed(rng: np.random.Generator, val: float) -> float:
    return -val if rng.random() < 0.5 else val


def _const(rng: np.random.Generator) -> float:
    g = rng.choice(3, p=CONST_GROUP_W)
    pool = (CONSTS_SMALL, CONSTS_FRAC, CONSTS_BIG)[g]
    return float(pool[rng.integers(len(pool))])


def _weighted_choice(weights: dict[str, float], rng: np.random.Generator) -> str:
    keys = list(weights)
    probs = np.array([weights[k] for k in keys], dtype=float)
    probs /= probs.sum()
    return str(rng.choice(keys, p=probs))


def _pick_safe(diff: str, rng: np.random.Generator) -> str:
    return _weighted_choice(SAFE_FUNC_WEIGHTS[diff], rng)


# --- atomic builders -------------------------------------------------------

def _term(coef: float, body: str) -> str:
    """`coef * body` with sign baked into the leading operator (no `+ -`)."""
    if coef >= 0:
        return f"+ {_fmt_const(coef)} * {body}" if body else f"+ {_fmt_const(coef)}"
    return f"- {_fmt_const(-coef)} * {body}" if body else f"- {_fmt_const(-coef)}"


def _phase_arg(b: float, c: float | None) -> str:
    """`b * x` with optional `+ c` / `- c` phase, sign-aware."""
    if c is None:
        return f"{_fmt_const(b)} * x"
    if c >= 0:
        return f"{_fmt_const(b)} * x + {_fmt_const(c)}"
    return f"{_fmt_const(b)} * x - {_fmt_const(-c)}"


def _atom_plain(diff: str, rng: np.random.Generator) -> str:
    """a * f(b*x [+ c])"""
    a = _const(rng); b = _const(rng); f = _pick_safe(diff, rng)
    c: float | None = _signed(rng, _const(rng)) if rng.random() < 0.30 else None
    return f"{_fmt_const(a)} * {f}({_phase_arg(b, c)})"


def _atom_poly(diff: str, rng: np.random.Generator) -> str:
    """a * f(b*x ** p) — pow inside a periodic call."""
    a = _const(rng); b = _const(rng); f = _pick_safe(diff, rng)
    p = int(rng.choice([2, 3]))
    return f"{_fmt_const(a)} * {f}({_fmt_const(b)} * x ** {p})"


def _atom_sqrt(diff: str, rng: np.random.Generator) -> str:
    """a * sqrt(abs(atom) + 1)"""
    a = _const(rng)
    inner = _atom_plain(diff, rng)
    return f"{_fmt_const(a)} * np.sqrt(np.abs({inner}) + 1)"


def _atom_log(diff: str, rng: np.random.Generator) -> str:
    """a * log(abs(atom) + 1)"""
    a = _const(rng)
    inner = _atom_plain(diff, rng)
    return f"{_fmt_const(a)} * np.log(np.abs({inner}) + 1)"


def _atom_abs(diff: str, rng: np.random.Generator) -> str:
    """a * abs(f(b*x))"""
    a = _const(rng); b = _const(rng); f = _pick_safe(diff, rng)
    return f"{_fmt_const(a)} * np.abs({f}({_fmt_const(b)} * x))"


def _atom_exp_pos(diff: str, rng: np.random.Generator) -> str:
    """a * exp(b*x)  — small b to stay finite."""
    a = _const(rng)
    b = _signed(rng, float(rng.choice([0.3, 0.5, 0.8, 1.0])))
    return f"{_fmt_const(a)} * np.exp({_fmt_const(b)} * x)"


def _atom_exp_neg_x2(diff: str, rng: np.random.Generator) -> str:
    """a * exp(-c*x**2)  — Gaussian bump."""
    a = _const(rng)
    c = float(rng.choice([0.2, 0.3, 0.5, 0.8]))
    return f"{_fmt_const(a)} * np.exp(-{_fmt_const(c)} * x ** 2)"


def _atom_exp_atom(diff: str, rng: np.random.Generator) -> str:
    """a * exp(-c*x**2) * f(b*x)  — localized oscillation."""
    a = _const(rng)
    c = float(rng.choice([0.2, 0.3, 0.5, 0.8]))
    b = _const(rng)
    f = _pick_safe(diff, rng)
    return f"{_fmt_const(a)} * np.exp(-{_fmt_const(c)} * x ** 2) * {f}({_fmt_const(b)} * x)"


def _atom_nest_phase(diff: str, rng: np.random.Generator) -> str:
    """a * f(b*x + d*g(c*x))  — phase-modulated oscillator."""
    a = _const(rng); b = _const(rng); c = _const(rng); d = _signed(rng, _const(rng))
    f = _pick_safe(diff, rng); g = _pick_safe(diff, rng)
    sign = "+" if d >= 0 else "-"
    return (f"{_fmt_const(a)} * {f}({_fmt_const(b)} * x {sign} "
            f"{_fmt_const(abs(d))} * {g}({_fmt_const(c)} * x))")


_ATOM_BUILDERS: dict[str, Callable[[str, np.random.Generator], str]] = {
    "plain":      _atom_plain,
    "poly":       _atom_poly,
    "sqrt":       _atom_sqrt,
    "log":        _atom_log,
    "abs":        _atom_abs,
    "exp_pos":    _atom_exp_pos,
    "exp_neg_x2": _atom_exp_neg_x2,
    "exp_atom":   _atom_exp_atom,
    "nest_phase": _atom_nest_phase,
}


def _sample_atom(diff: str, rng: np.random.Generator) -> str:
    kind = _weighted_choice(STRUCT_WEIGHTS[diff], rng)
    return _ATOM_BUILDERS[kind](diff, rng)


def _sample_n_terms(diff: str, rng: np.random.Generator) -> int:
    pmf = TERM_PMF[diff]
    keys = list(pmf); probs = np.array([pmf[k] for k in keys], dtype=float)
    probs /= probs.sum()
    return int(rng.choice(keys, p=probs))


def _sample_tail(rng: np.random.Generator) -> str:
    """Linear or constant tail; sign embedded in leading `+`/`-`."""
    if rng.random() < 0.5:
        c = _signed(rng, _const(rng))
        sign = "+" if c >= 0 else "-"
        return f"{sign} {_fmt_const(abs(c))} * x"
    c = _signed(rng, _const(rng))
    sign = "+" if c >= 0 else "-"
    return f"{sign} {_fmt_const(abs(c))}"


def sample_expression(diff: str, rng: np.random.Generator) -> str:
    n_terms = _sample_n_terms(diff, rng)
    parts = [_sample_atom(diff, rng) for _ in range(n_terms)]
    head = parts[0]
    body = head
    for atom in parts[1:]:
        body += f" + {atom}"
    if n_terms == 1 and rng.random() < TAIL_PROB[diff]:
        body += " " + _sample_tail(rng)
    if n_terms >= 2:
        # extra tail allowed too
        if rng.random() < TAIL_PROB[diff]:
            body += " " + _sample_tail(rng)
    return body


# --- ref/test windows + points --------------------------------------------

def pick_ref_window(diff: str, rng: np.random.Generator) -> tuple[float, float]:
    span = REF_SPAN[diff] * float(rng.uniform(0.95, 1.10))
    center = float(rng.uniform(-1.5, 1.5))
    lo = center - span / 2
    hi = center + span / 2
    return float(lo), float(hi)


def pick_test_window(
    diff: str, ref_window: tuple[float, float], rng: np.random.Generator,
) -> tuple[float, float]:
    rmin, rmax = ref_window
    span = TEST_SPAN[diff] * float(rng.uniform(0.95, 1.10))
    if rmax - rmin <= span + 0.1:
        return rmin, rmin + span
    x_min = float(rng.uniform(rmin, rmax - span))
    return x_min, x_min + span


def make_ref_xs(
    x_min: float, x_max: float, n: int, rng: np.random.Generator,
) -> list[float]:
    """Pin the two endpoints so the achieved span equals the configured one,
    then draw n-2 interior points uniformly. Without pinning, the expected
    achieved span of n uniform draws is W*(n-1)/(n+1) — systematically short.
    """
    if n <= 2:
        return [round(float(x_min), 4), round(float(x_max), 4)][:n]
    interior = rng.uniform(x_min, x_max, size=n - 2).tolist()
    xs = sorted([x_min, x_max] + interior)
    return [round(float(v), 4) for v in xs]


def make_test_xs(x_min: float, x_max: float, n: int = TEST_N) -> list[float]:
    return [round(float(v), 4) for v in np.linspace(x_min, x_max, n)]


def eval_expr(expr_numpy: str, xs: list[float]) -> np.ndarray:
    x = np.asarray(xs, dtype=float)
    with np.errstate(all="ignore"):
        y = eval(expr_numpy, {"x": x, **EVAL_GLOBALS})
    y = np.asarray(y, dtype=float)
    if y.shape == ():
        y = np.full_like(x, float(y))
    return y


def validate(y_ref: np.ndarray, y_test: np.ndarray) -> bool:
    if not (np.all(np.isfinite(y_ref)) and np.all(np.isfinite(y_test))):
        return False
    if float(np.var(y_test)) <= Y_VAR_MIN:
        return False
    if float(np.max(np.abs(y_ref))) > Y_ABS_MAX:
        return False
    if float(np.max(np.abs(y_test))) > Y_ABS_MAX:
        return False
    return True


# --- function hints --------------------------------------------------------

_NP_FN_RE = re.compile(
    r"np\.(sin|cos|exp|sqrt|log|abs|tanh|arctan|arcsin|arccos|sinh|tan|cosh)\b"
)
_TRUE_HINT_EXCLUDE = {"np.abs"}  # dev convention


def extract_true_hints(expr_numpy: str) -> list[str]:
    seen: list[str] = []
    for m in _NP_FN_RE.finditer(expr_numpy):
        name = "np." + m.group(1)
        if name in _TRUE_HINT_EXCLUDE:
            continue
        if name not in seen:
            seen.append(name)
    return seen


def make_function_hints(
    diff: str, true_hints: list[str], rng: np.random.Generator,
) -> list[str]:
    lo, hi = HINT_LEN[diff]
    target = int(rng.integers(lo, hi + 1))
    selected = list(true_hints)[:target]
    pool = [f for f in ALL_FUNCS if f not in selected]
    while len(selected) < target and pool:
        probs = np.array([DISTRACTOR_WEIGHTS.get(f, 0.05) for f in pool], dtype=float)
        probs /= probs.sum()
        idx = int(rng.choice(len(pool), p=probs))
        selected.append(pool.pop(idx))
    return selected


# --- rendering -------------------------------------------------------------

FIGSIZE = (5.9, 3.9)
DPI = 100
LINE_COLOR = "steelblue"
GRID_COLOR = "#ededed"


def render_png(expr_numpy: str, x_min: float, x_max: float, out_path: Path) -> None:
    x = np.linspace(x_min, x_max, 400)
    with np.errstate(all="ignore"):
        y = eval(expr_numpy, {"x": x, **EVAL_GLOBALS})
    y = np.asarray(y, dtype=float)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.grid(True, color=GRID_COLOR, linewidth=1.0, zorder=0)
    ax.plot(x, y, linewidth=2, color=LINE_COLOR, zorder=2)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.set_size_inches(*FIGSIZE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=DPI)
    plt.close(fig)


# --- prompt rendering (mirrors eval.py:build_message) ----------------------

def _format_points(pts: list[list[float]]) -> str:
    return "  ".join(f"({x:.4f}, {y:.4f})" for x, y in pts)


def render_prompt(record: dict, template: str) -> str:
    hints = record.get("function_hints", [])
    function_hints = (
        "Available functions: " + ", ".join(hints) + "\n" if hints else ""
    )
    pts = record.get("data_points_text", [])
    data_points = (
        "Reference points: " + _format_points(pts) + "\n" if pts else ""
    )
    return template.format(
        function_hints=function_hints, data_points=data_points, axis_note="",
    )


# --- pipeline --------------------------------------------------------------

def build_one(
    diff: str, sid: int, rng: np.random.Generator,
    out_dir: Path, max_retries: int, template: str,
) -> dict:
    n_lo, n_hi = REF_N_RANGE[diff]
    last_err: str | None = None
    for _ in range(max_retries):
        expr_np = sample_expression(diff, rng)
        ref_win = pick_ref_window(diff, rng)
        test_win = pick_test_window(diff, ref_win, rng)
        n_ref = int(rng.integers(n_lo, n_hi + 1))
        ref_xs = make_ref_xs(*ref_win, n=n_ref, rng=rng)
        test_xs = make_test_xs(*test_win, n=TEST_N)
        try:
            y_ref = eval_expr(expr_np, ref_xs)
            y_test = eval_expr(expr_np, test_xs)
        except Exception as exc:
            last_err = str(exc); continue
        if not validate(y_ref, y_test):
            last_err = "validation failed"; continue
        # Round y values to 4 decimals to match prompt format exactly.
        ref_pts = [[xv, round(float(yv), 4)] for xv, yv in zip(ref_xs, y_ref.tolist())]
        test_pts = [[xv, round(float(yv), 4)] for xv, yv in zip(test_xs, y_test.tolist())]
        true_hints = extract_true_hints(expr_np)
        function_hints = make_function_hints(diff, true_hints, rng)
        img_rel = f"images/ind_{diff}_{sid:04d}.png"
        render_png(expr_np, *ref_win, out_dir / img_rel)
        record = {
            "id": sid,
            "split": "ind",
            "expression_str": re.sub(r"\bnp\.", "", expr_np),
            "expression_numpy": expr_np,
            "true_function_hints": true_hints,
            "function_hints": function_hints,
            "data_points_text": ref_pts,
            "image_path": img_rel,
            "test_points": test_pts,
        }
        record["prompt"] = render_prompt(record, template)
        return record
    raise RuntimeError(
        f"failed to build {diff} sample {sid} after {max_retries} retries "
        f"(last error: {last_err})"
    )


def parse_counts(spec: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise argparse.ArgumentTypeError(
                f"--counts expects key=val pairs, got {chunk!r}"
            )
        k, v = chunk.split("=", 1)
        k = k.strip()
        if k not in DIFFICULTIES:
            raise argparse.ArgumentTypeError(f"unknown difficulty {k!r}")
        out[k] = int(v)
    return out


def resolve_counts(args: argparse.Namespace) -> dict[str, int]:
    if args.counts:
        c = parse_counts(args.counts)
        for d in DIFFICULTIES:
            c.setdefault(d, 0)
        return {d: c[d] for d in DIFFICULTIES}
    n = args.n
    per = n // len(DIFFICULTIES)
    rem = n - per * len(DIFFICULTIES)
    counts = {d: per for d in DIFFICULTIES}
    for d in list(DIFFICULTIES)[:rem]:
        counts[d] += 1
    return counts


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--counts", type=str, default=None,
                   help="per-difficulty counts, e.g. easy=400,medium=400,...")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-retries", type=int, default=DEFAULT_RETRIES)
    p.add_argument("--prompt-file", type=Path, default=None,
                   help="prompt template; defaults to in_domain/prompt.txt")
    args = p.parse_args(argv)

    counts = resolve_counts(args)
    total = sum(counts.values())
    if total == 0:
        raise SystemExit("no samples requested")

    out_dir = args.out
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"

    if args.prompt_file is not None:
        template = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    else:
        default_path = Path(__file__).resolve().parent.parent / "prompt.txt"
        template = (default_path.read_text(encoding="utf-8").strip()
                    if default_path.exists() else DEFAULT_PROMPT)

    rng = np.random.default_rng(args.seed)
    print(f"[INFO] writing {total} samples -> {samples_path}")
    print(f"[INFO] counts: {counts}")
    sid = 0
    t0 = time.time()
    with open(samples_path, "w", encoding="utf-8") as fh:
        for diff in DIFFICULTIES:
            for _ in range(counts[diff]):
                rec = build_one(diff, sid, rng, out_dir, args.max_retries, template)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sid += 1
                if sid % 100 == 0 or sid == total:
                    el = time.time() - t0
                    eta = (total - sid) / max(sid / el, 1e-9)
                    print(f"  [{sid}/{total}] {el:.0f}s ETA={eta:.0f}s")
    print(f"[INFO] done. samples: {samples_path}")
    print(f"[INFO] images dir: {out_dir / 'images'}")
    return samples_path


if __name__ == "__main__":
    main()
