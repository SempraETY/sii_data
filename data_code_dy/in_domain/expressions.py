
"""Per-difficulty expression generation, learning weights from dev set.

All sampling probabilities (which periodic to pick, whether to wrap with abs /
log / sqrt, how many top-level terms, how long function_hints should be, which
distractors to draw) come from in_domain.dev_distribution, which reads
0518_test_liubw/data/task/dev/samples.jsonl with Laplace +0.1 smoothing —
unseen functions still have ~0.2% probability, matching the user direction
that "unseen in 300 samples ≠ impossible".
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np

from . import dev_distribution as dd

DIFFICULTIES = ("easy", "medium", "hard", "expert", "extreme")

CONSTS_INT = [1, 2, 3]
CONSTS_FRAC = [0.5, 1.5, 0.3, 0.8, 0.2]
CONSTS_BIG = [4, 5, 0]
CONST_GROUP_WEIGHTS = (0.7, 0.2, 0.1)

# Functions safe to call on raw `b * x` (no domain restriction); these are
# what fills the "periodic-style" slot inside templates.
SAFE_FUNCS = ["np.sin", "np.cos", "np.tanh", "np.tan", "np.sinh", "np.cosh", "np.arctan"]

# dev's true_function_hints does not list np.abs — abs is treated as a
# structural wrapper, not a base function. We mirror that convention.
_TRUE_HINT_EXCLUDE = {"np.abs"}


def sample_constant(rng: np.random.Generator, allow_zero: bool = False) -> float:
    group = rng.choice(3, p=CONST_GROUP_WEIGHTS)
    pool = (CONSTS_INT, CONSTS_FRAC, CONSTS_BIG)[group]
    val = pool[rng.integers(len(pool))]
    if val == 0 and not allow_zero:
        return 1
    return float(val) if isinstance(val, float) else int(val)


def signed(rng: np.random.Generator, val: float) -> float:
    return -val if rng.random() < 0.5 else val


def _fmt(v: float) -> str:
    if isinstance(v, int) or v == int(v):
        return str(int(v))
    return f"{v:g}"


def _pick_safe(diff: str, rng: np.random.Generator) -> str:
    return dd.pick_function(dd.expr_function_weights(diff), rng, among=SAFE_FUNCS)


def _bernoulli(p: float, rng: np.random.Generator) -> bool:
    return bool(rng.random() < p)


def _build_atom(diff: str, rng: np.random.Generator, allow_pow: bool) -> str:
    """One periodic-ish atom: a*f(b*x) or a*f(b*x**p) or a*f(b*x+c)."""
    a = sample_constant(rng)
    b = sample_constant(rng)
    f = _pick_safe(diff, rng)
    flags = dd.structural_flags(diff)
    use_pow = allow_pow and _bernoulli(flags["has_pow"], rng)
    if use_pow:
        p = int(rng.choice([2, 3]))
        return f"{_fmt(a)} * {f}({_fmt(b)} * x ** {p})"
    if _bernoulli(0.4, rng):
        c = sample_constant(rng)
        return f"{_fmt(a)} * {f}({_fmt(b)} * x + {_fmt(c)})"
    return f"{_fmt(a)} * {f}({_fmt(b)} * x)"


def _exp_factor(rng: np.random.Generator, force_neg_xsq: bool = False) -> str:
    if force_neg_xsq or _bernoulli(0.5, rng):
        c = float(rng.choice([0.2, 0.3, 0.5, 0.8]))
        return f"np.exp(-{_fmt(c)} * x ** 2)"
    b = float(rng.choice([0.3, 0.5, 0.8, 1.0]))
    b = signed(rng, b)
    return f"np.exp({_fmt(b)} * x)"


def _wrap_abs(inner: str) -> str:
    return f"np.abs({inner}) + 1"


def _wrap_log(inner: str) -> str:
    return f"np.log({_wrap_abs(inner)})"


def _wrap_sqrt(inner: str) -> str:
    return f"np.sqrt({_wrap_abs(inner)})"


def _easy(rng: np.random.Generator, diff: str = "easy") -> str:
    """Single-term, possibly with phase or exp(b*x)."""
    flags = dd.structural_flags(diff)
    a = sample_constant(rng)
    if _bernoulli(flags["has_exp_neg_xsq"] + 0.1, rng):
        return f"{_fmt(a)} * {_exp_factor(rng)}"
    if _bernoulli(0.25, rng):
        b = float(rng.choice([0.3, 0.5, 0.8, 1.0, 1.5]))
        b = signed(rng, b)
        return f"{_fmt(a)} * np.exp({_fmt(b)} * x)"
    return _build_atom(diff, rng, allow_pow=False)


def _medium(rng: np.random.Generator, diff: str = "medium") -> str:
    """Dev medium: 1 functional atom + tail of `c*x + d` style additions."""
    flags = dd.structural_flags(diff)
    n_terms = dd.sample_n_terms(diff, rng)  # mostly 2, sometimes 3
    if _bernoulli(flags["has_log"], rng):
        head = (f"{_fmt(sample_constant(rng))} * "
                + _wrap_log(_build_atom(diff, rng, allow_pow=False)))
    elif _bernoulli(flags["has_abs"], rng):
        head = (f"{_fmt(sample_constant(rng))} * "
                + f"np.abs({_pick_safe(diff, rng)}({_fmt(sample_constant(rng))} * x))")
    elif _bernoulli(flags["has_exp_neg_xsq"], rng):
        head = f"{_fmt(sample_constant(rng))} * {_exp_factor(rng, force_neg_xsq=True)}"
    else:
        head = _build_atom(diff, rng, allow_pow=False)
    parts = [head]
    while len(parts) < n_terms:
        choice = rng.random()
        if choice < 0.5:
            parts.append(f"{_fmt(sample_constant(rng))} * x")
        else:
            parts.append(_fmt(signed(rng, sample_constant(rng))))
    return " + ".join(parts)


def _hard(rng: np.random.Generator, diff: str = "hard") -> str:
    flags = dd.structural_flags(diff)
    if _bernoulli(flags["has_exp_neg_xsq"] + 0.1, rng):
        a = sample_constant(rng)
        return f"{_fmt(a)} * {_exp_factor(rng, force_neg_xsq=True)} * "\
               f"{_pick_safe(diff, rng)}({_fmt(sample_constant(rng))} * x)"
    if _bernoulli(0.4, rng):
        a = sample_constant(rng)
        b = signed(rng, float(rng.choice([0.3, 0.5, 0.8, 1.0])))
        return f"{_fmt(a)} * np.exp({_fmt(b)} * x) * "\
               f"{_pick_safe(diff, rng)}({_fmt(sample_constant(rng))} * x)"
    if _bernoulli(flags["has_pow"], rng):
        return _build_atom(diff, rng, allow_pow=True)
    if _bernoulli(flags["has_sqrt"], rng):
        return f"{_fmt(sample_constant(rng))} * "\
               f"{_wrap_sqrt(_build_atom(diff, rng, allow_pow=False))}"
    return _build_atom(diff, rng, allow_pow=False)


def _expert(rng: np.random.Generator, diff: str = "expert") -> str:
    flags = dd.structural_flags(diff)
    n_terms = dd.sample_n_terms(diff, rng)
    parts: list[str] = []
    for _ in range(n_terms):
        choice = rng.random()
        if choice < flags["has_log"]:
            atom_inner = _build_atom(diff, rng, allow_pow=_bernoulli(flags["has_pow"], rng))
            parts.append(f"{_fmt(sample_constant(rng))} * "
                         + _wrap_log(f"{_fmt(sample_constant(rng))} + {atom_inner}"))
        elif choice < flags["has_log"] + flags["has_sqrt"]:
            parts.append(f"{_fmt(sample_constant(rng))} * "
                         + _wrap_sqrt(_build_atom(diff, rng, allow_pow=False)))
        elif choice < flags["has_log"] + flags["has_sqrt"] + flags["has_exp_neg_xsq"]:
            parts.append(f"{_fmt(sample_constant(rng))} * {_exp_factor(rng, force_neg_xsq=True)}")
        else:
            parts.append(_build_atom(diff, rng, allow_pow=_bernoulli(flags["has_pow"], rng)))
    expr = " + ".join(parts)
    if _bernoulli(0.2, rng):
        expr += f" + {_fmt(sample_constant(rng))} * x"
    return expr


def _extreme(rng: np.random.Generator, diff: str = "extreme") -> str:
    flags = dd.structural_flags(diff)
    n_terms = dd.sample_n_terms(diff, rng)
    parts: list[str] = []
    for i in range(n_terms):
        if i == 0 and _bernoulli(flags["has_log"] + 0.1, rng):
            inner_atom = _build_atom(diff, rng, allow_pow=_bernoulli(flags["has_pow"], rng))
            parts.append(_wrap_log(f"{inner_atom} + {_fmt(sample_constant(rng))}"))
        elif i == 0 and _bernoulli(flags["has_exp_neg_xsq"], rng):
            f1 = _pick_safe(diff, rng)
            parts.append(f"{_fmt(sample_constant(rng))} * "
                         f"np.exp(-{_fmt(float(rng.choice([0.3, 0.5, 0.8])))} "
                         f"* {f1}({_fmt(sample_constant(rng))} * x) ** 2)")
        elif _bernoulli(0.3, rng):
            f1 = _pick_safe(diff, rng)
            f2 = _pick_safe(diff, rng)
            parts.append(f"{_fmt(sample_constant(rng))} * "
                         f"{f1}({_fmt(sample_constant(rng))} * x + "
                         f"{_fmt(sample_constant(rng))} * {f2}({_fmt(sample_constant(rng))} * x))")
        else:
            parts.append(_build_atom(diff, rng, allow_pow=_bernoulli(flags["has_pow"], rng)))
    return " + ".join(parts)


_SAMPLERS: dict[str, Callable[[np.random.Generator, str], str]] = {
    "easy": _easy,
    "medium": _medium,
    "hard": _hard,
    "expert": _expert,
    "extreme": _extreme,
}


_NP_FN_RE = re.compile(
    r"np\.(sin|cos|exp|sqrt|log|abs|tanh|arctan|arcsin|arccos|sinh|tan|cosh)\b"
)


def extract_true_hints(expr_numpy: str) -> list[str]:
    """Functions to put in true_function_hints. Mirrors dev convention:
    np.abs is excluded (treated as a structural wrapper, not a base function)."""
    seen: list[str] = []
    for m in _NP_FN_RE.finditer(expr_numpy):
        name = "np." + m.group(1)
        if name in _TRUE_HINT_EXCLUDE:
            continue
        if name not in seen:
            seen.append(name)
    return seen


def to_plain(expr_numpy: str) -> str:
    return re.sub(r"\bnp\.", "", expr_numpy)


def sample_expression(
    diff: str, rng: np.random.Generator,
) -> tuple[str, str, list[str]]:
    if diff not in _SAMPLERS:
        raise ValueError(f"unknown difficulty: {diff}")
    expr_np = _SAMPLERS[diff](rng, diff)
    return expr_np, to_plain(expr_np), extract_true_hints(expr_np)


def make_function_hints(
    true_hints: list[str],
    rng: np.random.Generator,
    diff: str,
) -> list[str]:
    """Build prompt function_hints by sampling target length + distractors
    from dev distribution per difficulty."""
    return dd.sample_distractors(diff, true_hints, rng)
