"""Per-difficulty empirical distributions learned from the dev set.

The dev set at
  /inspire/qb-ilm2/project/26summer-camp-21/26210880/0518_test_liubw/data/task/dev/samples.jsonl
is the ground-truth distribution. This module reads it once at import time
and exposes per-difficulty sampling weights for:

- function presence in expression_numpy (np.sin, np.cos, ..., 13-fn pool)
- function presence in true_function_hints
- function presence in function_hints (prompt 池)
- structural flags: has_pow / has_abs / has_log / has_sqrt / has_exp_neg_xsq
- top-level term count
- function_hints length

A Laplace +0.1 smoothing is applied so a function unseen in 60 dev samples
of a given difficulty can still be sampled with small probability
(unseen != impossible, per user direction).
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DEV_PATH = Path(
    "/inspire/qb-ilm2/project/26summer-camp-21/26210880/"
    "0518_test_liubw/data/task/dev/samples.jsonl"
)

ALL_FUNCS = (
    "np.sin", "np.cos", "np.exp", "np.sqrt", "np.log", "np.abs",
    "np.tanh", "np.arctan", "np.arcsin", "np.arccos",
    "np.sinh", "np.tan", "np.cosh",
)

LAPLACE_ALPHA = 0.1

_FN_RE = re.compile(r"np\.([a-zA-Z_]\w*)")


def _difficulty(image_path: str) -> str:
    return Path(image_path).name.split("_")[1]


def _top_terms(expr: str) -> int:
    depth = 0
    n = 1
    prev = " "
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and ch in "+-" and prev not in "*+-/(^ ":
            n += 1
        if ch != " ":
            prev = ch
    return n


def _funcs_in(expr: str) -> set[str]:
    return {"np." + m.group(1) for m in _FN_RE.finditer(expr)}


def _build():
    """Returns dict[diff] -> dict of empirical statistics + smoothed weights."""
    by_diff: dict[str, list[dict]] = defaultdict(list)
    if not DEV_PATH.exists():
        raise FileNotFoundError(f"dev set not found: {DEV_PATH}")
    for line in DEV_PATH.read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        by_diff[_difficulty(s["image_path"])].append(s)

    out: dict[str, dict] = {}
    for diff, samples in by_diff.items():
        n = len(samples)
        # function presence in 3 sources
        expr_freq: Counter = Counter()
        true_hint_freq: Counter = Counter()
        prompt_hint_freq: Counter = Counter()
        for s in samples:
            for fn in _funcs_in(s["expression_numpy"]):
                expr_freq[fn] += 1
            for fn in set(s.get("true_function_hints") or []):
                true_hint_freq[fn] += 1
            for fn in set(s.get("function_hints") or []):
                prompt_hint_freq[fn] += 1
        # structural flags
        flags = {
            "has_pow":         sum(1 for s in samples if "**" in s["expression_numpy"]) / n,
            "has_abs":         sum(1 for s in samples if "np.abs" in s["expression_numpy"]) / n,
            "has_log":         sum(1 for s in samples if "np.log" in s["expression_numpy"]) / n,
            "has_sqrt":        sum(1 for s in samples if "np.sqrt" in s["expression_numpy"]) / n,
            "has_exp_neg_xsq": sum(1 for s in samples if re.search(r"np\.exp\s*\(\s*-", s["expression_numpy"])) / n,
        }
        term_counts = Counter(_top_terms(s["expression_numpy"]) for s in samples)
        hint_lens = Counter(len(s.get("function_hints") or []) for s in samples)

        out[diff] = {
            "n": n,
            "expr_freq":        {fn: c / n for fn, c in expr_freq.items()},
            "true_hint_freq":   {fn: c / n for fn, c in true_hint_freq.items()},
            "prompt_hint_freq": {fn: c / n for fn, c in prompt_hint_freq.items()},
            "expr_w":         _smoothed_weights(expr_freq, n),
            "true_hint_w":    _smoothed_weights(true_hint_freq, n),
            "prompt_hint_w":  _smoothed_weights(prompt_hint_freq, n),
            "flags": flags,
            "term_pmf":  {k: v / n for k, v in term_counts.items()},
            "hint_len_pmf": {k: v / n for k, v in hint_lens.items()},
        }
    return out


def _smoothed_weights(counter: Counter, n: int) -> dict[str, float]:
    """Laplace +0.1 over the 13-function pool. Returns sample-presence weights
    (NOT normalized to sum 1). Used by callers as independent Bernoullis."""
    return {
        fn: (counter.get(fn, 0) + LAPLACE_ALPHA) / (n + LAPLACE_ALPHA * len(ALL_FUNCS))
        for fn in ALL_FUNCS
    }


DEV_STATS = _build()


def expr_function_weights(diff: str) -> dict[str, float]:
    """Smoothed P(np.f appears in expression_numpy | difficulty)."""
    return DEV_STATS[diff]["expr_w"]


def true_hint_weights(diff: str) -> dict[str, float]:
    return DEV_STATS[diff]["true_hint_w"]


def prompt_hint_weights(diff: str) -> dict[str, float]:
    return DEV_STATS[diff]["prompt_hint_w"]


def structural_flags(diff: str) -> dict[str, float]:
    return DEV_STATS[diff]["flags"]


def sample_n_terms(diff: str, rng: np.random.Generator) -> int:
    pmf = DEV_STATS[diff]["term_pmf"]
    keys = list(pmf.keys())
    probs = np.array([pmf[k] for k in keys], dtype=float)
    probs /= probs.sum()
    return int(rng.choice(keys, p=probs))


def sample_hint_len(diff: str, rng: np.random.Generator) -> int:
    pmf = DEV_STATS[diff]["hint_len_pmf"]
    keys = list(pmf.keys())
    probs = np.array([pmf[k] for k in keys], dtype=float)
    probs /= probs.sum()
    return int(rng.choice(keys, p=probs))


def pick_function(
    weights: dict[str, float],
    rng: np.random.Generator,
    among: list[str] | None = None,
) -> str:
    """Sample one function from `weights`, optionally restricted to `among`."""
    keys = among if among is not None else list(weights.keys())
    probs = np.array([weights.get(k, 0.0) for k in keys], dtype=float)
    s = probs.sum()
    if s <= 0:
        # all-zero (shouldn't happen with smoothing); fall back to uniform
        probs = np.ones_like(probs)
        s = probs.sum()
    probs /= s
    return str(rng.choice(keys, p=probs))


def sample_distractors(
    diff: str,
    true_hints: list[str],
    rng: np.random.Generator,
) -> list[str]:
    """Build prompt function_hints by:
    1. start with true_hints (preserve order)
    2. sample target length from dev hint_len_pmf
    3. fill remaining slots by drawing from dev prompt_hint distribution,
       excluding already-included functions.
    """
    target_len = sample_hint_len(diff, rng)
    selected = list(dict.fromkeys(true_hints))[:target_len]
    if len(selected) >= target_len:
        return selected
    pw = prompt_hint_weights(diff)
    remaining = [fn for fn in ALL_FUNCS if fn not in selected]
    while len(selected) < target_len and remaining:
        # weighted draw without replacement
        probs = np.array([pw.get(fn, LAPLACE_ALPHA) for fn in remaining], dtype=float)
        s = probs.sum()
        if s <= 0:
            break
        probs /= s
        idx = int(rng.choice(len(remaining), p=probs))
        selected.append(remaining.pop(idx))
    return selected
