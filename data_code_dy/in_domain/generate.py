"""IND data generator CLI.

Produces samples.jsonl + images/ matching the dev-set schema, ready to be
consumed by:
  - symreg_experiments/scripts/build_symreg_data.py (SFT/RL conversion)
  - 0518_test_liubw/eval.py (optional eval on generated data)

Example:
    python in_domain/generate.py --n 1000 --seed 42 --out in_domain/output/ind_v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from in_domain import expressions as expr_mod  # noqa: E402
    from in_domain import sampling as samp_mod     # noqa: E402
    from in_domain.rendering import render_png     # noqa: E402
    from in_domain.prompt_template import (        # noqa: E402
        DEFAULT_PROMPT_PATH,
        load_template,
        render_prompt,
    )
else:
    from . import expressions as expr_mod
    from . import sampling as samp_mod
    from .rendering import render_png
    from .prompt_template import DEFAULT_PROMPT_PATH, load_template, render_prompt

DEFAULT_TOTAL = 1000
Y_ABS_MAX = 1e6
Y_VAR_MIN = 1e-6
DEFAULT_RETRIES = 50


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
        if k not in expr_mod.DIFFICULTIES:
            raise argparse.ArgumentTypeError(f"unknown difficulty {k!r}")
        out[k] = int(v)
    return out


def resolve_counts(args: argparse.Namespace) -> dict[str, int]:
    if args.counts:
        c = parse_counts(args.counts)
        for d in expr_mod.DIFFICULTIES:
            c.setdefault(d, 0)
        return {d: c[d] for d in expr_mod.DIFFICULTIES}
    n = args.n
    per = n // len(expr_mod.DIFFICULTIES)
    rem = n - per * len(expr_mod.DIFFICULTIES)
    counts = {d: per for d in expr_mod.DIFFICULTIES}
    for d in list(expr_mod.DIFFICULTIES)[:rem]:
        counts[d] += 1
    return counts


def validate(y_ref: np.ndarray, y_test: np.ndarray) -> bool:
    if not (np.all(np.isfinite(y_ref)) and np.all(np.isfinite(y_test))):
        return False
    if float(np.var(y_test)) <= Y_VAR_MIN:
        return False
    if float(np.max(np.abs(y_test))) > Y_ABS_MAX:
        return False
    if float(np.max(np.abs(y_ref))) > Y_ABS_MAX:
        return False
    return True


def build_one(
    diff: str,
    sample_id: int,
    rng: np.random.Generator,
    out_dir: Path,
    max_retries: int,
    template: str,
) -> dict:
    n_lo, n_hi = samp_mod.WINDOWS[diff]["ref_n_range"]
    last_err: str | None = None
    for _ in range(max_retries):
        expr_np, expr_str, true_hints = expr_mod.sample_expression(diff, rng)
        ref_win = samp_mod.pick_ref_window(diff, rng)
        test_win = samp_mod.pick_test_window(diff, ref_win, rng)
        n_ref = int(rng.integers(n_lo, n_hi + 1))
        ref_xs = samp_mod.make_ref_xs(*ref_win, n=n_ref, rng=rng)
        test_xs = samp_mod.make_test_xs(*test_win, n=samp_mod.TEST_N)
        try:
            y_ref = samp_mod.eval_expr(expr_np, ref_xs)
            y_test = samp_mod.eval_expr(expr_np, test_xs)
        except Exception as exc:
            last_err = str(exc)
            continue
        if not validate(y_ref, y_test):
            last_err = "validation failed"
            continue
        img_rel = f"images/ind_{diff}_{sample_id:04d}.png"
        render_png(expr_np, *ref_win, out_dir / img_rel)
        ref_pts = [[xv, float(round(yv, 6))] for xv, yv in zip(ref_xs, y_ref.tolist())]
        test_pts = [
            [float(round(xv, 6)), float(round(yv, 6))]
            for xv, yv in zip(test_xs, y_test.tolist())
        ]
        function_hints = expr_mod.make_function_hints(true_hints, rng, diff)
        return {
            "id": sample_id,
            "split": "ind",
            "expression_str": expr_str,
            "expression_numpy": expr_np,
            "true_function_hints": true_hints,
            "function_hints": function_hints,
            "data_points_text": ref_pts,
            "image_path": img_rel,
            "prompt": render_prompt(
                {"function_hints": function_hints, "data_points_text": ref_pts},
                template,
            ),
            "test_points": test_pts,
        }
    raise RuntimeError(
        f"failed to build {diff} sample {sample_id} after {max_retries} retries "
        f"(last error: {last_err})"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=DEFAULT_TOTAL,
                   help=f"total samples when --counts unset (default {DEFAULT_TOTAL})")
    p.add_argument("--counts", type=str, default=None,
                   help="per-difficulty counts, e.g. easy=200,medium=200,...")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True,
                   help="output directory; samples.jsonl + images/ written inside")
    p.add_argument("--max-retries", type=int, default=DEFAULT_RETRIES)
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help=f"prompt template file (default: {DEFAULT_PROMPT_PATH})",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    counts = resolve_counts(args)
    total = sum(counts.values())
    if total == 0:
        raise SystemExit("no samples requested")
    out_dir = args.out
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    template = load_template(args.prompt_file)
    rng = np.random.default_rng(args.seed)
    print(f"[INFO] writing {total} samples -> {samples_path}")
    print(f"[INFO] counts: {counts}")
    print(f"[INFO] prompt template: {args.prompt_file or DEFAULT_PROMPT_PATH}")
    sid = 0
    t0 = time.time()
    with open(samples_path, "w", encoding="utf-8") as fh:
        for diff in expr_mod.DIFFICULTIES:
            for _ in range(counts[diff]):
                rec = build_one(diff, sid, rng, out_dir, args.max_retries, template)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sid += 1
                if sid % 50 == 0 or sid == total:
                    el = time.time() - t0
                    eta = (total - sid) / max(sid / el, 1e-9)
                    print(f"  [{sid}/{total}] {el:.0f}s ETA={eta:.0f}s")
    print(f"[INFO] done. samples: {samples_path}")
    print(f"[INFO] images dir: {out_dir / 'images'}")
    print(
        "[INFO] next: python "
        "/inspire/qb-ilm2/project/26summer-camp-21/26210880/symreg_experiments/scripts/build_symreg_data.py "
        f"--samples {samples_path} --out-dir {out_dir / 'sft_rl'}"
    )
    return samples_path


if __name__ == "__main__":
    main()
