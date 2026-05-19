"""End-to-end smoke test for the IND data generator.

Generates a tiny dataset and verifies:
  1. Each sample's ground-truth expression yields R² >= 0.999 on its own
     test_points (matches the way eval.py grades predictions).
  2. Each generated PNG is exactly 590x390 (matches dev image dims).
  3. The downstream SFT/RL builder (build_symreg_data.py) parses the file
     without errors and emits the expected number of records.

Run:
    python in_domain/smoke_test.py [--out <tmp_dir>] [--n 10]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from in_domain.generate import main as generate_main  # noqa: E402

BUILD_SYMREG_DATA = Path(
    "/inspire/qb-ilm2/project/26summer-camp-21/26210880/"
    "symreg_experiments/scripts/build_symreg_data.py"
)


def _png_dims(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    return struct.unpack(">II", data[16:24])


def _r2(expr: str, points: list[list[float]]) -> float:
    pts = points[:50]
    x = np.array([p[0] for p in pts], dtype=float)
    y_true = np.array([p[1] for p in pts], dtype=float)
    y_pred = eval(expr, {"x": x, "np": np, "__builtins__": {}})
    if np.isscalar(y_pred):
        y_pred = np.full_like(x, float(y_pred))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def _load_build_symreg_data():
    if not BUILD_SYMREG_DATA.exists():
        return None
    spec = importlib.util.spec_from_file_location("_bsd", BUILD_SYMREG_DATA)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def run(out: Path, n: int) -> None:
    if out.exists():
        shutil.rmtree(out)
    samples_path = generate_main(["--n", str(n), "--seed", "0", "--out", str(out)])

    records = [json.loads(l) for l in samples_path.read_text().splitlines() if l.strip()]
    assert len(records) == n, f"expected {n} records, got {len(records)}"

    failures = []
    for s in records:
        r2 = _r2(s["expression_numpy"], s["test_points"])
        w, h = _png_dims(out / s["image_path"])
        if r2 < 0.999:
            failures.append(("r2", s["id"], r2, s["expression_numpy"]))
        if (w, h) != (590, 390):
            failures.append(("dims", s["id"], (w, h)))
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(1)
    print(f"PASS: {n} samples, all R² >= 0.999, all PNGs 590x390")

    bsd = _load_build_symreg_data()
    if bsd is None:
        print("SKIP: build_symreg_data.py not found; downstream check skipped")
        return
    rows = list(bsd.build_rows(samples_path))
    assert len(rows) == n, f"build_symreg_data parsed {len(rows)} != {n}"
    diffs_seen = {meta["difficulty"] for _, _, _, meta in rows}
    expected_diffs = {"easy", "medium", "hard", "expert", "extreme"}
    if not diffs_seen <= expected_diffs:
        raise SystemExit(f"unexpected difficulties: {diffs_seen}")
    print(f"PASS: build_symreg_data parsed {n} rows; difficulties seen: {sorted(diffs_seen)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=Path(__file__).parent / "output" / "_smoke")
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args()
    run(args.out, args.n)


if __name__ == "__main__":
    main()
