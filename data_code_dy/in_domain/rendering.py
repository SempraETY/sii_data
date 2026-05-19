"""Render a function curve to a 590x390 PNG, matching dev image style.

Dev images: 590×390 RGBA, white background, light-grey grid lines (#ededed),
steelblue curve, black tick labels and spines, axis labels "x" / "y".
Effectively matplotlib default style + ax.grid(True) with a custom grey color.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .sampling import EVAL_GLOBALS

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
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.set_size_inches(*FIGSIZE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=DPI)
    plt.close(fig)
