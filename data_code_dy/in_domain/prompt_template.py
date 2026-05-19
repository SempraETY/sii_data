"""Prompt rendering for the in_domain symbolic-regression generator.

Mirrors the substitution contract in 0518_test_liubw/eval.py (build_message): the
caller is expected to keep ``prompt.txt`` placeholder-compatible with eval.py,
and this module is the single place where samples get turned into user-visible
prompt text.

Public API:
    DEFAULT_PROMPT_PATH   -- absolute path to in_domain/prompt.txt
    load_template(path)   -- read + strip the template file
    render_prompt(sample, template) -- substitute {function_hints}{data_points}{axis_note}
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompt.txt"


def load_template(path: Optional[Path] = None) -> str:
    """Load and strip the prompt template.

    Raises FileNotFoundError with an actionable message when the file is
    missing -- the generator treats prompt.txt as a required input, so a
    silent fallback would mask user-visible config drift.
    """
    p = Path(path) if path is not None else DEFAULT_PROMPT_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"prompt template not found: {p} "
            f"(expected the eval.py-compatible template at this path)"
        )
    return p.read_text(encoding="utf-8").strip()


def render_prompt(sample: dict, template: str) -> str:
    """Render one sample into a prompt string.

    Field-by-field substitution is identical to eval.py's build_message so
    that data generated here is byte-equivalent to what eval.py produces at
    inference time.
    """
    hints = sample.get("function_hints", [])
    function_hints = (
        "Available functions: " + ", ".join(hints) + "\n" if hints else ""
    )

    pts = sample.get("data_points_text", [])
    if pts:
        data_points = (
            "Reference points: "
            + "  ".join(f"({x:.4f}, {y:.4f})" for x, y in pts)
            + "\n"
        )
    else:
        data_points = ""

    axis_note = ""

    return template.format(
        function_hints=function_hints,
        data_points=data_points,
        axis_note=axis_note,
    )
