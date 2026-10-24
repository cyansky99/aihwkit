#!/usr/bin/env python3
"""Compatibility entry point for the fork-specific three-mode comparison.

Requires this fork's Python API and rebuilt extension. For an unmodified upstream
installation use issue_experiments/output_gradient_mask/reproduce_upstream.py.
Old result files are retained. New runs use the public comparison JSON schema.
"""
from pathlib import Path
import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "issue_experiments/output_gradient_mask")
)
from compare_modes import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
