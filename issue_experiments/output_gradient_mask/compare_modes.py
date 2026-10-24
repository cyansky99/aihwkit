#!/usr/bin/env python3
# Licensed under the MIT license. See LICENSE.txt in the project root.
"""Compare legacy/native/disabled: requires this fork's Python and rebuilt binding.

For an existing upstream installation use reproduce_upstream.py instead.
Missing capabilities never silently substitute another policy.
"""
import argparse
from pathlib import Path
from reproduce_upstream import execute

BUILD_HINT = "Build this fork's extension and use PYTHONPATH=src; see the reproduction README."


def configure(cfg, mode):
    if not hasattr(cfg.forward, "output_gradient_mask_mode"):
        raise RuntimeError(f"Missing experimental Python mask configuration. {BUILD_HINT}")
    cfg.forward.output_gradient_mask_mode = mode


def check_tile(tile, mode):
    if not hasattr(tile, "get_output_gradient_mask_modes"):
        raise RuntimeError(f"Missing experimental tile API. {BUILD_HINT}")
    if mode == "native" and not hasattr(tile.tile, "forward_with_output_mask"):
        raise RuntimeError(f"Missing forward_with_output_mask native binding. {BUILD_HINT}")
    if tile.get_output_gradient_mask_modes()[1] != mode:
        raise RuntimeError(f"Requested {mode}, but the tile applied a different policy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", action="store_true", help="Also save PNG/PDF; requires --output")
    parser.add_argument(
        "--mask-mode", choices=("all", "legacy", "native", "disabled"), default="all"
    )
    parser.add_argument("--json-stdout", action="store_true")
    args = parser.parse_args()
    modes = ("legacy", "native", "disabled") if args.mask_mode == "all" else (args.mask_mode,)
    try:
        return execute(args, [(mode, True) for mode in modes], "three_modes", configure, check_tile)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
