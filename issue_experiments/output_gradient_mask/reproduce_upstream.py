#!/usr/bin/env python3
"""Existing-API reproduction: no fork-specific mask-mode API required.

Copy this file to an unmodified AIHWKit environment and run it there. No data,
optimizer step, new configuration field, or new native binding is required.
Text/JSON/CSV work with this file alone. For --plot also copy plot_results.py
alongside this script and install Matplotlib.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_info(package_path):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(Path(package_path).parent), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        status = git("status", "--porcelain", "--untracked-files=normal")
        return {
            "root": git("rev-parse", "--show-toplevel"),
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(status),
            "status": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"root": None, "commit": None, "dirty": None, "status": None}


def collect(device, variants, experiment, configure=None, check_tile=None):

    import torch
    import aihwkit
    from aihwkit.nn import AnalogLinear
    from aihwkit.simulator import rpu_base
    from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice
    from aihwkit.simulator.parameters import BoundManagementType, NoiseManagementType

    if device == "cuda" and not (torch.cuda.is_available() and rpu_base.cuda.is_compiled()):
        raise RuntimeError("CUDA requested but GPU/native CUDA is unavailable; no CPU fallback")
    torch.set_num_threads(1)
    torch.manual_seed(1)
    result = {
        "schema_version": 1,
        "experiment": experiment,
        "modes": [mode for mode, _ in variants],
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "aihwkit": aihwkit.__version__,
            "package_path": aihwkit.__file__,
            "binding_path": rpu_base.__file__,
            "binding_sha256": sha256(rpu_base.__file__),
            "source": source_info(aihwkit.__file__),
            "device": device,
            "provenance_note": "Paths and hashes identify this run; they do not certify a pristine upstream build.",
        },
        "protocol": {
            "weights": [[1, 1]],
            "inputs": [[1, 1]],
            "bias": False,
            "loss": "sum(output)",
            "inp_bound": 2,
            "out_bound": 1,
            "inp_res": -1,
            "out_res": -1,
            "noise": 0,
            "noise_management": "NONE",
            "backward_is_perfect": True,
            "finite_difference_epsilon": 0.001,
            "optimizer_steps": 0,
        },
        "cases": [],
    }
    if device == "cuda":
        result["environment"]["gpu"] = torch.cuda.get_device_name()
    for case in ("managed", "perfect", "clipped"):
        for mode, mask in variants:
            cfg = SingleRPUConfig(
                device=ConstantStepDevice(
                    w_min=-2,
                    w_max=2,
                    w_min_dtod=0,
                    w_max_dtod=0,
                    dw_min_dtod=0,
                    dw_min_std=0,
                    up_down_dtod=0,
                )
            )
            cfg.forward.is_perfect = case == "perfect"
            cfg.forward.inp_bound = 2
            cfg.forward.out_bound = 1
            cfg.forward.inp_res = cfg.forward.out_res = -1
            cfg.forward.inp_noise = cfg.forward.out_noise = 0
            cfg.forward.noise_management = NoiseManagementType.NONE
            cfg.forward.bound_management = (
                BoundManagementType.ITERATIVE if case == "managed" else BoundManagementType.NONE
            )
            cfg.backward.is_perfect = True
            cfg.mapping.weight_scaling_omega = 0
            if configure is not None:
                configure(cfg, mode)
            layer = AnalogLinear(2, 1, bias=False, rpu_config=cfg).to(device)
            layer.set_weights(torch.ones(1, 2))
            actual_weights = layer.get_weights()[0].detach().cpu()
            if not torch.equal(actual_weights, torch.ones(1, 2)):
                raise RuntimeError("Device did not retain requested weights [1,1]")
            for tile in layer.analog_tiles():
                tile.handle_output_bound = mask
                if check_tile is not None:
                    check_tile(tile, mode)
            x = torch.ones(1, 2, device=device, requires_grad=True)
            y = layer(x)
            y.sum().backward()
            expected = [0.0, 0.0] if case == "clipped" else [1.0, 1.0]
            finite_difference = []
            with torch.no_grad():
                for index in range(2):
                    plus, minus = x.detach().clone(), x.detach().clone()
                    plus[0, index] += 1e-3
                    minus[0, index] -= 1e-3
                    finite_difference.append(float((layer(plus) - layer(minus)).sum() / 2e-3))
            row = {
                "case": case,
                "mode": mode,
                "status": "observed",
                "actual_weights": actual_weights.tolist(),
                "forward_is_perfect": cfg.forward.is_perfect,
                "backward_is_perfect": True,
                "bound_management": cfg.forward.bound_management.name,
                "handle_output_bound": mask,
                "output": y.item(),
                "expected_output": 1.0 if case == "clipped" else 2.0,
                "input_gradient": x.grad.detach().cpu().flatten().tolist(),
                "expected_input_gradient": expected,
                "finite_difference_input_gradient": finite_difference,
            }
            if not all(
                math.isfinite(v)
                for v in [row["output"], *row["input_gradient"], *finite_difference]
            ):
                raise RuntimeError("Non-finite measurement; no comparison figure generated")
            result["cases"].append(row)
    result["status"] = "completed"
    return result


def summary_text(result):
    env = result["environment"]
    lines = [
        f"{result['experiment']} | {env['device']} | {result['status']}",
        f"aihwkit={env['aihwkit']} torch={env['torch']}",
        f"Python package: {env['package_path']}",
        f"Native binding: {env['binding_path']}",
        "W=[1,1], x=[1,1], loss=sum(y), out_bound=1; noise/quantization off",
        "",
        "Case      Mode     F.perf B.perf BM         Y/expected   dX               FD dX            Expected dX",
    ]

    def vec(values):
        return "[" + ",".join(f"{v:.5g}" for v in values) + "]"

    for row in result["cases"]:
        lines.append(
            f"{row['case']:9} {row['mode']:8} {str(row['forward_is_perfect']):6} True   "
            f"{row['bound_management']:10} {row['output']:g}/{row['expected_output']:g}          "
            f"{vec(row['input_gradient']):16} {vec(row['finite_difference_input_gradient']):16} "
            f"{vec(row['expected_input_gradient'])}"
        )
    return "\n".join(lines) + "\n"


def output_paths(output, plot=False):
    output = Path(output)
    if output.suffix.lower() != ".json":
        raise ValueError("--output must have a .json extension")
    if "BEFORE_FIX" in output.name.upper():
        raise ValueError("Choose a new filename; historical BEFORE_FIX evidence is protected")
    paths = [output, output.with_suffix(".csv"), output.with_suffix(".summary.txt")]
    if plot:
        paths += [output.with_suffix(".png"), output.with_suffix(".pdf")]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    return paths


def save_result(result, output):
    paths = output_paths(output)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    with paths[0].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    fields = [
        "case",
        "mode",
        "status",
        "forward_is_perfect",
        "backward_is_perfect",
        "bound_management",
        "handle_output_bound",
        "output",
        "expected_output",
        "dx1",
        "dx2",
        "fd_dx1",
        "fd_dx2",
        "expected_dx1",
        "expected_dx2",
    ]
    with paths[1].open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["cases"]:
            flat = {key: row[key] for key in fields if key in row}
            for field, prefix in [
                ("input_gradient", "dx"),
                ("finite_difference_input_gradient", "fd_dx"),
                ("expected_input_gradient", "expected_dx"),
            ]:
                flat.update({f"{prefix}{i+1}": value for i, value in enumerate(row[field])})
            writer.writerow(flat)
    with paths[2].open("x", encoding="utf-8") as stream:
        stream.write(summary_text(result))


def execute(args, variants, experiment, configure=None, check_tile=None):
    if args.plot and not args.output:
        raise ValueError("--plot requires --output")
    if args.output:
        output_paths(args.output, args.plot)
    renderer = None
    if args.plot:
        try:
            from plot_results import render_result
            import matplotlib  # noqa: F401 -- check before measurement

            renderer = render_result
        except ImportError as exc:
            raise RuntimeError(
                "For --plot, copy plot_results.py alongside this script and install matplotlib"
            ) from exc
    result = collect(args.device, variants, experiment, configure, check_tile)
    tool_paths = [Path(__file__).resolve(), Path(sys.argv[0]).resolve()]
    if renderer is not None:
        tool_paths.append(Path(__file__).with_name("plot_results.py"))
    result["tool_sha256"] = {str(p): sha256(p) for p in tool_paths if p.is_file()}
    print(
        json.dumps(result, indent=2)
        if getattr(args, "json_stdout", False)
        else summary_text(result)
    )
    if args.output:
        save_result(result, args.output)
        print(f"Results: {args.output}", file=sys.stderr)
    if renderer is not None:
        renderer(result, Path(args.output).with_suffix(""))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", action="store_true", help="Also save PNG/PDF; requires --output")
    args = parser.parse_args()
    try:
        return execute(args, [("legacy", True), ("disabled", False)], "upstream_mask_toggle")
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
