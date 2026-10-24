#!/usr/bin/env python3
"""Minimal public-API reproducer: works without this fork's mask-mode API.

Copy this file to an unmodified AIHWKit environment and run it there. No data,
optimizer step, new configuration field, or new native binding is required.
"""
import argparse
import json
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new filename to preserve evidence.")

    import torch
    import aihwkit
    from aihwkit.nn import AnalogLinear
    from aihwkit.simulator import rpu_base
    from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice
    from aihwkit.simulator.parameters import BoundManagementType, NoiseManagementType

    if args.device == "cuda" and not (torch.cuda.is_available() and rpu_base.cuda.is_compiled()):
        parser.error("CUDA requested but GPU/native CUDA is unavailable")
    torch.set_num_threads(1)
    result = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "aihwkit": aihwkit.__version__,
            "package_path": aihwkit.__file__,
            "binding_path": rpu_base.__file__,
            "device": args.device,
        },
        "cases": [],
    }
    if args.device == "cuda":
        result["environment"]["gpu"] = torch.cuda.get_device_name()
    print(json.dumps(result["environment"], indent=2))
    print("W=[1,1], x=[1,1], loss=sum(y), out_bound=1; noise/quantization off")
    print("Case               F.perf B.perf BM         Mask   Y    dX        Expected dX")
    for case in ("managed", "perfect", "clipped"):
        for mask in (True, False):
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
            layer = AnalogLinear(2, 1, bias=False, rpu_config=cfg).to(args.device)
            layer.set_weights(torch.ones(1, 2))
            for tile in layer.analog_tiles():
                tile.handle_output_bound = mask
            x = torch.ones(1, 2, device=args.device, requires_grad=True)
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
            result["cases"].append(row)
            print(
                f"{case:18} {str(cfg.forward.is_perfect):6} True   "
                f"{row['bound_management']:10} {str(mask):5} {y.item():4g} "
                f"{str(row['input_gradient']):9} {expected}"
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(f"Full results: {args.output}")


if __name__ == "__main__":
    main()
