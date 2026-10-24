#!/usr/bin/env python3
"""ConstantStep MNIST: historical low-noise settings, three output-mask modes.

The name 'native' refers only to this fork's experimental output mask.
No simulator source changes or extra diagnostic forward passes are required.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
MODES = ("legacy", "native", "disabled")
COLORS = dict(legacy="#dc8b27", native="#287bb5", disabled="#369552")


def encode(obj):
    if is_dataclass(obj):
        return {f.name: encode(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__qualname__}"
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    return obj


def fingerprint(obj):
    return hashlib.sha256(json.dumps(encode(obj), sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(encode(obj), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def make_config(mode, pulse_noise, seed):
    from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice
    from aihwkit.simulator.parameters.enums import (
        AnalogMVType, BoundManagementType, NoiseManagementType, PulseType,
    )
    if mode not in MODES or not math.isfinite(pulse_noise) or pulse_noise < 0:
        raise ValueError("Invalid mask mode or pulse noise")
    cfg = SingleRPUConfig(device=ConstantStepDevice(
        construction_seed=seed + 1, dw_min=0.002, w_min=-2, w_max=2,
        dw_min_std=pulse_noise, dw_min_dtod=0, w_min_dtod=0, w_max_dtod=0,
        up_down=0, up_down_dtod=0, reset_std=0,
    ))
    if not hasattr(cfg.forward, "output_gradient_mask_mode"):
        raise RuntimeError("Build this fork to use its three output-gradient mask modes")
    for io in (cfg.forward, cfg.backward):
        io.is_perfect = False
        io.mv_type = AnalogMVType.ONE_PASS
        io.inp_res = io.out_res = 1 / 4094
        io.inp_bound, io.out_bound, io.out_noise = 1, 12, 0
        io.noise_management = NoiseManagementType.ABS_MAX
        io.bound_management = BoundManagementType.ITERATIVE
    cfg.forward.output_gradient_mask_mode = mode
    cfg.mapping.digital_bias = True
    cfg.mapping.max_input_size = cfg.mapping.max_output_size = 512
    cfg.mapping.weight_scaling_omega = 0
    cfg.mapping.weight_scaling_columnwise = False
    cfg.mapping.learn_out_scaling = False
    cfg.update.desired_bl = 31
    cfg.update.fixed_bl = True
    cfg.update.update_management = cfg.update.update_bl_management = True
    cfg.update.pulse_type = PulseType.STOCHASTIC_COMPRESSED
    return cfg


def make_model(cfg, seed, device):
    import torch
    from torch import nn
    from aihwkit.nn import AnalogLinear, AnalogSequential
    shapes = [(784, 256), (256, 128), (128, 10)]
    layers = [AnalogLinear(i, o, True, rpu_config=deepcopy(cfg)) for i, o in shapes]
    model = AnalogSequential(layers[0], nn.Sigmoid(), layers[1], nn.Sigmoid(), layers[2])
    # A private CPU generator makes initialization independent of tile RNG use.
    gen = torch.Generator().manual_seed(seed)
    initial = []
    for layer, (i, o) in zip(layers, shapes):
        bound = 1 / math.sqrt(i)
        w = torch.empty(o, i).uniform_(-bound, bound, generator=gen)
        b = torch.empty(o).uniform_(-bound, bound, generator=gen)
        layer.set_weights(w, b)
        initial.append(dict(requested_weight=w.clone(), requested_bias=b.clone()))
    model = model.to(device)
    digest = hashlib.sha256()
    for layer, state in zip((model[0], model[2], model[4]), initial):
        w, b = layer.get_weights()
        state.update(weight=w.cpu().detach().clone(), bias=b.cpu().detach().clone())
        for key in ("weight", "bias"):
            digest.update(state[key].numpy().tobytes())
    tiles = list(model.analog_tiles())
    tile_shapes = [list(t.get_weights(apply_weight_scaling=False)[0].shape) for t in tiles]
    if tile_shapes != [[256, 392], [256, 392], [128, 256], [10, 128]]:
        raise RuntimeError(f"Unexpected physical tile mapping: {tile_shapes}")
    if any(t.get_output_gradient_mask_modes()[1] != cfg.forward.output_gradient_mask_mode
           for t in tiles):
        raise RuntimeError("Requested mask is not effective on every tile")
    if not all(hasattr(t.tile, "forward_with_output_mask") for t in tiles):
        raise RuntimeError("Rebuild this fork: forward_with_output_mask binding is missing")
    return model, initial, digest.hexdigest(), tile_shapes


def epoch_order(size, seed, epoch):
    import torch
    return torch.randperm(size, generator=torch.Generator().manual_seed(seed * 1000 + epoch))


class Diagnostics:
    """Observe existing tensors; never replay MVMs, mutate masks, or draw RNG.

    In non-native modes actual ADC clipping is unknown, not assumed to be zero.
    Per-context metadata also keeps successive forward calls independent.
    """
    def __init__(self, model):
        self.active = False
        self.epoch = self.batch = 0
        self.rows = []
        self.originals = []
        for index, tile in enumerate(model.analog_tiles()):
            forward, backward = tile.post_forward, tile.pre_backward
            self.originals.append((tile, forward, backward))

            def post(x_output, dim, is_test=False, ctx=None,
                     _tile=tile, _index=index, _forward=forward):
                result = _forward(x_output, dim, is_test, ctx)
                if self.active and not is_test and ctx is not None:
                    mask = ctx.saved_analog_tensors[1] if len(ctx.saved_analog_tensors) > 1 else None
                    bound = _tile.get_forward_out_bound()
                    legacy = x_output.detach().abs() >= bound
                    count = x_output.numel()
                    mode = _tile.get_output_gradient_mask_modes()[1]
                    row = dict(epoch=self.epoch, batch=self.batch, tile=_index,
                               layer=(0, 0, 1, 2)[_index], mode=mode, elements=count,
                               masked=int(mask.sum()) if mask is not None else 0,
                               mask_fraction=float(mask.float().mean()) if mask is not None else 0.,
                               legacy_threshold_fraction=float(legacy.float().mean()),
                               actual_clip_fraction=None, legacy_false_positive_fraction=None,
                               legacy_false_negative_fraction=None)
                    if mode == "native":
                        row.update(actual_clip_fraction=float(mask.float().mean()),
                                   legacy_false_positive_fraction=float((legacy & ~mask).float().mean()),
                                   legacy_false_negative_fraction=float((~legacy & mask).float().mean()))
                    ctx.constantstep_diagnostic = row
                return result

            def pre(d_input, dim, ctx=None, _backward=backward):
                result = _backward(d_input, dim, ctx)
                row = getattr(ctx, "constantstep_diagnostic", None)
                if row is not None:
                    row.update(error_l2_before=float(d_input.detach().norm()),
                               error_l2_after=float(result.detach().norm()),
                               error_zero_fraction_before=float((d_input == 0).float().mean()),
                               error_zero_fraction_after=float((result == 0).float().mean()))
                    self.rows.append(row)
                    del ctx.constantstep_diagnostic
                return result

            tile.post_forward, tile.pre_backward = post, pre

    def close(self):
        for tile, forward, backward in self.originals:
            tile.post_forward, tile.pre_backward = forward, backward


def weight_diagnostics(model, epoch):
    rows = []
    for i, tile in enumerate(model.analog_tiles()):
        w = tile.get_weights(apply_weight_scaling=False)[0]
        rows.append(dict(epoch=epoch, tile=i, layer=(0, 0, 1, 2)[i],
                         abs_mean=float(w.abs().mean()), abs_max=float(w.abs().max()),
                         boundary_fraction=float((w.abs() >= 2 - 1e-6).float().mean()),
                         within_one_pulse_fraction=float((w.abs() >= 2 - .002).float().mean())))
    return rows


def training_summary(history):
    import numpy as np
    acc = np.array([r["test_accuracy"] for r in history])
    tail = acc[-5:]
    return dict(final_test_accuracy=float(acc[-1]),
                best_test_accuracy=float(acc.max()),
                last5_mean=float(tail.mean()), last5_range=float(np.ptp(tail)),
                last5_sample_sd=float(tail.std(ddof=1)) if len(tail) > 1 else None,
                max_epoch_drop=float(max(0., -np.diff(acc).min())) if len(acc) > 1 else 0.,
                final_train_loss=history[-1]["train_loss"],
                final_online_train_accuracy=history[-1]["train_accuracy"],
                final_mask_fraction=history[-1]["mask_fraction"])


def is_complete(directory, spec):
    try:
        status = json.loads((directory / "status.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        with (directory / "history.csv").open() as stream:
            history = list(csv.DictReader(stream))
        return (status.get("state") == "completed" and
                status.get("fingerprint") == fingerprint(spec) == summary.get("fingerprint") and
                len(history) == spec["epochs"] and int(history[-1]["epoch"]) == spec["epochs"] and
                (directory / "final_model.pt").is_file())
    except (OSError, ValueError, KeyError, IndexError):
        return False


def aggregate(output, final=False):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest = json.loads((output / "campaign.json").read_text())
    curves, summaries = [], []
    done = 0
    for job in manifest["jobs"]:
        directory, spec = output / job["path"], job["spec"]
        complete = is_complete(directory, spec)
        done += complete
        try:
            with (directory / "history.csv").open() as stream:
                history = [{k: float(v) if v else float("nan") for k, v in r.items()}
                           for r in csv.DictReader(stream)]
            if history:
                curves.append((spec, history, complete))
                summaries.append(dict(mode=spec["mode"], pulse_noise=spec["pulse_noise"],
                                      seed=spec["seed"], complete=complete, epochs=len(history),
                                      **training_summary(history)))
        except OSError:
            pass
    write_json(output / "progress.json", dict(completed=done, total=len(manifest["jobs"])))
    write_csv(output / "summary.csv", summaries)
    for noise in dict.fromkeys(j["spec"]["pulse_noise"] for j in manifest["jobs"]):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        keys = ("train_loss", "train_accuracy", "test_accuracy", "mask_fraction")
        labels = ("Online train cross-entropy", "Online train accuracy (%)",
                  "Test accuracy (%)", "Applied mask (%) — sampled elements")
        for ax, key, label, scale in zip(axes.flat, keys, labels, (1, 100, 100, 100)):
            for mode in MODES:
                selected = [(h, c) for s, h, c in curves if s["mode"] == mode and s["pulse_noise"] == noise]
                complete_histories = [h for h, c in selected if c]
                for h, c in selected:
                    ax.plot([r["epoch"] for r in h], [r[key] * scale for r in h],
                            color=COLORS[mode], linestyle="-" if c else "--",
                            marker="o" if len(h) == 1 else None,
                            alpha=.3 if c and len(complete_histories) > 1 else .9,
                            label=f"{mode} ({'complete' if c else 'partial'}, epoch {len(h)})")
                if len(complete_histories) > 1:
                    values = np.array([[r[key] * scale for r in h] for h in complete_histories])
                    epochs = [r["epoch"] for r in complete_histories[0]]
                    mean, sd = values.mean(0), values.std(0, ddof=1)
                    ax.plot(epochs, mean, color=COLORS[mode], linewidth=2,
                            label=f"{mode} mean ± SD, n={len(values)}")
                    ax.fill_between(epochs, mean - sd, mean + sd, color=COLORS[mode], alpha=.15)
            ax.set(xlabel="Epoch", ylabel=label)
            ax.grid(alpha=.2)
            if ax.lines:
                handles, names = ax.get_legend_handles_labels()
                unique = dict(zip(names, handles))
                ax.legend(unique.values(), unique.keys(), fontsize=8)
        fig.suptitle(f"ConstantStep · pulse noise {noise:g} · LR 1.5 · batch 16\n"
                     f"{done}/{len(manifest['jobs'])} conditions complete; native = experimental mask")
        path = output / f"comparison_noise_{noise:g}.png"
        tmp = path.with_suffix(".tmp.png")
        fig.savefig(tmp, dpi=150)
        tmp.replace(path)
        if final:
            fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)


def provenance():
    import torch
    import aihwkit
    from aihwkit.simulator import rpu_base
    tracked = subprocess.check_output(["git", "ls-files", "src"], cwd=ROOT, text=True).splitlines()
    source_hash = hashlib.sha256()
    for name in tracked:
        source_hash.update(name.encode())
        source_hash.update((ROOT / name).read_bytes())
    return dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                working_tree=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                source_sha256=source_hash.hexdigest(),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                binding_sha256=hashlib.sha256(Path(rpu_base.__file__).read_bytes()).hexdigest(),
                python=sys.version, torch=torch.__version__, aihwkit=aihwkit.__version__,
                python_path=aihwkit.__file__, binding_path=rpu_base.__file__,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)


def run_worker(spec_path):
    import numpy as np
    import torch
    from torchvision.datasets import MNIST
    from aihwkit.optim import AnalogSGD
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    output = spec_path.parent
    campaign = Path(spec["campaign_dir"])
    write_json(output / "status.json", dict(state="running", fingerprint=fingerprint(spec)))
    try:
        torch.set_num_threads(spec["threads"])
        random.seed(spec["seed"])
        np.random.seed(spec["seed"])
        torch.manual_seed(spec["seed"])
        cfg = make_config(spec["mode"], spec["pulse_noise"], spec["seed"])
        model, initial, initial_hash, shapes = make_model(cfg, spec["seed"], spec["device"])
        # Fail before training if another mode has different programmed weights.
        for job in json.loads((campaign / "campaign.json").read_text())["jobs"]:
            if job["spec"]["seed"] == spec["seed"]:
                config_path = campaign / job["path"] / "configuration.json"
                if config_path.exists():
                    assert json.loads(config_path.read_text())["initial_weights_sha256"] == initial_hash
        torch.save(initial, output / "initial_weights.pt")
        write_json(output / "configuration.json", dict(
            spec=spec, rpu_config=encode(cfg), tile_shapes=shapes,
            effective_modes=[t.get_output_gradient_mask_modes() for t in model.analog_tiles()],
            initial_weights_sha256=initial_hash,
            seed_scope="Matched initialization, epoch permutations and construction seed; backend pulse RNG not guaranteed paired",
            data_transform="uint8 / 255, equivalent to ToTensor for MNIST; no normalization/augmentation",
            backward_bound_note="Bound management is a forward operation; backward retains backend behavior",
        ))
        # Cache MNIST tensors: identical values to ToTensor, no per-image Python overhead.
        arrays = []
        for train, limit in ((True, spec["train_limit"]), (False, spec["test_limit"])):
            data = MNIST(spec["data_dir"], train=train, download=False)
            x, y = data.data, data.targets
            if limit:
                x, y = x[:limit], y[:limit]
            arrays.append((x.to(spec["device"], dtype=torch.float32).flatten(1).div_(255),
                           y.to(spec["device"])))
        (x, y), (tx, ty) = arrays
        optimizer = AnalogSGD(model.parameters(), lr=1.5)
        optimizer.regroup_param_groups(model)
        diag = Diagnostics(model)
        history, weights, orders = [], [], []
        started = time.monotonic()
        for epoch in range(1, spec["epochs"] + 1):
            epoch_start = time.monotonic()
            model.train()
            order_cpu = epoch_order(len(x), spec["seed"], epoch)
            orders.append(hashlib.sha256(order_cpu.numpy().tobytes()).hexdigest())
            order = order_cpu.to(spec["device"])
            total_loss = torch.zeros((), device=spec["device"])
            correct = torch.zeros((), device=spec["device"], dtype=torch.long)
            for batch, offset in enumerate(range(0, len(x), 16), 1):
                ids = order[offset:offset + 16]
                diag.active = batch % spec["diagnostic_every"] == 0 or batch == 1
                diag.epoch, diag.batch = epoch, batch
                optimizer.zero_grad()
                pred = model(x[ids])
                loss = torch.nn.functional.cross_entropy(pred, y[ids])
                loss.backward()
                optimizer.step()
                total_loss += loss.detach() * len(ids)
                correct += (pred.detach().argmax(1) == y[ids]).sum()
            diag.active = False
            model.eval()
            test_loss = torch.zeros((), device=spec["device"])
            test_correct = torch.zeros((), device=spec["device"], dtype=torch.long)
            with torch.no_grad():
                for offset in range(0, len(tx), 256):
                    pred = model(tx[offset:offset + 256])
                    labels = ty[offset:offset + 256]
                    test_loss += torch.nn.functional.cross_entropy(pred, labels, reduction="sum")
                    test_correct += (pred.argmax(1) == labels).sum()
            sampled = [r for r in diag.rows if r["epoch"] == epoch]
            row = dict(epoch=epoch, lr=1.5, train_loss=float(total_loss) / len(x),
                       train_accuracy=float(correct) / len(x), test_loss=float(test_loss) / len(tx),
                       test_accuracy=float(test_correct) / len(tx),
                       mask_fraction=sum(r["masked"] for r in sampled) / sum(r["elements"] for r in sampled),
                       epoch_seconds=time.monotonic() - epoch_start,
                       elapsed_seconds=time.monotonic() - started)
            if not all(math.isfinite(v) for v in row.values()):
                raise RuntimeError(f"Non-finite metric at epoch {epoch}")
            history.append(row)
            weights.extend(weight_diagnostics(model, epoch))
            write_csv(output / "history.csv", history)
            write_csv(output / "diagnostics.csv", diag.rows)
            write_csv(output / "weights.csv", weights)
            write_json(output / "data_order_sha256.json", orders)
            remaining = row["elapsed_seconds"] / epoch * (spec["epochs"] - epoch)
            print(f"Epoch {epoch:02d}/{spec['epochs']} {spec['mode']} noise={spec['pulse_noise']:g} "
                  f"seed={spec['seed']} train_loss={row['train_loss']:.5f} "
                  f"online_train_acc={100*row['train_accuracy']:.2f}% "
                  f"test_acc={100*row['test_accuracy']:.2f}% mask={100*row['mask_fraction']:.2f}% "
                  f"epoch={row['epoch_seconds']:.1f}s elapsed={row['elapsed_seconds']:.0f}s ETA={remaining:.0f}s",
                  flush=True)
            aggregate(campaign)
        diag.close()
        torch.save(dict(model=model.state_dict(), spec=spec), output / "final_model.pt")
        write_json(output / "summary.json", dict(fingerprint=fingerprint(spec),
                   initial_weights_sha256=initial_hash, elapsed_seconds=time.monotonic() - started,
                   **training_summary(history)))
        write_json(output / "status.json", dict(state="completed", fingerprint=fingerprint(spec)))
    except BaseException as exc:
        write_json(output / "status.json", dict(state="failed", fingerprint=fingerprint(spec),
                                                error=f"{type(exc).__name__}: {exc}"))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--pulse-noise", nargs="+", type=float, default=[0.])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--diagnostic-every", type=int, default=250)
    parser.add_argument("--train-limit", type=int, default=0, help="Smoke only; 0 = full MNIST")
    parser.add_argument("--test-limit", type=int, default=0, help="Smoke only; 0 = full MNIST")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/DATASET")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results/constantstep_noise_seed0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_spec:
        run_worker(args.worker_spec)
        return
    if min(args.epochs, args.threads, args.diagnostic_every) < 1:
        parser.error("epochs, threads and diagnostic-every must be positive")
    if min(*args.seeds, args.train_limit, args.test_limit) < 0 or max(args.seeds) >= 2**31 - 2:
        parser.error("limits/seeds must be nonnegative; seeds must be < 2**31-2")
    if any(not math.isfinite(n) or n < 0 for n in args.pulse_noise):
        parser.error("pulse-noise must be finite and nonnegative")
    if args.train_limit > 60000 or args.test_limit > 10000:
        parser.error("MNIST limits cannot exceed 60000/10000")
    import torch
    from aihwkit.simulator import rpu_base
    if args.device == "cuda" and not (torch.cuda.is_available() and rpu_base.cuda.is_compiled()):
        parser.error("CUDA unavailable; no automatic CPU fallback")
    output = args.output_dir.resolve()
    env = provenance()
    identity = {k: env[k] for k in ("source_sha256", "runner_sha256", "binding_sha256", "torch", "python")}
    settings = dict(epochs=args.epochs, device=args.device, threads=args.threads,
                    diagnostic_every=args.diagnostic_every, train_limit=args.train_limit,
                    test_limit=args.test_limit, data_dir=str(args.data_dir.resolve()),
                    campaign_dir=str(output), identity=identity)
    jobs = []
    for noise in dict.fromkeys(args.pulse_noise):
        for seed in dict.fromkeys(args.seeds):
            for mode in dict.fromkeys(args.modes):
                spec = dict(settings, mode=mode, pulse_noise=noise, seed=seed)
                jobs.append(dict(path=f"noise_{noise:g}/seed_{seed}/{mode}", spec=spec))
    print(f"{len(jobs)} conditions × {args.epochs} epochs on {args.device}; LR=1.5 batch=16", flush=True)
    if args.dry_run:
        for job in jobs:
            print(job["path"])
        return
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / ".campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "campaign.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["jobs"] != jobs:
                raise RuntimeError("Settings, runner, binding or selection changed: choose a new output directory")
        else:
            write_json(manifest_path, dict(jobs=jobs, environment=env))
            (output / "executed_runner.py").write_bytes(Path(__file__).read_bytes())
        for job in jobs:
            directory, spec = output / job["path"], job["spec"]
            if is_complete(directory, spec):
                print(f"SKIP completed {job['path']}", flush=True)
                continue
            # Preserve partial evidence, but never treat it as completed training.
            if directory.exists():
                archive = output / "interrupted" / (job["path"].replace("/", "_") + f"_{time.time_ns()}")
                archive.parent.mkdir(exist_ok=True)
                directory.rename(archive)
            write_json(directory / "spec.json", spec)
            print(f"RUN {job['path']}", flush=True)
            with (directory / "run.log").open("w") as log:
                process = subprocess.Popen([sys.executable, "-u", __file__, "--worker-spec", str(directory / "spec.json")],
                                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    if process.wait() != 0:
                        raise RuntimeError(f"Failed condition: {job['path']}; see {directory / 'run.log'}")
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        process.wait()
                    raise
            aggregate(output)
        aggregate(output, final=True)
        print(f"COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
