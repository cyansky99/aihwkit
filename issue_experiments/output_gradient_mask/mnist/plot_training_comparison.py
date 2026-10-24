#!/usr/bin/env python3
"""Plot saved ConstantStep trials with mean ± sample SD; no Torch/AIHWKit imports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

MODES = ("legacy", "native", "disabled", "workaround")
COLORS = {"legacy": "#dc8b27", "native": "#287bb5", "disabled": "#369552", "workaround": "#9456b3"}
METRICS = ("test_accuracy", "train_loss", "train_accuracy", "mask_fraction")
LABELS = ("Test accuracy (%)", "Online training cross-entropy",
          "Online training accuracy (%)", "Applied mask (%) · sampled elements")


def read_job(root, job):
    """Validate saved observations; absent or partial runs never become zeros."""
    spec = job["spec"]
    directory = Path(root) / job["path"]
    rows = []
    try:
        with (directory / "history.csv").open(newline="") as stream:
            for raw in csv.DictReader(stream):
                row = {key: float(raw[key]) for key in ("epoch", *METRICS)}
                if not all(math.isfinite(value) for value in row.values()):
                    raise ValueError(f"Non-finite observation in {directory}")
                if row["epoch"] != len(rows) + 1 or row["epoch"] > spec["epochs"]:
                    raise ValueError(f"Nonconsecutive or unexpected epoch in {directory}")
                if "trial_id" in raw and int(raw["trial_id"]) != spec["trial_id"]:
                    raise ValueError(f"Trial identity mismatch in {directory}")
                rows.append(row)
    except FileNotFoundError:
        pass
    try:
        status = json.loads((directory / "status.json").read_text())
    except FileNotFoundError:
        status = {}
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if status.get("fingerprint", digest) != digest:
        raise ValueError(f"Status/configuration identity mismatch in {directory}")
    state = status.get("state", "not started")
    complete = state == "completed" and len(rows) == spec["epochs"]
    # Checkpoints are deliberately not required: published evidence excludes them.
    return dict(spec=spec, rows=rows, complete=complete, state=state,
                trial_id=spec.get("trial_id", 1))


def load_jobs(root):
    manifest = json.loads((Path(root) / "campaign.json").read_text())
    jobs = [read_job(root, job) for job in manifest["jobs"]]
    if not jobs:
        raise ValueError("campaign.json contains no jobs")
    identities = [(j["spec"]["pulse_noise"], j["spec"]["seed"],
                   j["spec"]["mode"], j["trial_id"]) for j in jobs]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate trial identity in campaign")
    return jobs


def curve_statistics(jobs, key):
    histories = [j["rows"] for j in jobs if j["complete"]]
    if not histories:
        return [], [], None
    epochs = [row["epoch"] for row in histories[0]]
    if any([row["epoch"] for row in h] != epochs for h in histories):
        raise ValueError("Completed trials have different epoch grids")
    samples = [[h[i][key] for h in histories] for i in range(len(epochs))]
    mean = [statistics.mean(values) for values in samples]
    sd = [statistics.stdev(values) for values in samples] if len(histories) > 1 else None
    return epochs, mean, sd


def trial_statistics(jobs):
    """Final-epoch variation across completed trials, never across epochs."""
    rows = []
    groups = sorted({(j["spec"]["pulse_noise"], j["spec"]["seed"]) for j in jobs})
    for noise, seed in groups:
        for mode in MODES:
            selected = [j for j in jobs if (j["spec"]["pulse_noise"], j["spec"]["seed"],
                                            j["spec"]["mode"]) == (noise, seed, mode)]
            if not selected:
                continue
            values = [j["rows"][-1]["test_accuracy"] for j in selected if j["complete"]]
            rows.append(dict(pulse_noise=noise, seed=seed, mode=mode,
                             completed_trials=len(values), planned_trials=len(selected),
                             final_test_accuracy_mean=statistics.mean(values) if values else None,
                             final_test_accuracy_sample_sd=statistics.stdev(values) if len(values) > 1 else None))
    return rows


def save_figure(fig, path):
    """Readers see either the previous complete figure or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        fig.savefig(temporary, dpi=160, facecolor="white", format=path.suffix[1:])
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_figure(jobs, noise, seed):
    """A fixed cohort of completed trials defines each mean and SD band."""
    selected = [j for j in jobs if (j["spec"]["pulse_noise"], j["spec"]["seed"]) == (noise, seed)]
    if not selected:
        raise ValueError("No trials for requested noise/seed")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False}):
        # Independent x axes keep the detail windows from cropping the overview.
        fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                                 gridspec_kw={"width_ratios": [1.15, .85, 1.15]})
        panels = (
            (axes[0, 0], "test_accuracy", 100, LABELS[0], None),
            (axes[0, 1], "test_accuracy", 100, LABELS[0], (97.8, 98.6)),
            (axes[0, 2], "train_loss", 1, LABELS[1], None),
            (axes[1, 0], "train_accuracy", 100, LABELS[2], None),
            (axes[1, 1], "train_accuracy", 100, LABELS[2], (99.75, 100.02)),
            (axes[1, 2], "mask_fraction", 100, LABELS[3], None),
        )
        fig.subplots_adjust(left=.065, right=.98, bottom=.17, top=.77,
                            wspace=.34, hspace=.30)
        fig.text(.08, .957, "ConstantStep MNIST: output-gradient mask comparison", fontsize=18, weight="bold")
        done = sum(j["complete"] for j in selected)
        fig.text(.08, .92, f"Pulse noise {noise:g}  ·  nominal seed {seed}  ·  LR 1.5  ·  batch 16"
                 f"  ·  {done}/{len(selected)} runs complete", color="#4C5663")
        handles = []
        for mode in MODES:
            trials = [j for j in selected if j["spec"]["mode"] == mode]
            if not trials:
                continue
            n = sum(j["complete"] for j in trials)
            color = COLORS[mode]
            handles.append(Line2D([], [], color=color, lw=2.5, label=f"{mode} · n={n}/{len(trials)}"))
            for ax, key, scale, _, _ in panels:
                for trial in trials:
                    h = trial["rows"]
                    if h:
                        ax.plot([r["epoch"] for r in h], [r[key] * scale for r in h],
                                color=color, linewidth=1, alpha=.3 if trial["complete"] else .75,
                                linestyle="-" if trial["complete"] else "--",
                                marker="o" if len(h) == 1 else None, markersize=4)
                epochs, mean, sd = curve_statistics(trials, key)
                if epochs:
                    ax.plot(epochs, [v * scale for v in mean], color=color, linewidth=2.4,
                            marker="o" if len(epochs) == 1 else None, markersize=4)
                if sd is not None:
                    ax.fill_between(epochs, [(m - s) * scale for m, s in zip(mean, sd)],
                                    [(m + s) * scale for m, s in zip(mean, sd)], color=color, alpha=.16,
                                    edgecolor="none", linewidth=0)
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.07, .895),
                   frameon=False, ncol=min(4, len(handles)), fontsize=11)
        partial = [f"{j['spec']['mode']} trial {j['trial_id']}: {len(j['rows'])}/{j['spec']['epochs']} epochs"
                   f" ({j['state']})" for j in selected if not j["complete"] and j["rows"]]
        progress = "All runs complete" if done == len(selected) else (
            "; ".join(partial[:2]) + (f"; +{len(partial)-2} partial runs" if len(partial) > 2 else "")
            if partial else "Waiting for the first saved epoch")
        fig.text(.08, .815, progress, fontsize=10, color="#4C5663")
        target = max(j["spec"]["epochs"] for j in selected)
        for ax, key, _, label, zoom in panels:
            ax.set(xlabel="Epoch", ylabel=label,
                   xlim=(20, 30) if zoom else (.5, target + .5))
            if zoom:
                ax.set_ylim(*zoom)
                ax.set_title("Detail · epochs 20–30", fontsize=11)
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                if not any(20 <= r["epoch"] <= 30 for j in selected for r in j["rows"]):
                    ax.text(.5, .5, "No saved epochs 20–30 yet", ha="center",
                            fontsize=9, transform=ax.transAxes)
            ax.tick_params(labelbottom=True)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.grid(alpha=.18)
            if not ax.lines and not zoom:
                ax.text(.5, .5, "No saved epochs yet", ha="center", transform=ax.transAxes)
        fig.text(.08, .105, "Center panels magnify fixed accuracy ranges at epochs 20–30; values outside these windows are not shown there.", fontsize=9.5, color="#4C5663")
        fig.text(.08, .08, "Thick lines: completed-trial mean. Shading: ±1 sample SD (n ≥ 2), not a confidence interval.", fontsize=10)
        fig.text(.08, .052, "Thin lines: individual trials. Dashed: partial runs, excluded from mean/SD. Same seed does not pair backend RNG.", fontsize=9.5)
        fig.text(.08, .024, "native: experimental mask. workaround: legacy mask, forward NM/BM off. Training metrics online; mask rates sampled.", fontsize=9.5, color="#4C5663")
    return fig


def render(root, output_dir=None, pdf=False):
    root = Path(root)
    output_dir = Path(output_dir) if output_dir else root
    jobs = load_jobs(root)
    paths = []
    for noise, seed in sorted({(j["spec"]["pulse_noise"], j["spec"]["seed"]) for j in jobs}):
        fig = make_figure(jobs, noise, seed)
        try:
            prefix = output_dir / f"training_comparison_noise_{noise:g}_seed_{seed}"
            for suffix in ([".png", ".pdf"] if pdf else [".png"]):
                path = Path(str(prefix) + suffix)
                save_figure(fig, path)
                paths.append(path)
        finally:
            plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing campaign.json")
    parser.add_argument("--output-dir", type=Path, help="Default: input directory; figures are replaced")
    args = parser.parse_args()
    for path in render(args.input, args.output_dir, pdf=True):
        print(path)


if __name__ == "__main__":
    main()
