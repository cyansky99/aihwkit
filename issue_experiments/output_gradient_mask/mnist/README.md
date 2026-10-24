# ConstantStep MNIST: output-gradient mask comparison

Compare `legacy`, `native`, and `disabled` while keeping the learning settings
fixed. Here `native` is this fork's experimental clipping-mask implementation.
The network is a dense 784–256–128–10 classifier with sigmoid hidden layers,
using standalone ConstantStepDevice, LR 1.5 and batch 16.

The default campaign runs **five separate trials per mode at nominal seed 0**:
15 runs, each with 20 epochs of full MNIST. Repetitions share initialization and
data order; they do not assume paired backend pulse randomness. The completed
[one-trial pilot](sample_results/pilot_seed0/README.md) is separate from this
campaign and is not counted toward its five trials.

## Run

Build this fork once using the [CPU/CUDA build instructions](../README.md#2-this-fork-build-once-then-compare-three-policies).
An existing matching extension can be reused. Training requires Torch,
Torchvision, NumPy and Matplotlib; plotting saved results needs only Matplotlib.

From the repository root, in the environment containing that build:

```bash
conda activate aihwkit_upstream_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH=src:.

# List the 15 runs without loading MNIST, importing Torch or using the GPU.
python issue_experiments/output_gradient_mask/mnist/run_constantstep_noise.py --dry-run

# Five trials per mode, all at the same nominal seed.
python -u issue_experiments/output_gradient_mask/mnist/run_constantstep_noise.py \
  --device cuda --seeds 0 --trials 5 --pulse-noise 0 --epochs 20
```

The dataset must already exist under `data/DATASET` (or `--data-dir`). To download
it explicitly when needed, run this once before training:

```bash
python -c 'from torchvision.datasets import MNIST; MNIST("data/DATASET", train=True, download=True); MNIST("data/DATASET", train=False, download=True)'
```

CUDA requests fail if the GPU or CUDA backend is unavailable; there is no CPU
fallback. Use `--device cpu` explicitly for a CPU campaign. To change repetitions,
use `--trials N`. Additional nominal seeds or pulse-size noises can be selected
with `--seeds 0 1` or `--pulse-noise 0 0.02 0.05`; these are not run by default.
Use a new `--output-dir` when changing settings or code.

## Results and resume

Default output: `issue_experiments/output_gradient_mask/mnist/results/constantstep_mask_comparison/`
from the repository root.

Repeat the same command to skip verified completed runs. Interrupted runs are
archived and restart at epoch 1. There is no mid-epoch or checkpoint resume.
Settings and execution-code/binding fingerprints prevent mixing different
campaigns. Each trial runs in a fresh process; trials execute sequentially,
with legacy, native, disabled in that order within each trial.

Python regenerates a four-panel PNG after each saved epoch; no watcher process
is needed. When all runs finish, it also writes a PDF. The four panels show test
accuracy, online training loss, online training accuracy and sampled mask rate.

**Thick lines show the mean of completed trials; shaded regions show ±1 sample
standard deviation (not a confidence interval).** Thin lines show individual
trials. Partial runs use dashed lines and are excluded from the mean and SD.
The legend reports completed/planned trial counts; no band is drawn for n=1.
Noise/seed combinations have separate figures and are never pooled.

`summary.csv` contains individual trial metrics. `trial_summary.csv` contains
final test-accuracy means and sample SDs across completed trials. Accuracy values
in CSV are fractions; plots show percentages. Last-five-epoch fluctuations in
`summary.csv` describe a trajectory, not between-trial variability.

## Redraw without training

This reads saved CSV/JSON only and writes PNG/PDF. It does not require Torch,
AIHWKit, model checkpoints or a GPU. Existing figure files at the destination
are replaced; measurement files are not changed.

```bash
python issue_experiments/output_gradient_mask/mnist/plot_training_comparison.py \
  --input issue_experiments/output_gradient_mask/mnist/results/constantstep_mask_comparison

# Redraw the published pilot into a separate directory.
python issue_experiments/output_gradient_mask/mnist/plot_training_comparison.py \
  --input issue_experiments/output_gradient_mask/mnist/sample_results/pilot_seed0 \
  --output-dir /tmp/constantstep_pilot_figures
```

## Measured pilot

The existing full-MNIST CUDA pilot has one 20-epoch trial per mode at seed 0 and
pulse-size noise 0. Final test accuracy was **97.53% legacy, 98.27% native and
98.21% disabled**. These are individual runs, not five-trial means. Runtime pulse
randomness was not paired, so repeated trials are needed to quantify the
accuracy difference and its variability.

![One-trial learning comparison](sample_results/pilot_seed0/training_comparison_noise_0_seed_0.png)

See the [pilot report and recorded provenance](sample_results/pilot_seed0/README.md),
[full protocol](CONSTANTSTEP_NOISE.md), and [implementation/validation record](IMPLEMENTATION_LOG.md).
The published snapshot contains measurements and figures, not datasets or model
checkpoints. The issue's small numerical reproducer remains a separate experiment.
