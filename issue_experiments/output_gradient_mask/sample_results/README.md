# Measured examples — experimental build

These are fresh executions, not bars drawn from expected values. Both the
existing-API mask-toggle runs and the three-mode runs use the same experimental
binding. In particular, `upstream_*.json` describes an **existing-API experiment**,
not a pristine upstream binary. Its mask-on cases exercise retained legacy behavior.

| Experiment | CPU | CUDA |
| --- | --- | --- |
| Existing API: mask on/off | [JSON](upstream_cpu.json), [CSV](upstream_cpu.csv), [summary](upstream_cpu.summary.txt), [PNG](upstream_cpu.png), [PDF](upstream_cpu.pdf) | [JSON](upstream_cuda.json), [CSV](upstream_cuda.csv), [summary](upstream_cuda.summary.txt), [PNG](upstream_cuda.png), [PDF](upstream_cuda.pdf) |
| Three mask policies | [JSON](modes_cpu.json), [CSV](modes_cpu.csv), [summary](modes_cpu.summary.txt), [PNG](modes_cpu.png), [PDF](modes_cpu.pdf) | [JSON](modes_cuda.json), [CSV](modes_cuda.csv), [summary](modes_cuda.summary.txt), [PNG](modes_cuda.png), [PDF](modes_cuda.pdf) |

All use W=[1,1], x=[1,1], ADC bound 1, no noise/quantization, perfect backward,
and no optimizer update. They include 6 existing-API cases or 9 three-mode cases
per backend. Both gradient components and finite differences are retained.

The implementation commit is `57af3ed`; tools/docs were still being prepared when
these results were captured. JSON records that dirty worktree honestly and hashes
the executed scripts and native binding. Uncommitted local MNIST material appears
in status metadata but was not imported or run and is not part of these commits.
The implementation sources and binary matched the earlier validated build
fingerprints, so no rebuild was necessary for the Python-only plotting additions.

CPU and actual CUDA execution used Python 3.12.12, Torch 2.9.1+cu126 and an RTX
4080 SUPER for CUDA. Architecture 89 was used by the existing local build. This
does not establish A100 execution. Original-source Python compatibility was also
checked in a temporary tree, using the same experimental binary; that is not a
fresh pristine build either.

Commands from the repository root (choose a fresh output location to repeat):

```bash
PYTHONPATH=src python issue_experiments/output_gradient_mask/reproduce_upstream.py \
  --device cpu --output results/upstream_cpu.json --plot
PYTHONPATH=src python issue_experiments/output_gradient_mask/reproduce_upstream.py \
  --device cuda --output results/upstream_cuda.json --plot
PYTHONPATH=src python issue_experiments/output_gradient_mask/compare_modes.py \
  --device cpu --output results/modes_cpu.json --plot
PYTHONPATH=src python issue_experiments/output_gradient_mask/compare_modes.py \
  --device cuda --output results/modes_cuda.json --plot
```

Validation logs: [related tile/mask regression](related_regression.txt) and
[public tools tests](tools_tests.txt). The related suite passed 788 tests with
87 skips; the public tools tests passed 17 tests. GPU numerical tests executed;
the reported skips in the related suite are not a blanket CUDA skip.

The examples establish the isolated gradient inconsistency and control behavior.
They do not establish a training accuracy benefit or a universal exact derivative
for noisy, quantized or dynamically managed analog operations.
