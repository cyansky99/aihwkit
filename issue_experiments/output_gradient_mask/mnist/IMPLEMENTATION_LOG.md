# ConstantStep MNIST implementation and validation record

## Pilot execution before publication

The existing pilot completed full MNIST on an RTX 4080 SUPER: seed 0,
dw_min_std=0, 20 epochs each for legacy/native/disabled. Final test accuracies
were 97.53%, 98.27% and 98.21%. Initial weight/bias hashes and all epoch-order
hashes matched. The source and compiled binding matched the validated mask
implementation; no rebuild was needed.

The historical pipeline checks included a small MNIST smoke run, completed-run
skipping and an interrupted-run restart. Earlier example campaigns and the
interrupted plotting-fix attempt remain locally archived. The published pilot
contains the subsequently completed three runs, not those earlier attempts.

## Five-trial runner

Added explicit trial IDs, with five separate processes per mode at nominal seed
0 by default. Trial IDs do not modify initialization, construction or data seeds.
Learning settings and diagnostic tensors are unchanged. Completed runs are
verified before skipping; interrupted runs are preserved and restart at epoch 1.
The new default output directory does not reuse the pilot.

The runner calls `plot_training_comparison.py` after saving each epoch. It plots
test accuracy, online train loss/accuracy and sampled mask rate, using completed
trials for mean ± sample SD and showing partial trials separately. Noise/seed
groups remain separate. CSV/JSON-only redraws require neither checkpoints nor
the simulator. Final-accuracy trial summaries are separate from within-trajectory
fluctuation descriptors.

## Validation during publication preparation

No new MNIST training or dataset downloads were performed. No simulator source,
compiled binding or issue text was changed.

Command, with the existing `aihwkit_upstream_env` and `PYTHONPATH=src:.`:

```bash
python -m pytest tests/test_constantstep_mask_training.py \
  tests/test_constantstep_trials.py tests/test_output_gradient_mask.py \
  -q -p no:cacheprovider --disable-warnings
```

Result: **95 passed, 1 skipped**, with actual CPU and CUDA execution. The skipped
case is the CPU parameterization of a CUDA-stream-only test. Existing warnings
were emitted by dependencies; no test failed.

Checks cover configuration equivalence, actual tile mapping and initial weights,
diagnostic transparency, 15 default jobs, trial ordering, preserved seeds,
complete/interrupted recovery, incompatible identity rejection, known mean and
sample-SD values, partial-trial exclusion, independent noise/seed groups, missing
and non-finite observations, and plotting/dry-run with backend imports forbidden.
The pilot's final measurements, initial hashes, data orders and executed-runner
hash were checked against its original records.

The real pilot figure was regenerated from CSV/JSON and visually inspected.
Synthetic five-trial figures exercised SD bands and legends only; they are test
fixtures, not published training evidence. Every public copied artifact has an
original/published hash record; CSV changes are line-ending normalization only.

## Publication boundary

The implementation and documentation form one commit; the measured one-trial
pilot forms a separate commit on the same branch. Models, datasets, native
binaries, earlier incomplete campaigns and unrelated local handoff files are
excluded. Publishing this material does not run the five-trial campaign or edit
the GitHub issue.
