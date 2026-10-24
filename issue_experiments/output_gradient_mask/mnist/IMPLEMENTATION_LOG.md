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

## Default duration update

Changed the new campaign default from 20 to 30 epochs per trial at user request
(15 runs, 450 epochs total). Updated example commands and default-value tests.
The existing 20-epoch pilot and its recorded execution source remain unchanged.
The explicit --epochs override remains available. No training was launched.

## Forward-only management control

Added separate `run_forward_management.py` and `plot_forward_management.py`
entrypoints while the 10-trial baseline campaign was running. Neither baseline
Python file nor its compiled backend was modified. The new condition disables
forward NM and BM only; backward NM remains ABS_MAX and the backend's unsupported
backward BM behavior is unchanged. Other configuration fields are identical.
The default new condition is `none`; `managed none` selects the six combinations
with the three mask policies. Grouped plots and final-accuracy statistics do not
pool the two forward conditions. The default is five trials, 30 epochs per run.

Validation: `tests/test_forward_management.py`: 8 passed, no MNIST training or
GPU allocation. Tests verify the exact two-field configuration difference for
all modes, worker configuration selection before model/data creation, job counts,
separate statistics/figures, identity checks, four archived source files and a
dry-run without modifying the baseline files. Baseline source hashes were
checked before and after this addition. New training, commit and push were not
performed for this step.

Scope correction: the additional condition is legacy-only with forward NM/BM NONE. The existing three managed mask conditions remain unchanged. The separate runner now rejects native/disabled and managed, defaults to five legacy trials, and schedules ten runs for --trials 10. README and tests were corrected. No new training, commit or push.

## 2026-10-24 — fourth comparison condition and borderless bands

- Added `workaround` to the common runner/plotter. It uses legacy masking and
  disables only forward NM/BM; backward NM remains ABS_MAX.
- Default selection is four conditions × five trials × 30 epochs. Configuration
  and diagnostic records distinguish condition labels from actual mask policies.
- Removed the sample-SD band's edge stroke; retained individual-trial curves.
- Replaced documentation for the earlier separate forward-management entrypoint
  with unified four-condition commands. The superseded local prototype is archived
  privately. Historical measurement files and archived runners remain unchanged.
- A private one-off adapter under ignored `temp/` combines the completed
  10-trial/30-epoch baseline with ten new workaround trials. It validates matching
  simulator/binding, configuration, initialization and data order. It is not a
  general relaxation of campaign resume checks and is not published.
- Validation and launch details are recorded in the private comparison README.


## 2026-10-24 — accuracy detail panels and publication checks

- Changed the shared plotter from 2×2 to two rows × three columns, retaining
  the original four views and inserting narrower test/train accuracy detail
  panels for epochs 20–30. Test limits: 97.8–98.6%; train: 99.75–100.02%.
- Detail views share the exact trial data and completed-trial statistics with
  the overview; axes are independent. SD bands remain borderless. Before
  epoch 20, a label explains the empty detail windows.
- The campaign's existing per-epoch aggregate call uses this plotter directly.
  No watcher, new training, dataset download or backend rebuild was used.
- Existing completed evidence was redrawn and the generated PNG inspected.
- `tests/test_constantstep_mask_training.py` and
  `tests/test_constantstep_trials.py`: **20 passed**, CPU and actual CUDA,
  using the existing aihwkit_upstream_env. The plot test checks all six axes,
  exact limits, matching overview/detail values and absent SD outlines.


## 2026-10-24 — completed ten-trial evidence export

Published all 40 completed 30-epoch runs (first three conditions measured
first, workaround later), without retraining or selecting trials. Copied 326
raw CSV/JSON and executed-source artifacts byte-for-byte. Checked completion,
run fingerprints, configuration differences, four tiles, actual mask policies,
common simulator/binding hashes, equal programmed weights and all epoch-order
hashes. Archived model/data-order helpers and the training loop are structurally
identical across stages. The public plotting manifest uses relative paths and
unchanged original specs; provenance retains the two original manifests.

The private one-off adapter remains untracked; its hash and role are disclosed.
Public code runs all four conditions directly. Published records omit datasets,
checkpoints, binaries, interrupted runs and temporary orchestration code.
The six-panel figure was regenerated from the copied records. Final accuracy
means/SDs were recomputed from epoch 30, rather than best-epoch selection.
