# ConstantStep MNIST protocol

This reconstructs the low-noise settings that motivated the training comparison.
The dense sigmoid architecture and AnalogSGD workflow follow upstream's
[MNIST example](../../../examples/03_mnist_training.py). This experiment uses
raw logits with mean CrossEntropy, constant LR 1.5 and batch 16, rather than the
example's LogSoftmax/NLLLoss and LR schedule. It is separate from the earlier
local ConstantStep/SoftBounds/TransferCompound campaign.

## Fixed settings

| Setting | Value |
| --- | --- |
| Model | 784–256–128–10; sigmoid hidden layers, digital bias, raw logits |
| Data | Full MNIST 60,000 train / 10,000 test; uint8/255, equivalent to ToTensor; no augmentation |
| Optimizer / loss | AnalogSGD, mean CrossEntropy, constant LR 1.5, batch 16 |
| Device | Standalone ConstantStepDevice; dw_min=0.002, weight bounds ±2 |
| Device variation | D-to-D variation, asymmetry and reset noise zero; only dw_min_std varies |
| Default pulse-size noise | dw_min_std=0; stochastic pulse generation remains enabled |
| Forward / backward | Non-perfect ONE_PASS; input bound 1, output bound 12; resolutions 1/4094; output noise 0 |
| Management | ABS_MAX noise management and ITERATIVE forward BM; workaround sets only forward NM/BM to NONE; update and bit-length management enabled |
| Mapping | Maximum input/output size 512; weight scaling disabled; first layer split 392+392 |
| Physical tiles | 256×392, 256×392, 128×256, 10×128 |
| Update | STOCHASTIC_COMPRESSED, desired_bl=31, fixed_bl=True |
| New campaign | Seed 0; five trials per condition (four conditions); 30 epochs per trial; CUDA by default |

Backward retains the backend's behavior: bound management is a forward
operation. There is no DCSR, refresh, compound device or simulator modification
in this runner. The `native` mode changes input-gradient masking; it does not
change the optimizer's local error-storage rule. `disabled` is a diagnostic
control and also allows gradients through true clipping.

`trial_id` identifies separate process executions without changing seeds.
A private CPU generator programs the same initial weights/biases, and
`seed * 1000 + epoch` determines each epoch's data permutation. Programmed-weight
hashes, permutation hashes and device construction seeds are recorded. Initial
weight hashes are checked against earlier runs at the same seed. Backend runtime
pulse RNG is not guaranteed paired, even when pulse-size noise is zero.

## Saved records

Each `noise_<value>/seed_<n>/trial_<id>/<mode>/` stores:

- `spec.json`, `configuration.json`, `status.json`, and `run.log`.
- `history.csv`: trial ID, epoch, train/test loss and accuracy, sampled mask rate and timing.
- `diagnostics.csv`: trial ID, tile/layer, mask counts and error norms before/after masking.
- `weights.csv`: trial ID and physical weight-boundary occupancy per epoch and tile.
- `data_order_sha256.json`, `initial_weights.pt`, `final_model.pt`, `summary.json`.

The campaign records the source commit/worktree state, simulator source hash,
runner/plotter hashes, compiled binding hash and path, Python/Torch/package
versions and GPU. Copies of the executed runner and plotter are retained.
Changes to code, binding, environment identity or selected conditions require a
new output directory. A lock prevents simultaneous launchers for one campaign.
Completed-run validation requires the full epoch sequence, matching identity and
final model. The figure reader deliberately does not require model files, so
published results remain independently plottable.

Diagnostics observe batch 1 and every 250th batch without additional forward
calls or random-number generation. In native mode, actual clipping and the old
threshold are compared on the same forward tensors. Actual clipping is unknown
for legacy/disabled and stays empty in those fields, rather than being filled
with zero. Physical weight occupancy is measured without noisy readout.

## Interpretation

Training accuracy and loss are accumulated while the model is updating. Test
metrics use the complete test set after each epoch. Best test accuracy is a
trajectory descriptor, not a validation-based selection criterion.

The primary variability measure is final test accuracy **across trials** at the
same nominal seed and pulse noise. The figure uses mean ± sample SD (`ddof=1`)
over completed trials at each epoch. A single trial has no estimated SD. The
same completed cohort is used throughout each plotted mean curve; partial
trials are shown individually. This avoids silently mixing sample counts across
epochs. Missing measurements are not interpolated or filled.

Last-five-epoch range/SD and maximum consecutive-epoch accuracy drop remain
secondary descriptors. They do not measure trial-to-trial variability. The
published pilot is n=1 per mode and is not used to initialize, fill or select
conditions for the new five-trial campaign.

## Verification

From the repository root, using the matching fork build:

```bash
PYTHONPATH=src:. python -m pytest \
  tests/test_constantstep_mask_training.py tests/test_constantstep_trials.py -q
```

The configuration and diagnostics tests use small tensors; they do not load or
train on MNIST. The diagnostic transparency check uses deterministic updates to
isolate observer effects from unpaired pulse randomness. Campaign/statistics
checks use synthetic records, never published as measured training evidence.
See [README.md](README.md) for training and redraw commands.

## Workaround condition

The common runner accepts `--modes workaround` alongside legacy/native/disabled.
Its actual simulator mask policy is legacy; only forward NM and BM are disabled.
The backward configuration is identical to legacy. Configuration JSON separates
`condition` from `mask_mode`; diagnostics label the condition and actual mask
policy independently. `workaround` is never passed as an IO mask-policy value.

The published one-trial pilot predates this addition and still contains only the
original three conditions. Its measurements and archived execution code remain
unchanged. Historical three-condition campaigns remain readable by the plotter.

The mean ± sample SD region has no stroke or edge color. Thin individual-trial
curves remain visible, and all four condition names appear in the legend when
present. The general resume checks still reject changed campaign identities.


## Figure layout

The two-row, three-column figure includes full accuracy curves on the left,
accuracy detail views in the center, and online loss/mask rate on the right.
Detail views cover epochs 20–30 and fixed ranges of 97.8–98.6% test accuracy and
99.75–100.02% online training accuracy. These limits affect display only, not
the completed-trial cohort, mean or sample SD. The campaign calls the same
plotter immediately after every epoch's CSV save; redraw needs no training.
