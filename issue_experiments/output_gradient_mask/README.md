# Output-gradient-mask reproduction

This commit adds a standalone reproducer and explanation on top of upstream
`7f22aa7ff0054609fa120d1c2696414460f13f91` (source version 1.2.0).
It does not change AIHWKit's simulator or include a training benchmark.

- [Standalone reproducer](reproduce_upstream.py)
- [Issue draft](ISSUE_DRAFT.md)

## Run

Copy the script into an environment with an unmodified AIHWKit installation:

```bash
python reproduce_upstream.py --device cpu --output upstream_cpu.json
python reproduce_upstream.py --device cuda --output upstream_cuda.json
```

It uses existing public APIs, does not modify the installation, and needs no
dataset, optimizer step, new mask-mode field or new native binding. Check its
printed package/binding paths to verify the installation used. Existing output
files are not overwritten. JSON includes the environment, forward outputs,
input gradients, expected values and central finite differences.

The six cases use W=[1,1], x=[1,1], no bias, ADC bound 1, DAC bound 2,
noise/quantization off, no noise management, perfect backward and loss=sum(y).

| Forward case | Mask enabled | Observed output | Observed dL/dx | Expected dL/dx |
| --- | --- | --- | --- | --- |
| Iterative bound management | Yes | 2 | [0,0] | [1,1] |
| Iterative bound management | No | 2 | [1,1] | [1,1] |
| Perfect forward | Yes | 2 | [0,0] | [1,1] |
| Perfect forward | No | 2 | [1,1] | [1,1] |
| Actual clipping, management off | Yes | 1 | [0,0] | [0,0] |
| Actual clipping, management off | No | 1 | [1,1] | [0,0] |

These are locally observed outcomes, not assertions that every installed release
must produce them. The script prints observations without asserting the bug
must exist. Mask-off cases change only `tile.handle_output_bound` on temporary
layer instances and demonstrate why unconditional mask removal is not a fix.

## Suspected mechanism

The original Python wrapper compares the returned output with
`0.999 * out_scale * out_bound`. Native bound management clips in ADC coordinates
and then restores the scale before returning. A restored value exceeding that
fixed threshold need not have been clipped. Perfect forward bypasses ADC clipping
but the Python mask is still applied.

Relevant baseline source:

- [Threshold and enablement](https://github.com/IBM/aihwkit/blob/7f22aa7ff0054609fa120d1c2696414460f13f91/src/aihwkit/simulator/tiles/rpucuda.py#L119)
- [Mask construction](https://github.com/IBM/aihwkit/blob/7f22aa7ff0054609fa120d1c2696414460f13f91/src/aihwkit/simulator/tiles/periphery.py#L871)
- [Native CUDA clipping and restoration](https://github.com/IBM/aihwkit/blob/7f22aa7ff0054609fa120d1c2696414460f13f91/src/rpucuda/cuda/io_manager.cu#L515)

## Evidence limits

Local observations include earlier CPU/CUDA records and independent CPU checks
on original Python with an experimental native binary. The earlier records do
not retain a contemporaneous original-binary hash; the later checks are not a
fully pristine upstream-runtime verification. Raw historical records and the
experimental correction are intentionally outside this minimal commit. Readers
can independently run this script against their own upstream installation.

The isolated discrepancy concerns propagated input gradients, not a claim that
every local weight-update error is zeroed. No training-accuracy effect or impact
on all AIHWKit research is claimed. Quantized/noisy models and derivatives
through dynamic management require separate analysis.
