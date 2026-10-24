# Output-bound masking can zero valid input gradients after scale restoration or perfect forward

## Description

The Python output-bound mask can suppress a valid input gradient when a returned
forward output exceeds the nominal ADC bound but was not actually clipped.
This occurs in a small dense-layer example with iterative bound management,
and also with `forward.is_perfect=True`.

The returned value is compared against `out_scale * out_bound * 0.999` after
native output scale restoration. That threshold need not describe clipping in
the internal ADC coordinates. Perfect forward should not introduce ADC clipping.

The source baseline examined is `7f22aa7ff0054609fa120d1c2696414460f13f91`
(source version string 1.2.0). The observations below were obtained locally.
Historical records lack a contemporaneous original-binary hash, and later
independent CPU checks reused an experimental native binary with original Python.
They are not presented as a fully authenticated pristine-build reproduction.
The standalone script enables independent verification against upstream.

## How to reproduce

1. Use an environment with an unmodified AIHWKit installation.
2. Copy `issue_experiments/output_gradient_mask/reproduce_upstream.py` from
   this fork. It only uses existing upstream APIs and does not require a dataset.
3. Run `python reproduce_upstream.py --device cpu --output upstream_cpu.json`.
4. If CUDA is available, repeat with `--device cuda --output upstream_cuda.json`.
5. Check the printed package/binding paths to ensure the intended installation
   was used. The script runs six cases, including diagnostic mask-off controls.

The setup uses W=[1,1], x=[1,1], no bias, ADC bound 1, DAC bound 2,
noise/quantization off, no noise management, perfect backward, and `loss=sum(y)`.
No optimizer step is involved. Central finite differences are also saved.

| Forward case | Mask enabled | Output | Observed dL/dx | Expected dL/dx |
| --- | --- | --- | --- | --- |
| Iterative bound management | Yes | 2 | [0,0] | [1,1] |
| Iterative bound management | No | 2 | [1,1] | [1,1] |
| Perfect forward | Yes | 2 | [0,0] | [1,1] |
| Perfect forward | No | 2 | [1,1] | [1,1] |
| Actual clipping, management off | Yes | 1 | [0,0] | [0,0] |
| Actual clipping, management off | No | 1 | [1,1] | [0,0] |

## Expected behavior

For the managed and perfect examples, the returned output is the exact linear
result and the input gradient should be [1,1]. For actual clipping the input
gradient should remain zero. The diagnostic mask-off control in the last row
shows why simply removing all masking is not a general solution.

## Other information

- PyTorch: 2.9.1+cu126; reported Torch CUDA runtime 12.6.
- Package source version: 1.2.0; pinned source commit above, not a claim of a
  separately tested release-tag build.
- OS: Ubuntu userspace under WSL2 Linux (the script records the full platform).
- Python: 3.12.12 (Conda distribution).
- Conda: used; exact Conda version was not captured in the original evidence.
- GPU used for earlier local observations: NVIDIA GeForce RTX 4080 SUPER.

The original commit `dda0a9a` contains only the reproducer and explanation.
Later commits provide an experimental legacy/native/disabled implementation and
small measured comparison figures. Those new samples use the experimental binary,
including the existing-API mask-on/off samples; they are not pristine-upstream
verification. Training comparisons are not included.

The stored local weight-update error is separate from the masked input-gradient
path; this report does not claim that every local weight update is directly
zeroed. Training accuracy is outside this report; no training-impact conclusion
is included or required for this numerical reproduction.

Would recording the final native clipping decision be consistent with the
intended input-gradient semantics? I can prepare a focused PR and regression
tests based on maintainer feedback.

<!-- Before submission: link the published commit/permalinks and attached raw
results; check for duplicate reports. Add training evidence only if reviewed.
This draft has not been submitted. -->
