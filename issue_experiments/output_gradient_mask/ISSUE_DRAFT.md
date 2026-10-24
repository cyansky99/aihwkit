# Output-bound masking can silently zero valid input gradients after bound management or perfect forward

## Description

The Python output-bound mask can silently produce incorrect input gradients:
the forward pass returns the correct linear output, but backward returns zero
instead of a nonzero gradient, without raising an error. Checking only forward
outputs would miss this. The returned output exceeds the nominal ADC bound,
even though no clipping occurred.

This occurs in a small dense-layer example with iterative bound management
and also with `forward.is_perfect=True`.

The returned value is compared against `out_scale * out_bound * 0.999` after
the native code restores the output scale. At that point, exceeding the threshold
does not necessarily mean the ADC clipped the value. Perfect forward should not
apply ADC clipping at all.

I checked the upstream source at commit
[`7f22aa7`](https://github.com/IBM/aihwkit/tree/7f22aa7ff0054609fa120d1c2696414460f13f91)
(source version string 1.2.0). This report covers input-gradient propagation.
I have not established how much this affects training accuracy or which other
workloads are affected.

## How to reproduce

1. Use an environment with an existing AIHWKit installation.
2. Download the standalone
   [reproduction script](https://github.com/cyansky99/aihwkit/blob/a0ae95a/issue_experiments/output_gradient_mask/reproduce_upstream.py).
   It uses existing upstream APIs and requires no fork-specific settings,
   rebuild, dataset, or training run.
3. Run:

```bash
python reproduce_upstream.py --device cpu --output upstream_cpu.json
# Optional: requires a CUDA-enabled installation and an available GPU.
python reproduce_upstream.py --device cuda --output upstream_cuda.json
```

Check the printed package and binding paths to confirm which installation
was used. Each command saves JSON, CSV, and a terminal summary; use fresh output
filenames when repeating it. The script runs six cases, including controls with
`handle_output_bound=False`.

The setup uses `W=[1,1]`, `x=[1,1]`, no bias, ADC bound 1, and DAC bound 2.
Noise, quantization, and noise management are disabled. Backward is perfect,
and the loss is `sum(y)`. No optimizer step is taken. The script also saves
central finite-difference estimates of the input gradients.

I obtained the following results on CPU and CUDA using the experimental build
described below, which preserves the original masking behavior:

| Forward case | Mask enabled | Output | Observed dL/dx | Expected dL/dx |
| --- | --- | --- | --- | --- |
| Iterative bound management | Yes | 2 | [0,0] | [1,1] |
| Iterative bound management | No | 2 | [1,1] | [1,1] |
| Perfect forward | Yes | 2 | [0,0] | [1,1] |
| Perfect forward | No | 2 | [1,1] | [1,1] |
| Actual clipping, management off | Yes | 1 | [0,0] | [0,0] |
| Actual clipping, management off | No | 1 | [1,1] | [0,0] |

Finite differences are approximately `[0.99993, 0.99993]` for the managed and
perfect cases, and `[0, 0]` for actual clipping.

## Expected behavior

For the managed and perfect examples, the returned output is the exact linear
result, so the input gradient should be `[1,1]`. With actual clipping, the input
gradient should remain zero. The last row shows why disabling the mask
altogether would not fix both cases.

Masking should follow the actual clipping decision before output scale
restoration, and should not suppress gradients for perfect forward.

## Other information

* Pytorch version: 2.9.1+cu126 (Torch CUDA runtime 12.6)
* Package version: source version 1.2.0, based on upstream commit `7f22aa7`; experimental fork used for the attached measurements
* OS: Ubuntu 22.04.4 LTS under WSL2, Linux 6.18.54.1-microsoft-standard-WSL2, x86_64
* Python version: 3.12.12
* Conda version (or N/A): 24.11.3
* GPU: NVIDIA GeForce RTX 4080 SUPER

The reproduction above uses existing AIHWKit APIs and requires no rebuild.
The three-mode comparison below requires building this fork, since `native`
uses a new C++/CUDA binding.

The fork includes an experimental comparison between `legacy` (original
masking), `native` (actual final ADC clipping mask), and `disabled` (no mask).
All three produce the same forward outputs in these examples; `native` gives
the expected input gradients in all three cases.

![CPU comparison of outputs, autograd gradients, and finite differences](https://raw.githubusercontent.com/cyansky99/aihwkit/a0ae95a/issue_experiments/output_gradient_mask/sample_results/modes_cpu.png)

[Reproduction and comparison instructions](https://github.com/cyansky99/aihwkit/blob/a0ae95a/issue_experiments/output_gradient_mask/README.md)
and [CPU/CUDA raw results and figures](https://github.com/cyansky99/aihwkit/tree/a0ae95a/issue_experiments/output_gradient_mask/sample_results)
are available. The JSON records include source state, package/binding paths,
and the binding hash.

All attached measurements, including the mask-on/off runs using the existing
API, were made with the experimental build, not a fresh unmodified upstream
build. The standalone script can also be run with an upstream installation.

The error signal stored for local weight updates is separate from the masked
input gradient, so a zero input gradient does not mean every local weight update
is zero. No training-accuracy comparison is included here.

If using the final native clipping decision matches the intended backward
behavior, I can prepare a focused PR with regression tests.
