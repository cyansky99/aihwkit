I ran a training comparison to check whether the gradient-mask behavior affects accuracy in a small MNIST model. In this setup, the original mask gave lower mean accuracy and much larger variation between repeated runs.

Each condition was run ten times for 30 epochs on full MNIST. The model is a dense 784–256–128–10 network with sigmoid hidden layers, digital bias and standalone ConstantStepDevice. Training uses AnalogSGD, constant LR 1.5 and batch size 16. Pulse-size noise is zero, but update pulse generation is still stochastic.

| Condition | Mask | Forward noise / bound management | Final test accuracy, mean ± sample SD (%) |
| --- | --- | --- | ---: |
| legacy | Original threshold | ABS_MAX / ITERATIVE | 96.577 ± 1.372 |
| native | Experimental final ADC clipping mask in my fork | ABS_MAX / ITERATIVE | 98.366 ± 0.066 |
| disabled | No output-gradient mask | ABS_MAX / ITERATIVE | 98.347 ± 0.089 |
| workaround | Original threshold | NONE / NONE | 98.267 ± 0.094 |

These are epoch-30 results from all ten trials, not the best epochs. Backward noise management remains ABS_MAX in every condition. The first three conditions change only the mask policy. The workaround also changes forward computation, so it is a separate control rather than another mask implementation.

![MNIST learning curves: four conditions, ten trials each](https://raw.githubusercontent.com/cyansky99/aihwkit/621ae1c69c4b5aa789893f41ae1c68c3e4678f60/issue_experiments/output_gradient_mask/mnist/sample_results/seed0_trials10_epochs30/training_comparison_noise_0_seed_0.png)

Thick lines show the mean; shaded regions show ±1 sample SD across trials; thin lines show individual runs. The center panels magnify epochs 20–30. Training accuracy and loss are accumulated during updates. Test accuracy uses all 10,000 test images after each epoch.

All trials use nominal seed 0 with identical programmed initial weights and data order. Backend pulse randomness is not paired, so these are repeated runs at one initialization, not ten different seeds. The first three conditions were measured together; the workaround was measured later in a separate campaign using the same simulator source and compiled binding. All measurements use the experimental fork build with the original behavior retained in legacy mode.

For the workaround I disabled only these two forward settings:

```python
from aihwkit.simulator.parameters.enums import BoundManagementType, NoiseManagementType

rpu_config.forward.noise_management = NoiseManagementType.NONE
rpu_config.forward.bound_management = BoundManagementType.NONE
```

These settings already exist upstream. In this setup they avoid the dynamic scale-restoration mismatch, but disabling management can increase real clipping and leaves the legacy 0.999 threshold in place. I would not treat this as a general replacement for a clipping-aware mask. The disabled-mask condition is also only a diagnostic control because it passes gradients through real clipping.

The [full settings, individual results and provenance](https://github.com/cyansky99/aihwkit/blob/621ae1c69c4b5aa789893f41ae1c68c3e4678f60/issue_experiments/output_gradient_mask/mnist/sample_results/seed0_trials10_epochs30/README.md) and [four-condition runner](https://github.com/cyansky99/aihwkit/blob/621ae1c69c4b5aa789893f41ae1c68c3e4678f60/issue_experiments/output_gradient_mask/mnist/README.md) are available in my fork. The runner now executes all four conditions directly; `native` and the shared diagnostics require the fork's matching build. Saved figures can be redrawn without AIHWKit or a GPU. These results show a training effect in this configuration; they do not establish its size in other workloads.
