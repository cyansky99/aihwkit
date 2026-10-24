# Licensed under the MIT license. See LICENSE.txt in the project root.
"""Native clipping evidence and legacy/disabled controls on CPU and CUDA."""
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch

from aihwkit.exceptions import ConfigError
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator import rpu_base
from aihwkit.simulator.configs import SingleRPUConfig, InferenceRPUConfig, TorchInferenceRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.parameters import BoundManagementType as BM, NoiseManagementType as NM
from aihwkit.simulator.parameters.enums import AnalogMVType, RPUDataType, PulseType
from aihwkit.simulator.parameters.io import IOParameters, IOParametersIRDropT
from aihwkit.simulator.parameters.helpers import parameters_to_bindings
from aihwkit.simulator.tiles import AnalogTile


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not (torch.cuda.is_available() and rpu_base.cuda.is_compiled()):
        pytest.skip("CUDA GPU or CUDA binding unavailable")
    return request.param


def config(mode="native", bm=BM.NONE, perfect=False):
    cfg = SingleRPUConfig(
        device=ConstantStepDevice(
            w_min=-10.0,
            w_max=10.0,
            w_min_dtod=0.0,
            w_max_dtod=0.0,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down_dtod=0.0,
        )
    )
    cfg.forward.output_gradient_mask_mode = mode
    cfg.forward.is_perfect = perfect
    cfg.forward.inp_bound = 20.0
    cfg.forward.out_bound = 1.0
    cfg.forward.inp_res = cfg.forward.out_res = -1.0
    cfg.forward.inp_noise = cfg.forward.out_noise = 0.0
    cfg.forward.noise_management = NM.NONE
    cfg.forward.bound_management = bm
    cfg.backward.is_perfect = True
    return cfg


def layer_for(cfg, device, weights=None):
    weights = torch.ones(1, 2) if weights is None else torch.tensor(weights, dtype=torch.float32)
    layer = AnalogLinear(weights.shape[1], weights.shape[0], bias=False, rpu_config=cfg)
    layer.set_weights(weights)
    return layer.to(device)


def run(layer, values):
    device = next(layer.analog_tiles()).device
    x = torch.tensor(values, device=device, dtype=torch.float32, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    return y.detach(), x.grad.detach()


def assert_values(actual, expected):
    torch.testing.assert_close(
        actual.cpu(), torch.tensor(expected, dtype=actual.dtype), atol=2e-5, rtol=2e-5
    )


@pytest.mark.parametrize("mode", ["legacy", "native", "disabled"])
@pytest.mark.parametrize("scenario", ["iterative", "perfect", "clipping"])
def test_three_modes(device, mode, scenario):
    cfg = config(mode, BM.ITERATIVE if scenario == "iterative" else BM.NONE, scenario == "perfect")
    y, dx = run(layer_for(cfg, device), [[1.0, 1.0]])
    assert_values(y, [[1.0 if scenario == "clipping" else 2.0]])
    expected = 0.0 if mode == "legacy" or (mode == "native" and scenario == "clipping") else 1.0
    assert_values(dx, [[expected, expected]])


def test_defaults_binding_inheritance_and_validation(device):
    assert IOParameters().output_gradient_mask_mode == "legacy"
    for cls in (IOParameters, IOParametersIRDropT):
        params = cls(output_gradient_mask_mode="native")
        bound = parameters_to_bindings(params, RPUDataType.FLOAT)
        assert not hasattr(bound, "output_gradient_mask_mode")
    layer = layer_for(config("invalid"), device)
    with pytest.raises(ConfigError, match="output_gradient_mask_mode"):
        layer(torch.ones(1, 2, device=device))
    layer = layer_for(config("legacy", BM.ITERATIVE), device)
    tile = next(layer.analog_tiles())
    del tile.rpu_config.forward.output_gradient_mask_mode
    assert_values(run(layer, [[1.0, 1.0]])[1], [[0.0, 0.0]])
    tile.handle_output_bound = False
    assert_values(run(layer, [[1.0, 1.0]])[1], [[1.0, 1.0]])


def test_boundaries_signed_and_unlimited(device):
    layer = layer_for(config(), device, [[1.0]])
    values = [[-1.2], [-1.0], [-0.9995], [0.0], [0.9995], [1.0], [1.2]]
    y, dx = run(layer, values)
    assert_values(y, [[-1.0], [-1.0], [-0.9995], [0.0], [0.9995], [1.0], [1.0]])
    assert_values(dx, [[0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0]])
    cfg = config()
    cfg.forward.out_bound = 0.0
    assert_values(run(layer_for(cfg, device, [[1.0]]), [[2.0]])[1], [[1.0]])


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("bm", [BM.NONE, BM.ITERATIVE, BM.ITERATIVE_WORST_CASE])
def test_noise_management_final_mask_and_zero_input(device, batch, bm):
    cfg = config(bm=bm)
    cfg.forward.noise_management = NM.ABS_MAX
    cfg.forward.inp_bound = 1.0
    cfg.forward.out_scale = 3.0
    layer = layer_for(cfg, device, [[1.0, 0.0]])
    tile = next(layer.analog_tiles())
    x = torch.tensor([[4.0, 0.0]] * batch, device=device)
    y, mask = tile.tile.forward_with_output_mask(x)
    assert_values(y, [[12.0]] * batch)
    assert not mask.any()  # Restored output exceeds ADC bound but did not clip.
    zero_y, zero_mask = tile.tile.forward_with_output_mask(torch.zeros_like(x))
    assert not zero_y.any() and not zero_mask.any()
    assert not mask.any()  # Owned independently of the next call.


def test_retry_limits_and_negative_clipping(device):
    cfg = config(bm=BM.ITERATIVE)
    cfg.forward.max_bm_factor = 1
    layer = layer_for(cfg, device)
    y, mask = next(layer.analog_tiles()).tile.forward_with_output_mask(
        torch.tensor([[8.0, 8.0], [0.1, 0.1]], device=device)
    )
    assert_values(y, [[2.0], [0.2]])
    assert_values(mask, [[True], [False]])
    cfg = config(bm=BM.ITERATIVE)
    cfg.forward.bm_test_negative_bound = False
    # Batch path: mask must record negative saturation even when it cannot trigger a retry.
    tile = next(layer_for(cfg, device).analog_tiles())
    y, mask = tile.tile.forward_with_output_mask(
        torch.tensor([[-1.0, -1.0], [0.1, 0.1]], device=device)
    )
    assert_values(y, [[-1.0], [0.2]])
    assert_values(mask, [[True], [False]])


@pytest.mark.parametrize("x_trans", [False, True])
@pytest.mark.parametrize("d_trans", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_binding_batch_layout_bias_and_ownership(device, x_trans, d_trans, bias):
    cfg = config()
    cfg.mapping.digital_bias = False
    tile = AnalogTile(2, 2, cfg, bias=bias).to(device)
    tile.set_weights(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0.2, -0.2]) if bias else None
    )
    x = torch.tensor([[0.2, 1.4], [1.2, -0.2], [-1.4, 0.1]], device=device)
    expected = x + (torch.tensor([0.2, -0.2], device=device) if bias else 0.0)
    want_mask = (expected > 1.0) | (expected < -1.0)
    y, mask = tile.tile.forward_with_output_mask(x.T if x_trans else x, bias, x_trans, d_trans)
    torch.testing.assert_close(
        y, expected.clamp(-1.0, 1.0).T if d_trans else expected.clamp(-1.0, 1.0)
    )
    torch.testing.assert_close(mask, want_mask.T if d_trans else want_mask)
    saved = mask.clone()
    tile.tile.forward_with_output_mask(
        torch.zeros_like(x.T if x_trans else x), bias, x_trans, d_trans
    )
    assert torch.equal(mask, saved)


def test_snapshot_reversed_backward_clone_and_serialization(device):
    layer = layer_for(config(), device, [[1.0]])
    tile = next(layer.analog_tiles())
    a = torch.tensor([[2.0]], device=device, requires_grad=True)
    b = torch.tensor([[0.3]], device=device, requires_grad=True)
    ya, yb = layer(a), layer(b)
    tile.rpu_config.forward.output_gradient_mask_mode = "disabled"
    tile.handle_output_bound = False
    yb.sum().backward()
    ya.sum().backward()
    assert_values(a.grad, [[0.0]])
    assert_values(b.grad, [[1.0]])
    tile.handle_output_bound = True
    tile.rpu_config.forward.output_gradient_mask_mode = "native"
    clone = deepcopy(layer)
    assert_values(run(clone, [[2.0]])[1], [[0.0]])
    data = BytesIO()
    torch.save(layer.state_dict(), data)
    data.seek(0)
    restored = layer_for(config("legacy"), device, [[1.0]])
    restored.load_state_dict(torch.load(data, weights_only=False))
    assert next(restored.analog_tiles()).rpu_config.forward.output_gradient_mask_mode == "native"
    assert_values(run(restored, [[2.0]])[1], [[0.0]])


def test_mapping_and_input_range_share_saved_mask(device):
    cfg = config(bm=BM.ITERATIVE)
    cfg.mapping.weight_scaling_omega = 0.5
    layer = layer_for(cfg, device)
    assert_values(run(layer, [[1.0, 1.0]])[1], [[1.0, 1.0]])
    cfg = config()
    ir = cfg.pre_post.input_range
    ir.enable = True
    ir.init_value = 2.0
    ir.init_from_data = 0
    ir.decay = 0.0
    ir.manage_output_clipping = True
    tile = next(layer_for(cfg, device).analog_tiles())
    x = torch.tensor([[1.5, 1.5]], device=device)
    ctx = SimpleNamespace(saved_analog_tensors=[x])
    tile.joint_forward(x, False, ctx)
    assert ctx.saved_analog_tensors[1].item()
    assert not tile.pre_backward(torch.ones(1, 1, device=device), 1, ctx).any()
    tile.post_backward(torch.zeros_like(x), 1, ctx)
    assert tile.input_range.grad.item() < 0  # Same mask drives range management.


def test_native_unsupported_paths_and_master_switch(device):
    cfg = config()
    cfg.forward.mv_type = AnalogMVType.POS_NEG_SEPARATE
    layer = layer_for(cfg, device)
    tile = next(layer.analog_tiles())
    with pytest.raises(ConfigError, match="ONE_PASS"):
        layer(torch.ones(1, 2, device=device))
    tile.handle_output_bound = False
    layer(torch.ones(1, 2, device=device))  # Explicit master disable takes precedence.
    tile = next(layer_for(config(), device).analog_tiles())
    with pytest.raises(ConfigError, match="indexed"):
        tile.joint_forward_indexed(torch.ones(1, 1, 3, device=device))
    tile.tile = SimpleNamespace(forward=lambda *args: None)
    with pytest.raises(ConfigError, match="rebuilt"):
        tile.joint_forward(torch.ones(1, 2, device=device))
    cfg = TorchInferenceRPUConfig()
    cfg.forward.output_gradient_mask_mode = "native"
    layer = layer_for(cfg, device)
    next(layer.analog_tiles()).handle_output_bound = True
    with pytest.raises(ConfigError, match="backend"):
        layer(torch.ones(1, 2, device=device))


def test_ideal_alias(device):
    cfg = config()
    cfg.forward.mv_type = AnalogMVType.IDEAL
    assert_values(run(layer_for(cfg, device), [[1.0, 1.0]])[1], [[1.0, 1.0]])


def test_finite_difference_and_small_training(device):
    layer = layer_for(config(bm=BM.ITERATIVE), device)
    x = torch.tensor([[0.8, 0.9]], device=device, requires_grad=True)
    layer(x).sum().backward()
    for i in range(2):
        plus, minus = x.detach().clone(), x.detach().clone()
        plus[0, i] += 0.001
        minus[0, i] -= 0.001
        fd = (layer(plus).sum() - layer(minus).sum()) / 0.002
        torch.testing.assert_close(fd, x.grad[0, i], atol=0.001, rtol=0.001)
    cfg = config(bm=BM.ITERATIVE)
    cfg.update.pulse_type = PulseType.NONE
    model = AnalogSequential(
        layer_for(cfg, device, [[0.5, 0.5], [0.5, -0.5]]), layer_for(cfg, device)
    )
    optimizer = AnalogSGD(model.parameters(), lr=0.01)
    optimizer.regroup_param_groups(model)
    first = next(model.analog_tiles())
    before = first.get_weights()[0].clone()
    inp = torch.tensor([[0.8, 0.9]], device=device, requires_grad=True)
    model(inp).square().mean().backward()
    optimizer.step()
    assert torch.isfinite(inp.grad).all() and inp.grad.abs().sum() > 0
    assert not torch.equal(before, first.get_weights()[0])


def test_forward_rng_and_mode_invariance(device):
    # Deterministic forward invariance. Stochastic RNG equivalence is checked
    # separately in native_mask_rng_check.cpp, using native RNG state control.
    cfg = config(bm=BM.ITERATIVE)
    cfg.forward.out_noise = 0.0
    cfg.forward.out_sto_round = False
    cfg.forward.out_res = 0.01
    sequences = []
    for mode in ("legacy", "native", "disabled"):
        layer = layer_for(cfg, device)
        tile = next(layer.analog_tiles())
        tile.rpu_config.forward.output_gradient_mask_mode = mode
        sequences.append(
            [
                layer(torch.tensor([[0.2, 0.3], [0.3, 0.4]], device=device)).detach()
                for _ in range(3)
            ]
        )
    for seq in sequences[1:]:
        for a, b in zip(sequences[0], seq):
            assert torch.equal(a, b)


def test_nonblocking_cuda_stream(device):
    if device != "cuda":
        pytest.skip("CUDA stream test")
    layer = layer_for(config(), device)
    tile = next(layer.analog_tiles())
    tile.non_blocking = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y, dx = run(layer, [[2.0, 2.0]])
    torch.cuda.current_stream().wait_stream(stream)
    assert_values(y, [[1.0]])
    assert_values(dx, [[0.0, 0.0]])


@pytest.mark.parametrize("batch", [1, 3])
def test_mask_uses_quantized_adc_value(device, batch):
    cfg = config()
    cfg.forward.out_res = 0.1  # Native ADC step is 2 * bound * res = .2.
    tile = next(layer_for(cfg, device, [[1.0]]).analog_tiles())
    for value, output, clipped in [(1.05, 1.0, False), (1.11, 1.0, True), (-1.05, -1.0, False)]:
        y, mask = tile.tile.forward_with_output_mask(torch.full((batch, 1), value, device=device))
        assert_values(y, [[output]] * batch)
        assert_values(mask, [[clipped]] * batch)


def test_native_inference_tile_backward(device):
    cfg = InferenceRPUConfig()
    cfg.forward = config(bm=BM.ITERATIVE).forward
    layer = layer_for(cfg, device)
    assert_values(run(layer, [[1.0, 1.0]])[1], [[1.0, 1.0]])


def test_legacy_handle_override_keeps_original_six_controls(device):
    for bm, perfect, clipped in [
        (BM.ITERATIVE, False, False),
        (BM.NONE, True, False),
        (BM.NONE, False, True),
    ]:
        layer = layer_for(config("legacy", bm, perfect), device)
        tile = next(layer.analog_tiles())
        tile.handle_output_bound = False
        y, dx = run(layer, [[1.0, 1.0]])
        assert_values(y, [[1.0 if clipped else 2.0]])
        assert_values(dx, [[1.0, 1.0]])


def test_native_rejects_missing_forward_evidence(device):
    tile = next(layer_for(config(), device).analog_tiles())
    ctx = SimpleNamespace(saved_analog_tensors=[torch.ones(1, 2, device=device)])
    with pytest.raises(ConfigError, match="not returned"):
        tile.post_forward(torch.ones(1, 1, device=device), 1, ctx=ctx)


def test_native_rejects_tile_without_periphery(device):
    from aihwkit.simulator.tiles.analog import AnalogTileWithoutPeriphery

    tile = AnalogTileWithoutPeriphery(1, 2, config()).to(device)
    with pytest.raises(ConfigError, match="Python periphery"):
        tile(torch.ones(1, 2, device=device))
