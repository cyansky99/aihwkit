"""Controlled configuration, transparent diagnostics, and campaign recovery."""
import json

import pytest
import torch

from issue_experiments.output_gradient_mask.mnist.run_constantstep_noise import (
    Diagnostics, MODES, aggregate, encode, epoch_order, fingerprint, is_complete,
    make_config, make_model, training_summary, write_csv, write_json,
)


def test_configuration_changes_only_selected_factors():
    reference = encode(make_config("legacy", 0., 0))
    assert reference["device"]["dw_min"] == .002
    assert reference["mapping"]["max_input_size"] == 512
    for mode in MODES:
        cfg = encode(make_config(mode, .02, 0))
        if mode == "workaround":
            assert cfg["forward"]["noise_management"] == "NONE"
            assert cfg["forward"]["bound_management"] == "NONE"
            assert cfg["backward"] == reference["backward"]
            assert cfg["forward"]["output_gradient_mask_mode"] == "legacy"
            cfg["forward"]["noise_management"] = reference["forward"]["noise_management"]
            cfg["forward"]["bound_management"] = reference["forward"]["bound_management"]
        cfg["forward"]["output_gradient_mask_mode"] = "legacy"
        cfg["device"]["dw_min_std"] = 0.
        assert cfg == reference
    assert reference["forward"]["inp_res"] == 1 / 4094
    assert reference["backward"]["is_perfect"] is False
    assert reference["update"]["update_bl_management"] is True


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initial_weights_mapping_and_order(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Actual CUDA GPU unavailable")
    hashes = []
    for mode in MODES:
        _, _, digest, shapes = make_model(make_config(mode, 0., 0), 0, device)
        hashes.append(digest)
        assert shapes == [[256, 392], [256, 392], [128, 256], [10, 128]]
    assert len(set(hashes)) == 1
    first = epoch_order(200, 0, 1)
    torch.testing.assert_close(first, epoch_order(200, 0, 1))
    assert not torch.equal(first, epoch_order(200, 0, 2))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", MODES)
def test_diagnostics_do_not_change_forward_gradient_or_update(mode, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Actual CUDA GPU unavailable")
    from aihwkit.optim import AnalogSGD
    from aihwkit.simulator.parameters.enums import PulseType
    cfg = make_config(mode, 0., 0)
    # Deterministic updates isolate instrumentation from unpaired backend pulse RNG.
    cfg.update.pulse_type = PulseType.NONE
    model, _, _, _ = make_model(cfg, 0, device)
    for layer in (model[0], model[2], model[4]):
        w, b = layer.get_weights()
        layer.set_weights(torch.full_like(w, .15), torch.zeros_like(b))
    initial = [tuple(t.detach().clone() for t in l.get_weights())
               for l in (model[0], model[2], model[4])]
    results = []
    for observe in (False, True):
        for layer, (w, b) in zip((model[0], model[2], model[4]), initial):
            layer.set_weights(w, b)
        model.zero_grad()
        diag = Diagnostics(model, condition=mode) if observe else None
        if diag:
            diag.active = True
            diag.epoch = diag.batch = 1
        opt = AnalogSGD(model.parameters(), lr=.05)
        opt.regroup_param_groups(model)
        x = torch.full((16, 784), .3, device=device, requires_grad=True)
        labels = torch.arange(16, device=device) % 10
        rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
        pred = model(x)
        torch.nn.functional.cross_entropy(pred, labels).backward()
        opt.step()
        assert torch.equal(rng, torch.random.get_rng_state())
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        results.append((pred.detach().clone(), x.grad.clone(),
                        [tuple(t.detach().clone() for t in l.get_weights())
                         for l in (model[0], model[2], model[4])]))
        if diag:
            assert len(diag.rows) == 4
            assert {r["mode"] for r in diag.rows} == {mode}
            assert {r["mask_mode"] for r in diag.rows} == {"legacy" if mode == "workaround" else mode}
            if mode == "native":
                assert any(r["legacy_false_positive_fraction"] > 0 for r in diag.rows)
            if mode == "disabled":
                assert all(r["mask_fraction"] == 0 for r in diag.rows)
            diag.close()
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


def test_completion_requires_full_history_and_matching_identity(tmp_path):
    spec = dict(epochs=2, mode="native")
    write_json(tmp_path / "status.json", dict(state="running", fingerprint=fingerprint(spec)))
    assert not is_complete(tmp_path, spec)
    write_json(tmp_path / "status.json", dict(state="completed", fingerprint=fingerprint(spec)))
    write_json(tmp_path / "summary.json", dict(fingerprint=fingerprint(spec)))
    (tmp_path / "final_model.pt").write_bytes(b"test")
    write_csv(tmp_path / "history.csv", [dict(epoch=1)])
    assert not is_complete(tmp_path, spec)
    write_csv(tmp_path / "history.csv", [dict(epoch=1), dict(epoch=2)])
    assert is_complete(tmp_path, spec)
    assert not is_complete(tmp_path, dict(spec, mode="legacy"))


def test_summary_and_partial_plot(tmp_path):
    spec = dict(epochs=2, mode="legacy", pulse_noise=0., seed=0)
    write_json(tmp_path / "campaign.json", dict(jobs=[dict(path="run", spec=spec)]))
    (tmp_path / "run").mkdir()
    row = dict(epoch=1, train_loss=.2, train_accuracy=.95, test_accuracy=.94, mask_fraction=.3)
    write_csv(tmp_path / "run/history.csv", [row])
    aggregate(tmp_path, final=True)
    assert (tmp_path / "training_comparison_noise_0_seed_0.png").is_file()
    assert (tmp_path / "training_comparison_noise_0_seed_0.pdf").is_file()
    assert json.loads((tmp_path / "progress.json").read_text())["completed"] == 0
    stats = training_summary([row, dict(row, test_accuracy=.9)])
    assert stats["max_epoch_drop"] == pytest.approx(.04)
    assert stats["last5_range"] == pytest.approx(.04)
