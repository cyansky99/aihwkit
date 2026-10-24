"""Campaign/statistics verification without MNIST loading or training."""
import builtins
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import pytest

from issue_experiments.output_gradient_mask.mnist import plot_training_comparison as plot
from issue_experiments.output_gradient_mask.mnist import run_constantstep_noise as runner


def save_trial(root, trial_id, values, state="completed", mode="native", seed=0, noise=0.):
    spec = dict(epochs=2, mode=mode, seed=seed, pulse_noise=noise, trial_id=trial_id)
    job = dict(path=f"noise_{noise}/seed_{seed}/trial_{trial_id}/{mode}", spec=spec)
    directory = root / job["path"]
    runner.write_json(directory / "spec.json", spec)
    runner.write_json(directory / "status.json", dict(state=state, fingerprint=runner.fingerprint(spec)))
    runner.write_json(directory / "summary.json", dict(fingerprint=runner.fingerprint(spec)))
    runner.write_csv(directory / "history.csv", [dict(trial_id=trial_id, epoch=i,
                     test_accuracy=value, train_accuracy=value, train_loss=1-value,
                     mask_fraction=.1) for i, value in enumerate(values, 1)])
    return job


def test_default_five_trials_keep_same_seed_and_other_settings(tmp_path):
    args = runner.parse_args([])
    jobs = runner.build_jobs(args, {"binding": "example"}, tmp_path)
    assert len(jobs) == 20
    assert [(j["spec"]["trial_id"], j["spec"]["mode"]) for j in jobs] == [
        (trial, mode) for trial in range(1, 6) for mode in runner.MODES]
    assert {j["spec"]["seed"] for j in jobs} == {0}
    assert {j["spec"]["epochs"] for j in jobs} == {30}
    assert {j["spec"]["pulse_noise"] for j in jobs} == {0.}
    assert len({j["path"] for j in jobs}) == 20
    assert len({runner.fingerprint({k: v for k, v in j["spec"].items()
                                   if k not in ("mode", "trial_id")}) for j in jobs}) == 1
    assert args.output_dir.name == "constantstep_mask_comparison"
    with pytest.raises(SystemExit):
        runner.parse_args(["--trials", "0"])
    args = runner.parse_args(["--trials", "2", "--seeds", "0", "1", "--pulse-noise", "0", ".02"])
    assert len(runner.build_jobs(args, {}, tmp_path)) == 32


def test_resume_and_identity_guards(tmp_path):
    jobs = [save_trial(tmp_path, 1, [.8, .9]), save_trial(tmp_path, 2, [.7], "running")]
    runner.ensure_manifest(tmp_path, jobs, {})
    complete = tmp_path / jobs[0]["path"]
    (complete / "final_model.pt").write_bytes(b"test fixture, not a trained model")
    before = {p.name: p.read_bytes() for p in complete.iterdir()}
    assert not runner.prepare_condition(tmp_path, complete, jobs[0]["spec"])
    assert {p.name: p.read_bytes() for p in complete.iterdir()} == before
    partial = tmp_path / jobs[1]["path"]
    assert runner.prepare_condition(tmp_path, partial, jobs[1]["spec"])
    assert [p.name for p in partial.iterdir()] == ["spec.json"]
    assert len(list((tmp_path / "interrupted").glob("*/history.csv"))) == 1
    runner.ensure_manifest(tmp_path, jobs, {})
    changed = json.loads(json.dumps(jobs))
    changed[0]["spec"]["identity"] = {"binding_sha256": "different"}
    with pytest.raises(RuntimeError, match="changed"):
        runner.ensure_manifest(tmp_path, changed, {})


def test_sample_sd_and_partial_exclusion(tmp_path):
    jobs = [save_trial(tmp_path, 1, [.6, .8]), save_trial(tmp_path, 2, [.8, 1.]),
            save_trial(tmp_path, 3, [.1], "running")]
    runner.write_json(tmp_path / "campaign.json", dict(jobs=jobs))
    loaded = plot.load_jobs(tmp_path)
    epochs, mean, sd = plot.curve_statistics(loaded, "test_accuracy")
    assert epochs == [1, 2]
    assert mean == pytest.approx([.7, .9])
    assert sd == pytest.approx([.2 / 2**.5] * 2)
    stats = plot.trial_statistics(loaded)[0]
    assert stats["completed_trials"] == 2 and stats["planned_trials"] == 3
    assert stats["final_test_accuracy_mean"] == pytest.approx(.9)
    assert stats["final_test_accuracy_sample_sd"] == pytest.approx(.2 / 2**.5)
    assert plot.curve_statistics(loaded[:1], "test_accuracy")[2] is None
    assert plot.curve_statistics(loaded[2:], "test_accuracy") == ([], [], None)
    assert plot.trial_statistics(loaded[2:])[0]["final_test_accuracy_mean"] is None
    runner.aggregate(tmp_path, final=True)
    with (tmp_path / "summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    assert [int(r["trial_id"]) for r in summary] == [1, 2, 3]
    assert summary[-1]["complete"] == "False"
    assert (tmp_path / "trial_summary.csv").is_file()


def test_five_trial_figure_and_grouping(tmp_path):
    jobs = [save_trial(tmp_path, t, [.8 + t * .01, .9 + t * .01], mode=mode)
            for t in range(1, 6) for mode in runner.MODES]
    jobs += [save_trial(tmp_path, 1, [.5], "failed", seed=1),
             save_trial(tmp_path, 1, [.5, .6], noise=.02)]
    runner.write_json(tmp_path / "campaign.json", dict(jobs=jobs))
    loaded = plot.load_jobs(tmp_path)
    fig = plot.make_figure(loaded, 0., 0)
    try:
        assert len(fig.axes) == 6
        assert all(len(ax.collections) == 4 for ax in fig.axes)  # One SD band per mode.
        assert all(len(ax.lines) == 24 for ax in fig.axes)  # Five trials + mean per mode.
        assert all(band.get_edgecolors().size == 0 and all(w == 0 for w in band.get_linewidths())
                   for ax in fig.axes for band in ax.collections)
        assert [t.get_text() for t in fig.legends[0].get_texts()] == [f"{m} · n=5/5" for m in runner.MODES]
        assert fig.axes[0].lines[5].get_ydata() == pytest.approx([83, 93])
        assert fig.axes[0].get_xlim() == pytest.approx((.5, 2.5))
        assert fig.axes[1].get_xlim() == pytest.approx((20, 30))
        assert fig.axes[1].get_ylim() == pytest.approx((97.8, 98.6))
        assert fig.axes[4].get_xlim() == pytest.approx((20, 30))
        assert fig.axes[4].get_ylim() == pytest.approx((99.75, 100.02))
        assert fig.axes[1].lines[5].get_ydata() == pytest.approx(fig.axes[0].lines[5].get_ydata())
        assert fig.axes[4].lines[5].get_ydata() == pytest.approx(fig.axes[3].lines[5].get_ydata())
    finally:
        plt.close(fig)
    paths = plot.render(tmp_path, pdf=True)
    assert len(paths) == 6
    assert all(path.stat().st_size > 1000 for path in paths)
    assert len(list(tmp_path.glob("*.png"))) == 3


def test_absent_failed_and_corrupt_data(tmp_path):
    job = save_trial(tmp_path, 1, [.8], "failed")
    runner.write_json(tmp_path / "campaign.json", dict(jobs=[job]))
    loaded = plot.load_jobs(tmp_path)
    fig = plot.make_figure(loaded, 0., 0)
    try:
        assert not fig.axes[0].collections
        assert len(fig.axes[0].lines) == 1
        assert fig.axes[0].lines[0].get_linestyle() == "--"
    finally:
        plt.close(fig)
    (tmp_path / job["path"] / "history.csv").unlink()
    assert plot.load_jobs(tmp_path)[0]["rows"] == []
    runner.write_csv(tmp_path / job["path"] / "history.csv", [dict(epoch=1,
                     train_loss="nan", test_accuracy=.9, train_accuracy=.9, mask_fraction=0.)])
    with pytest.raises(ValueError, match="Non-finite"):
        plot.load_jobs(tmp_path)


def test_cli_dry_run_and_plot_without_torch(tmp_path):
    job = save_trial(tmp_path, 1, [.8, .9])
    runner.write_json(tmp_path / "campaign.json", dict(jobs=[job]))
    before = (tmp_path / job["path"] / "history.csv").read_bytes()
    # Fresh interpreter: deny heavy backends before importing either public entrypoint.
    code = '''import builtins, runpy, sys
original = builtins.__import__
def blocked(name, *a, **k):
    if name.split('.')[0] in ('torch', 'torchvision', 'aihwkit'):
        raise AssertionError('Unexpected backend import: ' + name)
    return original(name, *a, **k)
builtins.__import__ = blocked
script = sys.argv[1]
sys.path.insert(0, str(__import__('pathlib').Path(script).parent))
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
'''
    subprocess.run([sys.executable, "-c", code, plot.__file__, "--input", str(tmp_path)], check=True)
    dry = subprocess.run([sys.executable, "-c", code, runner.__file__, "--dry-run",
                          "--output-dir", str(tmp_path / "must_not_exist")], capture_output=True, text=True, check=True)
    assert "20 runs × 30 epochs" in dry.stdout
    assert not (tmp_path / "must_not_exist").exists()
    assert (tmp_path / job["path"] / "history.csv").read_bytes() == before


def test_pilot_values_and_original_runner_hash():
    root = Path(runner.__file__).parent / "sample_results/pilot_seed0"
    if not root.exists():
        pytest.skip("Pilot evidence is published in a separate commit")
    jobs = plot.load_jobs(root)
    assert len(jobs) == 3 and all(j["complete"] for j in jobs)
    values = {j["spec"]["mode"]: j["rows"][-1]["test_accuracy"] for j in jobs}
    assert values == pytest.approx(dict(legacy=.9753, native=.9827, disabled=.9821))
    manifest = json.loads((root / "campaign.json").read_text())
    assert hashlib.sha256((root / "executed_runner.py").read_bytes()).hexdigest() == manifest["environment"]["runner_sha256"]
    assert all(s["completed_trials"] == 1 and s["final_test_accuracy_sample_sd"] is None
               for s in plot.trial_statistics(jobs))
    initial_hashes = set()
    orders = []
    for job in manifest["jobs"]:
        directory = root / job["path"]
        initial_hashes.add(json.loads((directory / "configuration.json").read_text())["initial_weights_sha256"])
        orders.append(json.loads((directory / "data_order_sha256.json").read_text()))
    assert len(initial_hashes) == 1
    assert orders[0] == orders[1] == orders[2] and len(orders[0]) == 20
