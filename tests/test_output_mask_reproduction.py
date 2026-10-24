# Licensed under the MIT license. See LICENSE.txt in the project root.
"""Numerical and artifact checks for the public reproduction tools."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from aihwkit.simulator import rpu_base

TOOLS = Path(__file__).resolve().parents[1] / "issue_experiments/output_gradient_mask"
sys.path.insert(0, str(TOOLS))
import reproduce_upstream as upstream
import compare_modes as comparison
import plot_results as plotting


@pytest.fixture(scope="module")
def result():
    return upstream.collect(
        "cpu",
        [(m, True) for m in plotting.COLORS],
        "three_modes",
        comparison.configure,
        comparison.check_tile,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("experiment", ["three_modes", "upstream_mask_toggle"])
def test_measured_controls(device, experiment):
    if device == "cuda" and not (torch.cuda.is_available() and rpu_base.cuda.is_compiled()):
        pytest.skip("CUDA GPU/native binding unavailable")
    if experiment == "three_modes":
        r = upstream.collect(
            device,
            [(m, True) for m in plotting.COLORS],
            experiment,
            comparison.configure,
            comparison.check_tile,
        )
    else:
        r = upstream.collect(device, [("legacy", True), ("disabled", False)], experiment)
    for row in r["cases"]:
        assert row["output"] == (1.0 if row["case"] == "clipped" else 2.0)
        expected = (
            0.0
            if row["mode"] == "legacy" or (row["mode"] == "native" and row["case"] == "clipped")
            else 1.0
        )
        assert row["input_gradient"] == [expected, expected]
        numerical = 0.0 if row["case"] == "clipped" else 1.0
        assert row["finite_difference_input_gradient"] == pytest.approx([numerical] * 2, abs=2e-3)
        assert row["actual_weights"] == [[1.0, 1.0]]
    assert len(r["environment"]["binding_sha256"]) == 64


def test_saved_csv_and_plotted_values_match_measurements(result, tmp_path):
    output = tmp_path / "case.json"
    upstream.save_result(result, output)
    saved = json.loads(output.read_text())
    assert saved == result
    with output.with_suffix(".csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    for raw, row in zip(result["cases"], csv_rows):
        assert float(row["output"]) == raw["output"]
        for i in range(2):
            assert float(row[f"dx{i+1}"]) == raw["input_gradient"][i]
            assert float(row[f"fd_dx{i+1}"]) == raw["finite_difference_input_gradient"][i]
    fig = plotting.build_figure(saved)
    try:
        for col, scenario in enumerate(plotting.SCENARIOS):
            rows = [row for row in saved["cases"] if row["case"] == scenario]
            assert [p.get_height() for p in fig.axes[col].patches] == [r["output"] for r in rows]
            assert [p.get_height() for p in fig.axes[col + 3].patches] == [
                v for r in rows for v in r["input_gradient"]
            ]
            # Scatter collections alternate measured autograd marks and FD crosses.
            fd = [float(c.get_offsets()[0, 1]) for c in fig.axes[col + 3].collections[1::2]]
            assert fd == [v for r in rows for v in r["finite_difference_input_gradient"]]
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "failed", "nan"])
def test_invalid_data_never_makes_a_figure(result, tmp_path, damage):
    bad = deepcopy(result)
    if damage == "missing":
        bad["cases"].pop()
    elif damage == "duplicate":
        bad["cases"].append(deepcopy(bad["cases"][0]))
    elif damage == "failed":
        bad["cases"][0]["status"] = "error"
    else:
        bad["cases"][0]["input_gradient"][0] = float("nan")
    with pytest.raises(ValueError):
        plotting.render_result(bad, tmp_path / "bad")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("suffix", [".json", ".csv", ".summary.txt", ".png", ".pdf"])
def test_any_existing_artifact_prevents_measurement(tmp_path, monkeypatch, suffix):
    sentinel = tmp_path / ("case" + suffix)
    sentinel.write_text("keep")

    def unexpected(*args, **kwargs):
        pytest.fail("measurement must not start")

    monkeypatch.setattr(upstream, "collect", unexpected)
    args = SimpleNamespace(device="cpu", output=tmp_path / "case.json", plot=True)
    with pytest.raises(FileExistsError):
        upstream.execute(args, [("legacy", True)], "three_modes")
    assert sentinel.read_text() == "keep"


def test_plot_only_needs_no_torch_or_aihwkit(result, tmp_path):
    source = tmp_path / "data.json"
    source.write_text(json.dumps(result))
    code = "import sys,runpy; sys.modules['torch']=None; sys.modules['aihwkit']=None; runpy.run_path(sys.argv[1],run_name='__main__')"
    # run_path leaves argv[1:] intact, so adjust argv before invoking the script.
    code = code.replace(
        "runpy.run_path(sys.argv[1],run_name='__main__')",
        "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(TOOLS / "plot_results.py"),
            "--input",
            str(source),
            "--output-prefix",
            str(tmp_path / "render"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "render.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "render.pdf").read_bytes().startswith(b"%PDF")
    with pytest.raises(FileExistsError):
        plotting.render_result(result, tmp_path / "render")


def test_missing_api_and_cuda_fail_explicitly(monkeypatch):
    with pytest.raises(RuntimeError, match="Build this fork"):
        comparison.configure(SimpleNamespace(forward=SimpleNamespace()), "native")
    tile = SimpleNamespace(
        tile=SimpleNamespace(), get_output_gradient_mask_modes=lambda: ("native", "native")
    )
    with pytest.raises(RuntimeError, match="native binding"):
        comparison.check_tile(tile, "native")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        upstream.collect("cuda", [("legacy", True)], "upstream_mask_toggle")


def test_plot_requires_output():
    with pytest.raises(ValueError, match="requires --output"):
        upstream.execute(SimpleNamespace(output=None, plot=True), [], "three_modes")
