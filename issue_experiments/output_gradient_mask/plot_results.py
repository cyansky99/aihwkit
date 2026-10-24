#!/usr/bin/env python3
# Licensed under the MIT license. See LICENSE.txt in the project root.
"""Render saved measurements using only the standard library and Matplotlib.

No simulator imports, GPU access, or re-execution of experiments occur here.
"""
import argparse
import json
import math
from pathlib import Path

SCENARIOS = ("managed", "perfect", "clipped")
COLORS = {"legacy": "#d68722", "native": "#2673b8", "disabled": "#37854a"}
TITLES = ("Bound management", "Perfect forward", "Actual clipping")


def validated_rows(result):
    if result.get("schema_version") != 1 or result.get("status") != "completed":
        raise ValueError("Expected a complete schema_version=1 measurement; no figure generated")
    if result.get("experiment") not in ("upstream_mask_toggle", "three_modes"):
        raise ValueError("Unknown experiment")
    modes = result.get("modes", [])
    if not modes or len(set(modes)) != len(modes) or any(m not in COLORS for m in modes):
        raise ValueError("Invalid or missing mode list")
    if result["experiment"] == "upstream_mask_toggle" and modes != ["legacy", "disabled"]:
        raise ValueError("Mask-toggle comparison requires mask-on and mask-off measurements")
    rows = {}
    for row in result["cases"]:
        key = (row["case"], row["mode"])
        if key in rows or row.get("status") != "observed":
            raise ValueError("Duplicate or failed measurement; no figure generated")
        values = [row["output"], row["expected_output"]]
        for field in (
            "input_gradient",
            "expected_input_gradient",
            "finite_difference_input_gradient",
        ):
            if not isinstance(row[field], list) or len(row[field]) != 2:
                raise ValueError(f"Expected two measured input components in {field}")
            values.extend(row[field])
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in values):
            raise ValueError("Non-finite or missing observation; no figure generated")
        rows[key] = row
    if set(rows) != {(case, mode) for case in SCENARIOS for mode in modes}:
        raise ValueError("Incomplete case/mode matrix; no figure generated")
    return modes, rows


def build_figure(result):
    modes, rows = validated_rows(result)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    labels = {
        mode: (
            ("Mask on\n(legacy)" if mode == "legacy" else "Mask off\n(disabled)")
            if result["experiment"] == "upstream_mask_toggle"
            else mode
        )
        for mode in modes
    }
    forward_values = [r["output"] for r in rows.values()] + [
        r["expected_output"] for r in rows.values()
    ]
    grad_values = [
        v
        for r in rows.values()
        for field in (
            "input_gradient",
            "expected_input_gradient",
            "finite_difference_input_gradient",
        )
        for v in r[field]
    ]

    def limits(values):
        lo, hi = min(0.0, *values), max(0.0, *values)
        margin = max(hi - lo, 1.0) * 0.22
        return lo - margin * 0.45, hi + margin

    for col, scenario in enumerate(SCENARIOS):
        top, bottom = axes[0, col], axes[1, col]
        top.set_title(TITLES[col], fontsize=13)
        for index, mode in enumerate(modes):
            row = rows[scenario, mode]
            top.bar(index, row["output"], width=0.55, color=COLORS[mode], zorder=3)
            top.scatter([index], [row["output"]], marker="_", color=COLORS[mode], zorder=5)
            top.annotate(
                f"{row['output']:.4g}",
                (index, row["output"]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
            top.plot(
                [index - 0.34, index + 0.34],
                [row["expected_output"]] * 2,
                linestyle="--",
                color="#555555",
                linewidth=1.5,
                zorder=4,
            )
            for component, offset, hatch in ((0, -0.17, ""), (1, 0.17, "///")):
                position = index + offset
                value = row["input_gradient"][component]
                bottom.bar(
                    position,
                    value,
                    width=0.29,
                    color=COLORS[mode],
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )
                bottom.scatter([position], [value], marker="_", s=90, color=COLORS[mode], zorder=5)
                bottom.annotate(
                    f"{value:.4g}",
                    (position, value),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
                bottom.scatter(
                    [position],
                    [row["finite_difference_input_gradient"][component]],
                    marker="x",
                    s=48,
                    linewidths=1.6,
                    color="black",
                    zorder=6,
                )
                expected = row["expected_input_gradient"][component]
                bottom.plot(
                    [position - 0.14, position + 0.14],
                    [expected] * 2,
                    linestyle="--",
                    color="#555555",
                    linewidth=1.5,
                    zorder=4,
                )
        for axis in (top, bottom):
            axis.set_xticks(range(len(modes)), [labels[m] for m in modes])
            axis.set_xlim(-0.6, len(modes) - 0.4)
            axis.set_axisbelow(True)
            axis.grid(axis="y", alpha=0.2)
        top.set_ylim(*limits(forward_values))
        bottom.set_ylim(*limits(grad_values))
    axes[0, 0].set_ylabel("Forward output y")
    axes[1, 0].set_ylabel("Input gradient dL/dx")
    env = result["environment"]
    title = (
        "Existing API: mask on/off"
        if result["experiment"] == "upstream_mask_toggle"
        else "Three mask policies"
    )
    fig.suptitle(
        f"{title} — {env['device'].upper()}\n"
        "W=[1,1], x=[1,1], ADC bound=1; noise/quantization off; perfect backward",
        fontsize=13,
        y=0.99,
    )
    handles = [Patch(facecolor=COLORS[m], label=m) for m in modes]
    handles += [
        Patch(facecolor="#999999", label="dx1 (left)"),
        Patch(facecolor="#999999", edgecolor="white", hatch="///", label="dx2 (right)"),
        Line2D(
            [], [], color="black", marker="x", linestyle="None", label="Measured finite difference"
        ),
        Line2D([], [], color="#555555", linestyle="--", label="Analytical reference"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.015))
    fig.text(
        0.5,
        0.115,
        "Bars: measured autograd/output values. Zero values are explicitly marked.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.21, wspace=0.22, hspace=0.32)
    return fig


def render_result(result, prefix):
    validated_rows(result)
    prefix = Path(prefix)
    paths = [Path(str(prefix) + suffix) for suffix in (".png", ".pdf")]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    figure = build_figure(result)
    try:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for path in paths:
            with path.open("xb") as stream:
                figure.savefig(stream, format=path.suffix[1:], dpi=180)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = json.loads(args.input.read_text(encoding="utf-8"))
        paths = render_result(result, args.output_prefix)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(1, f"Cannot plot: {exc}\n")
    print("Saved: " + ", ".join(map(str, paths)))


if __name__ == "__main__":
    main()
