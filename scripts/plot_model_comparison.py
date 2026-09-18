"""Renders comparison charts from `reports/model_comparison_results.json`
(written by `scripts/run_model_comparison.py`) into `reports/figures/`.

Follows the project's dataviz conventions: fixed categorical color per model
(same hue everywhere a model appears, never re-colored by rank), one axis per
chart (accuracy/F1 share a 0-1 scale; distance and latency get their own
charts since they're different units), recessive hairline gridlines, direct
value labels instead of a cluttered legend, and a single-hue sequential ramp
for the confusion-matrix heatmaps.

Usage:
    python scripts/plot_model_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

from slouchfix import config

RESULTS_JSON = config.PROJECT_ROOT / "reports" / "model_comparison_results.json"
FIGURES_DIR = config.PROJECT_ROOT / "reports" / "figures"

# --- Palette (validated categorical slots 1-3: blue/orange/aqua clear every
# CVD + contrast gate all-pairs, in both light and dark) -------------------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# Fixed model -> color assignment, held constant across every chart so a
# reader can track "blue = XGBoost" across the whole report.
MODEL_COLORS = {
    "XGBoost": "#2a78d6",
    "MLP (PyTorch)": "#eb6834",
    "Random Forest": "#1baf7a",
}
MODEL_ORDER = ["XGBoost", "MLP (PyTorch)", "Random Forest"]

# Sequential blue ramp (palette.md step 150 -> 650) for the confusion-matrix heatmaps.
SEQ_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"]
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _rounded_bar(ax, x, height, width, color, baseline=0.0):
    """A bar with a small rounded cap at the data end, square at the baseline."""
    y0, y1 = (baseline, height) if height >= baseline else (height, baseline)
    bar_h = max(y1 - y0, 1e-9)
    rounding = min(width, bar_h) * 0.12
    rect = FancyBboxPatch(
        (x - width / 2, y0),
        width,
        bar_h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=0,
        facecolor=color,
        mutation_aspect=1,
    )
    ax.add_patch(rect)


def _style_axes(ax, y_grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)
    if y_grid:
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
        ax.set_axisbelow(True)


def plot_accuracy_f1(results: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    x = np.arange(len(MODEL_ORDER))
    bar_w = 0.32
    gap = 0.04

    acc = [results[m]["classification"]["accuracy"] for m in MODEL_ORDER]
    f1 = [results[m]["classification"]["macro_f1"] for m in MODEL_ORDER]

    acc_color, f1_color = "#2a78d6", "#eb6834"
    acc_x = x - (bar_w + gap) / 2
    f1_x = x + (bar_w + gap) / 2

    for xi, v in zip(acc_x, acc):
        _rounded_bar(ax, xi, v, bar_w, acc_color)
    for xi, v in zip(f1_x, f1):
        _rounded_bar(ax, xi, v, bar_w, f1_color)

    for xi, v in zip(acc_x, acc):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=9, color=INK_SECONDARY)
    for xi, v in zip(f1_x, f1):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=9, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=10)
    ax.set_xlim(x[0] - 0.65, x[-1] + 0.65)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (test set)")
    ax.set_title("Accuracy vs. Macro F1 by model", fontsize=13, color=INK_PRIMARY, pad=14)
    _style_axes(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, color=acc_color), plt.Rectangle((0, 0), 1, 1, color=f1_color)]
    ax.legend(handles, ["Accuracy", "Macro F1"], frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10)

    fig.tight_layout()
    out = FIGURES_DIR / "comparison_accuracy_f1.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_single_metric_bar(results: dict, metric_path: tuple[str, str], title: str, ylabel: str,
                            fmt: str, filename: str, log_scale: bool = False,
                            extra_label_fn=None) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    x = np.arange(len(MODEL_ORDER))
    bar_w = 0.5

    section, key = metric_path
    values = [results[m][section][key] for m in MODEL_ORDER]
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]

    baseline = min(values) * 0.5 if log_scale else 0.0
    for xi, v, c in zip(x, values, colors):
        _rounded_bar(ax, xi, v, bar_w, c, baseline=baseline if log_scale else 0.0)

    if log_scale:
        ax.set_yscale("log")

    top = max(values)
    for xi, v, m in zip(x, values, MODEL_ORDER):
        label = fmt.format(v)
        if extra_label_fn:
            label += extra_label_fn(results[m])
        label_y = v * 1.3 if log_scale else v + top * 0.02
        ax.text(xi, label_y, label, ha="center", va="bottom", fontsize=9.5, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=10)
    ax.set_xlim(x[0] - 0.65, x[-1] + 0.65)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, color=INK_PRIMARY, pad=14)
    if log_scale:
        ax.set_ylim(baseline, top * 6)
    else:
        ax.set_ylim(0, top * 1.22)
    _style_axes(ax)

    fig.tight_layout()
    out = FIGURES_DIR / filename
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_confusion_matrices(results: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    for ax, model_name in zip(axes, MODEL_ORDER):
        cls = results[model_name]["classification"]
        labels = cls["labels_order"]
        cm = np.array(cls["confusion_matrix"])
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap=SEQ_BLUE_CMAP, vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        short = [l.replace("_", "\n") for l in labels]
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8, color=INK_SECONDARY)
        ax.set_yticklabels(short, fontsize=8, color=INK_SECONDARY)
        ax.set_title(model_name, fontsize=11, color=INK_PRIMARY, pad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(len(labels)):
            for j in range(len(labels)):
                frac = cm_norm[i, j]
                text_color = "#ffffff" if frac > 0.5 else INK_PRIMARY
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=7.5, color=text_color)

    fig.suptitle("Confusion matrices - test set (row-normalized color, raw counts labeled)",
                  fontsize=13, color=INK_PRIMARY, y=1.02)
    cbar = fig.colorbar(axes[-1].images[0], ax=axes, fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label("Share of true class", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(color=INK_MUTED, labelcolor=INK_SECONDARY)

    out = FIGURES_DIR / "confusion_matrices.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_speed_vs_accuracy(results: dict) -> Path:
    """Scatter: does the latency cost buy you accuracy? (it doesn't, here)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    for m in MODEL_ORDER:
        r = results[m]
        x = r["latency"]["latency_ms"]
        y = r["classification"]["macro_f1"]
        ax.scatter([x], [y], s=140, color=MODEL_COLORS[m], zorder=3,
                    edgecolors=SURFACE, linewidths=2)
        ax.annotate(f"  {m}\n  {x:.2f} ms, F1={y:.3f}", (x, y), fontsize=9,
                     color=INK_SECONDARY, va="center")

    ax.set_xscale("log")
    ax.set_xlabel("Latency per frame, ms (log scale) - lower is better")
    ax.set_ylabel("Macro F1 (test set) - higher is better")
    ax.set_title("Accuracy bought per millisecond of latency", fontsize=13, color=INK_PRIMARY, pad=14)
    _style_axes(ax)
    ax.set_xlim(0.15, 300)
    ax.set_ylim(0.85, 0.92)

    fig.tight_layout()
    out = FIGURES_DIR / "comparison_speed_vs_accuracy.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(f"{RESULTS_JSON} not found - run scripts/run_model_comparison.py first.")
    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    results = payload["models"]
    missing = [m for m in MODEL_ORDER if m not in results]
    if missing:
        raise ValueError(f"Missing results for {missing} in {RESULTS_JSON}; run those models first.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    written = [
        plot_accuracy_f1(results),
        plot_single_metric_bar(
            results, ("distance", "mae_cm"), "Distance MAE by model", "Mean absolute error (cm)",
            "{:.1f} cm", "comparison_distance_mae.png",
        ),
        plot_single_metric_bar(
            results, ("latency", "latency_ms"), "Inference latency by model (log scale)",
            "Latency per frame, ms (log)", "{:.2f} ms",
            "comparison_latency.png", log_scale=True,
            extra_label_fn=lambda r: f"\n({r['latency']['fps']:.0f} FPS)",
        ),
        plot_confusion_matrices(results),
        plot_speed_vs_accuracy(results),
    ]

    print(f"Dataset: {payload['dataset']['n_people']} people / {payload['dataset']['n_rows']} rows")
    print("Wrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
