"""Plots training runs written by train.py: learning curves and a test-metric comparison.

Plain Python + matplotlib, no torch. Each run folder needs config.json, metrics.csv
and test_metrics.json (copy them from Colab / Drive; checkpoints are not needed).

    python3 plot_runs.py data/measurements/resnet18_001 data/measurements/smallcnn_001 --out docs/figures

Writes <out>/runs_curves.png and <out>/runs_test.png (use --prefix to change the names).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical slots (blue, orange, aqua, yellow), fixed order; text and grid in neutral ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

NAMES = {"resnet18": "ResNet18 (pretrained)", "smallcnn": "Small CNN (from scratch)",
         "mobilenet_v3_small": "MobileNetV3-S (pretrained)", "efficientnet_b0": "EfficientNet-B0 (pretrained)"}


def load_run(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    with open(path / "metrics.csv") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    test = json.loads((path / "test_metrics.json").read_text())
    model = config["args"]["model"]
    pretrained = not config["args"].get("no_pretrained", False) and model != "smallcnn"
    name = NAMES.get(model, model) if pretrained or model == "smallcnn" else f"{model} (random init)"
    return {"name": f"{name} · {path.name}" if len(path.name) > 0 else name, "label": name,
            "rows": rows, "test": test, "freeze": config["args"].get("freeze_epochs", 0) if pretrained else 0}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(runs, out):
    panels = [("val_bal_acc", "Validation balanced accuracy", "{:.3f}"),
              ("train_loss", "Training loss", "{:.2f}"),
              ("val_dist_mae", "Validation distance-ahead error [m]", "{:.2f} m")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    for ax, (key, title, fmt) in zip(axes, panels):
        style(ax)
        ends = []
        for k, run in enumerate(runs):
            ep = [r["epoch"] for r in run["rows"]]
            ys = [r[key] for r in run["rows"]]
            c = SERIES[k]
            ax.plot(ep, ys, color=c, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=run["label"], zorder=3)
            ends.append((ys[-1], ep[-1]))
        # direct label on the last point only; push apart labels that would overlap
        lo, hi = ax.get_ylim()
        gap = 0.06 * (hi - lo)
        placed = []
        for y, x in sorted(ends):
            ty = max(y, placed[-1] + gap) if placed else y
            placed.append(ty)
            ax.annotate(fmt.format(y), (x, y), xytext=(x + 0.25, ty), textcoords="data",
                        va="center", fontsize=9, color=INK)
        if any(r["freeze"] for r in runs):
            ax.axvspan(-0.4, 0.5, color=GRID, alpha=0.5, linewidth=0, zorder=0)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
        ax.set_xlabel("epoch", fontsize=9, color=INK_2)
        ax.set_xlim(-0.4, max(r["rows"][-1]["epoch"] for r in runs) + 1.6)
    if any(r["freeze"] for r in runs):
        axes[0].text(0.05, 0.03, "shaded: backbone frozen (pretrained only)", transform=axes[0].transAxes,
                     fontsize=8, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(runs), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_test(runs, out):
    metrics = [("bal_acc", "Balanced accuracy"), ("recall_L", "Recall L"), ("recall_F", "Recall F"),
               ("recall_R", "Recall R"), ("acc", "Accuracy (tie-aware)")]
    fig, ax = plt.subplots(figsize=(9, 4.4 + 0.15 * len(runs)), facecolor=SURFACE)
    style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    n = len(runs)
    offsets = [(k - (n - 1) / 2) * 0.22 for k in range(n)]
    for k, run in enumerate(runs):
        ys = [i + offsets[k] for i in range(len(metrics))]
        xs = [run["test"][m] for m, _ in metrics]
        ax.scatter(xs, ys, s=70, color=SERIES[k], edgecolor=SURFACE, linewidth=1.5, zorder=3, label=run["label"])
        for x, y in zip(xs, ys):
            ax.annotate(f"{x:.3f}", (x, y), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK)
    ax.set_yticks(range(len(metrics)), [t for _, t in metrics], color=INK, fontsize=10)
    ax.invert_yaxis()
    lo = min(run["test"][m] for run in runs for m, _ in metrics)
    ax.set_xlim(max(0.0, lo - 0.03), 1.02)
    ax.set_xlabel("score on test layouts (higher is better; axis does not start at 0)", fontsize=9, color=INK_2)
    extra = "\n".join(f"{r['label']}: unsafe {r['test']['unsafe']:.4f}, regret {r['test']['regret']:.4f}, "
                           f"distance error {r['test']['dist_mae']:.2f} m, best epoch {r['test']['best_epoch']}"
                           for r in runs)
    ax.set_title("Test metrics (100 held-out layouts, 2,500 images)", loc="left", fontsize=11, color=INK, pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=n, frameon=False, fontsize=9, labelcolor=INK,
              borderaxespad=0.2)
    fig.text(0.01, 0.01, extra, fontsize=8, color=INK_2, va="bottom", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.04 + 0.035 * len(runs), 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders with config.json, metrics.csv, test_metrics.json")
    ap.add_argument("--out", default="runs", help="output folder")
    ap.add_argument("--prefix", default="runs", help="file name prefix")
    a = ap.parse_args()
    if len(a.runs) > len(SERIES):
        ap.error(f"at most {len(SERIES)} runs per figure")
    runs = [load_run(p) for p in a.runs]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_curves(runs, out / f"{a.prefix}_curves.png")
    plot_test(runs, out / f"{a.prefix}_test.png")
    print("wrote", out / f"{a.prefix}_curves.png", "and", out / f"{a.prefix}_test.png")


if __name__ == "__main__":
    main()
