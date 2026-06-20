"""
paper_figures.py — Nature 风格论文配图

输出 (PDF + PNG, 300 DPI):
  figures/fig1_asi_analysis.*
  figures/fig2_main_results.*
  figures/fig3_delta_vs_asi.*
  figures/fig4_training_curves.*
  figures/fig_combined_summary.*  (可选组合图)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Nature-style global rcParams
# ---------------------------------------------------------------------------
NATURE_COLORS = {
    "patches": "#E64B35",      # NPG red
    "inclusion": "#4DBBD5",    # NPG cyan
    "scratches": "#00A087",    # NPG green
    "std": "#8491B4",          # gray-blue
    "conv": "#3C5488",         # navy
    "ours": "#E18727",         # orange-gold
    "accent": "#7E6148",
    "grid": "#ECECEC",
    "text": "#2D2D2D",
}

METHOD_STYLE = {
    "Std-LoRA": {"color": NATURE_COLORS["std"], "marker": "s", "hatch": ""},
    "Conv-LoRA": {"color": NATURE_COLORS["conv"], "marker": "o", "hatch": "//"},
    "ASI-Guided (Ours)": {"color": NATURE_COLORS["ours"], "marker": "D", "hatch": ""},
}

DEFECT_CLASSES = ["patches", "inclusion", "scratches"]
CLASS_LABELS = ["Patches", "Inclusion", "Scratches"]


def setup_nature_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "text.color": NATURE_COLORS["text"],
        "axes.labelcolor": NATURE_COLORS["text"],
        "axes.edgecolor": NATURE_COLORS["text"],
    })


def panel_label(ax, label: str, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")


def save_fig(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_dir / stem}.pdf/png")


def load_training_curves(log_path: Path) -> pd.DataFrame:
    with open(log_path) as f:
        data = json.load(f)
    rows = []
    for r in data["results"]:
        row = {"epoch": r["epoch"], "val_miou": r.get("val_miou", r.get("val_iou", 0))}
        pc = r.get("per_class", {})
        for c in DEFECT_CLASSES:
            if c in pc:
                row[c] = pc[c]
        rows.append(row)
    return pd.DataFrame(rows)


def load_best_per_class(log_path: Path) -> dict:
    df = load_training_curves(log_path)
    if df.empty:
        return {}
    best = df.loc[df["val_miou"].idxmax()]
    return {c: float(best[c]) for c in DEFECT_CLASSES if c in best}


def fig1_asi_analysis(asi_report: Path, out_dir: Path):
    with open(asi_report) as f:
        rep = json.load(f)

    image_asi = rep["image_asi"]
    adapt_gap = rep.get("adaptation_gap", {})
    fin_iou = rep.get("finetuned_iou", {})

    # 183 mm double-column ≈ 7.2 inch
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

    # (a) ASI bars
    ax = axes[0]
    vals = [image_asi[c] for c in DEFECT_CLASSES]
    colors = [NATURE_COLORS[c] for c in DEFECT_CLASSES]
    bars = ax.bar(CLASS_LABELS, vals, color=colors, width=0.62, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Image-level ASI")
    ax.set_ylim(0, 1.0)
    ax.set_title("Adaptation sufficiency")
    ax.grid(axis="y", color=NATURE_COLORS["grid"], linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7)
    panel_label(ax, "a")

    # (b) ASI vs IoU
    ax = axes[1]
    iou_vals = [fin_iou[c] for c in DEFECT_CLASSES]
    for c, a, i, lbl in zip(DEFECT_CLASSES, vals, iou_vals, CLASS_LABELS):
        ax.scatter(a, i, s=70, c=NATURE_COLORS[c], edgecolors="white", linewidths=0.8, zorder=3)
        ax.annotate(lbl, (a, i), textcoords="offset points", xytext=(6, 4), fontsize=7)
    if len(vals) >= 2:
        z = np.polyfit(vals, iou_vals, 1)
        xs = np.linspace(min(vals) - 0.05, max(vals) + 0.05, 50)
        ax.plot(xs, np.poly1d(z)(xs), "--", color="#AAAAAA", linewidth=1.0, zorder=1)
    ax.set_xlabel("Image-level ASI")
    ax.set_ylabel("Validation IoU")
    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(0.65, 0.98)
    ax.set_title("ASI–IoU correlation")
    ax.grid(True, color=NATURE_COLORS["grid"], linewidth=0.5)
    panel_label(ax, "b")

    # (c) Adaptation gap
    ax = axes[2]
    gaps = [adapt_gap.get(c, 0) for c in DEFECT_CLASSES]
    ax.bar(CLASS_LABELS, gaps, color=colors, width=0.62, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color=NATURE_COLORS["text"], linewidth=0.5, alpha=0.5)
    ax.set_ylabel("Linear probe IoU − Finetuned IoU")
    ax.set_title("Adaptation gap")
    ax.grid(axis="y", color=NATURE_COLORS["grid"], linewidth=0.5)
    panel_label(ax, "c")

    fig.suptitle("Class-wise Adaptation Sufficiency Analysis (NEU-Seg)", fontsize=10, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig1_asi_analysis")


def fig2_main_results(exp_dir: Path, out_dir: Path):
    logs = {
        "Std-LoRA": exp_dir / "multiclass_std_30ep" / "training_log.json",
        "Conv-LoRA": exp_dir / "multiclass_conv_30ep" / "training_log.json",
        "ASI-Guided (Ours)": exp_dir / "asi_guided_30ep" / "training_log.json",
    }
    available = {k: v for k, v in logs.items() if v.exists()}
    if not available:
        print("  [skip] fig2 — no training logs")
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    n_methods = len(available)
    n_classes = len(DEFECT_CLASSES)
    x = np.arange(n_classes)
    width = 0.8 / n_methods

    for i, (method, log_path) in enumerate(available.items()):
        pc = load_best_per_class(log_path)
        vals = [pc.get(c, 0) * 100 for c in DEFECT_CLASSES]
        style = METHOD_STYLE.get(method, {"color": "#888888", "hatch": ""})
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, vals, width * 0.92, label=method,
               color=style["color"], edgecolor="white", linewidth=0.6,
               hatch=style.get("hatch", ""))

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_ylabel("IoU (%)")
    ax.set_ylim(60, 100)
    ax.set_title("Per-class segmentation performance")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color=NATURE_COLORS["grid"], linewidth=0.5)
    panel_label(ax, "a", x=-0.14, y=1.06)

    fig.tight_layout()
    save_fig(fig, out_dir, "fig2_main_results")


def fig3_delta_vs_asi(asi_report: Path, exp_dir: Path, out_dir: Path):
    std_log = exp_dir / "multiclass_std_30ep" / "training_log.json"
    conv_log = exp_dir / "multiclass_conv_30ep" / "training_log.json"
    guided_log = exp_dir / "asi_guided_30ep" / "training_log.json"
    if not std_log.exists() or not conv_log.exists():
        print("  [skip] fig3 — need std and conv logs")
        return

    with open(asi_report) as f:
        image_asi = json.load(f)["image_asi"]

    std_pc = load_best_per_class(std_log)
    conv_pc = load_best_per_class(conv_log)
    guided_pc = load_best_per_class(guided_log) if guided_log.exists() else {}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # (a) Delta Conv - Std vs ASI
    ax = axes[0]
    asi_vals, deltas, labels = [], [], []
    for c, lbl in zip(DEFECT_CLASSES, CLASS_LABELS):
        if c in std_pc and c in conv_pc:
            asi_vals.append(image_asi[c])
            deltas.append((conv_pc[c] - std_pc[c]) * 100)
            labels.append(lbl)
            ax.scatter(image_asi[c], (conv_pc[c] - std_pc[c]) * 100,
                       s=80, c=NATURE_COLORS[c], edgecolors="white", linewidths=0.8, zorder=3)
            ax.annotate(lbl, (image_asi[c], (conv_pc[c] - std_pc[c]) * 100),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Image-level ASI")
    ax.set_ylabel("ΔIoU (Conv − Std) (%)")
    ax.set_title("Conv-LoRA gain vs ASI")
    ax.grid(True, color=NATURE_COLORS["grid"], linewidth=0.5)
    panel_label(ax, "a")

    # (b) All methods mIoU
    ax = axes[1]
    methods, mious, colors = [], [], []
    for name, log, col in [
        ("Std-LoRA", std_log, NATURE_COLORS["std"]),
        ("Conv-LoRA", conv_log, NATURE_COLORS["conv"]),
        ("ASI-Guided", guided_log, NATURE_COLORS["ours"]),
    ]:
        if not log.exists():
            continue
        df = load_training_curves(log)
        methods.append(name.replace("ASI-Guided", "Ours"))
        mious.append(df["val_miou"].max() * 100)
        colors.append(col)
    bars = ax.bar(methods, mious, color=colors, width=0.55, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("mIoU (%)")
    ax.set_ylim(min(mious) - 3 if mious else 70, max(mious) + 2 if mious else 90)
    ax.set_title("Overall mIoU (best epoch)")
    ax.grid(axis="y", color=NATURE_COLORS["grid"], linewidth=0.5)
    for bar, v in zip(bars, mious):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")
    panel_label(ax, "b")

    fig.tight_layout()
    save_fig(fig, out_dir, "fig3_delta_vs_asi")


def fig4_training_curves(exp_dir: Path, out_dir: Path):
    logs = {
        "Std-LoRA": (exp_dir / "multiclass_std_30ep" / "training_log.json", NATURE_COLORS["std"]),
        "Conv-LoRA": (exp_dir / "multiclass_conv_30ep" / "training_log.json", NATURE_COLORS["conv"]),
        "ASI-Guided (Ours)": (exp_dir / "asi_guided_30ep" / "training_log.json", NATURE_COLORS["ours"]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    has_any = False

    for name, (path, color) in logs.items():
        if not path.exists():
            continue
        has_any = True
        df = load_training_curves(path)
        axes[0].plot(df["epoch"], df["val_miou"] * 100, label=name, color=color, linewidth=1.4)
        best_ep = df.loc[df["val_miou"].idxmax(), "epoch"]
        best_val = df["val_miou"].max() * 100
        axes[0].scatter([best_ep], [best_val], color=color, s=30, zorder=5, edgecolors="white", linewidths=0.5)

    if not has_any:
        plt.close(fig)
        print("  [skip] fig4 — no logs")
        return

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation mIoU (%)")
    axes[0].set_title("Training convergence")
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[0].grid(True, color=NATURE_COLORS["grid"], linewidth=0.5)
    panel_label(axes[0], "a")

    # Per-class best for Ours vs Conv on patches
    ax = axes[1]
    conv_log = exp_dir / "multiclass_conv_30ep" / "training_log.json"
    guided_log = exp_dir / "asi_guided_30ep" / "training_log.json"
    for name, path, color in [
        ("Conv-LoRA", conv_log, NATURE_COLORS["conv"]),
        ("ASI-Guided (Ours)", guided_log, NATURE_COLORS["ours"]),
    ]:
        if not path.exists():
            continue
        df = load_training_curves(path)
        if "patches" in df.columns:
            ax.plot(df["epoch"], df["patches"] * 100, label=f"{name} (Patches)",
                    color=color, linewidth=1.4, linestyle="-" if "Ours" in name else "--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Patches IoU (%)")
    ax.set_title("Low-ASI class (Patches) learning curve")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(True, color=NATURE_COLORS["grid"], linewidth=0.5)
    panel_label(ax, "b")

    fig.tight_layout()
    save_fig(fig, out_dir, "fig4_training_curves")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", default="outputs/paper_exp")
    p.add_argument("--asi_report", default="outputs/thesis_validation/thesis_validation_report.json")
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    setup_nature_style()
    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.output_dir) if args.output_dir else exp_dir / "figures"
    asi_report = Path(args.asi_report)

    print(f"\n[paper_figures] output → {out_dir}")

    if asi_report.exists():
        fig1_asi_analysis(asi_report, out_dir)
    else:
        print(f"  [skip] fig1 — missing {asi_report}")

    fig2_main_results(exp_dir, out_dir)
    fig3_delta_vs_asi(asi_report, exp_dir, out_dir)
    fig4_training_curves(exp_dir, out_dir)
    print()


if __name__ == "__main__":
    main()
