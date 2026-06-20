"""
paper_sensors_figures.py — Sensors SAM2-LER manuscript figures (Fig.3–9, S4).

Fig.1–2 via figure1_rad.py / figure2_arch.py. Fig.6/S1/S3 via render_qualitative.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent

COLORS = {
    "sam2_lora": "#0072B2",
    "sam2_linear": "#56B4E9",
    "unet": "#D55E00",
    "unet_imagenet": "#CC6677",
    "deeplabv3plus": "#E69F00",
    "pspnet": "#009E73",
    "segformer": "#CC79A7",
    "positive": "#0072B2",
    "negative": "#D55E00",
    "neutral": "#999999",
    "patches": "#E64B35",
    "inclusion": "#4DBBD5",
    "scratches": "#00A087",
    "text": "#2D2D2D",
    "grid": "#ECECEC",
}

METHOD_PARAMS_M = {
    "SAM2-LER (Ours)": 0.72,
    "SAM2(frozen)+Linear (Ours)": 0.0,
    "U-Net": 7.8,
    "U-Net (ImageNet)": 7.8,
    "DeepLabV3+": 15.3,
    "PSPNet": 13.6,
    "SegFormer": 3.7,
}


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  saved {out_dir / stem}.pdf")


def panel_label(ax, label: str):
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def fig03_fig04_from_paper_de(exp_dir: Path, out_dir: Path):
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from paper_de_report import load_all, fig_data_efficiency, fig_lowdata_bar

    data, _ = load_all(exp_dir)
    fracs = [0.01, 0.05, 0.10, 0.25, 1.0]
    fig_data_efficiency(data, out_dir, fracs)
    fig_lowdata_bar(data, out_dir)
    for old, new in [
        ("fig_data_efficiency", "fig03_label_efficiency_curves"),
        ("fig_lowdata_bar", "fig04_lowdata_bar"),
    ]:
        for ext in ("pdf", "png"):
            src = out_dir / f"{old}.{ext}"
            dst = out_dir / f"{new}.{ext}"
            if src.exists():
                src.rename(dst)


def fig05_pareto(csv_path: Path, out_dir: Path):
    rows = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["Method"]] = row

    methods = [
        "SAM2(frozen)+Linear (Ours)",
        "SAM2-LER (Ours)",
        "U-Net",
        "SegFormer",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, col, title in zip(
        axes,
        ["1%", "10%"],
        ["@1% labels (NEU-Seg)", "@10% labels (NEU-Seg)"],
    ):
        xs, ys, cs, labels = [], [], [], []
        for m in methods:
            if m not in rows:
                continue
            val = float(rows[m][col].split("±")[0])
            params = METHOD_PARAMS_M.get(m, 5.0)
            xs.append(max(params, 0.05))
            ys.append(val)
            cs.append(COLORS["sam2_lora"] if "SAM2-LER" in m or "LoRA" in m else COLORS["unet"] if "U-Net" in m else COLORS["sam2_linear"])
            labels.append("SAM2-LER" if "SAM2-LER" in m or "LoRA" in m else m.split("(")[0].strip()[:12])
        ax.scatter(xs, ys, c=cs, s=80, edgecolors="white", linewidths=0.8, zorder=3)
        for x, y, lb in zip(xs, ys, labels):
            ax.annotate(lb, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=6.5)
        ax.set_xscale("log")
        ax.set_xlabel("Trainable params (M)")
        ax.set_ylabel("mIoU (%)")
        ax.set_title(title)
        ax.grid(True, color=COLORS["grid"], linewidth=0.5)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.tight_layout()
    save_fig(fig, out_dir, "fig05_pareto_params")


def _load_perclass(exp_dir: Path, method_key: str, frac: float):
    """Aggregate per-class IoU mean±std over seeds (skip runs with empty per_class)."""
    classes = ["patches", "inclusion", "scratches"]
    vals = {c: [] for c in classes}
    tag = f"f{int(frac * 1000):04d}"
    for jf in (exp_dir / "neu_seg").glob(f"{method_key}_{tag}_s*.json"):
        d = json.load(open(jf))
        pc = d.get("per_class") or {}
        if not all(c in pc for c in classes):
            continue
        for c in classes:
            vals[c].append(pc[c] * 100)
    return {
        c: (float(np.mean(vals[c])), float(np.std(vals[c])) if len(vals[c]) > 1 else 0.0)
        if vals[c]
        else (float("nan"), 0.0)
        for c in classes
    }


def fig07_perclass_heatmap(exp_dir: Path, out_dir: Path):
    methods = [
        ("sam2_lora", "SAM2-LER"),
        ("sam2_linear", "Linear"),
        ("unet", "U-Net"),
        ("segformer", "SegFormer"),
    ]
    classes = ["patches", "inclusion", "scratches"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6))
    im = None
    for ax, frac, plab in zip(axes, [0.01, 0.10], ["a", "b"]):
        mat = np.zeros((len(methods), len(classes)))
        for i, (mk, _) in enumerate(methods):
            pc = _load_perclass(exp_dir, mk, frac)
            for j, c in enumerate(classes):
                mat[i, j] = pc[c][0]
        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=80)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(["Patches", "Inclusion", "Scratches"])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([m[1] for m in methods])
        ax.set_title(f"Per-class IoU @ {int(frac*100)}% labels")
        for i in range(len(methods)):
            for j in range(len(classes)):
                val = mat[i, j]
                txt = "—" if np.isnan(val) else f"{val:.0f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")
        panel_label(ax, plab)
    fig.subplots_adjust(right=0.86, wspace=0.35)
    cax = fig.add_axes([0.88, 0.16, 0.018, 0.68])
    fig.colorbar(im, cax=cax, label="IoU (%)")
    save_fig(fig, out_dir, "fig07_perclass_heatmap")


def fig08_asi(asi_path: Path, out_dir: Path):
    if not asi_path.exists():
        print(f"  skip fig08 — missing {asi_path}")
        return
    with open(asi_path) as f:
        rep = json.load(f)
    classes = ["patches", "inclusion", "scratches"]
    labels = ["Patches", "Inclusion", "Scratches"]
    image_asi = rep["image_asi"]
    fin_iou = rep.get("finetuned_iou", {})
    per_image = rep.get("per_image", [])

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    vals = [image_asi[c] for c in classes]
    colors = [COLORS[c] for c in classes]

    ax = axes[0]
    ax.bar(labels, vals, color=colors, width=0.62, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Class-level ASI")
    ax.set_ylim(0, 1.05)
    ax.set_title("Adaptation sufficiency")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    panel_label(ax, "a")

    ax = axes[1]
    if per_image:
        for pt in per_image:
            c = pt.get("class_name", "patches")
            ax.scatter(pt["separability"], pt["iou"], s=12, c=COLORS.get(c, COLORS["patches"]),
                       alpha=0.55, edgecolors="none", zorder=3)
        ax.set_xlabel("Per-image separability")
        ax.set_ylabel("Validation IoU (100\% labels)")
        ax.set_title("Separability–IoU (per image)")
    else:
        iou_vals = [fin_iou.get(c, 0) for c in classes]
        for c, a, i, lbl in zip(classes, vals, iou_vals, labels):
            ax.scatter(a, i, s=70, c=COLORS[c], edgecolors="white", linewidths=0.8, zorder=3)
            ax.annotate(lbl, (a, i), textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax.set_xlabel("Class-level ASI")
        ax.set_ylabel("Validation IoU (100\% labels)")
        ax.set_title("ASI–IoU (class-level)")
    ax.grid(True, color=COLORS["grid"], linewidth=0.5)
    panel_label(ax, "b")

    ax = axes[2]
    gaps = [100.0 * rep.get("adaptation_gap", {}).get(c, 0) for c in classes]
    ax.bar(labels, gaps, color=colors, width=0.62, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color=COLORS["text"], linewidth=0.5, alpha=0.5)
    ax.set_ylabel("Adaptation gap (pp)")
    ax.set_title("Linear vs finetuned")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    panel_label(ax, "c")

    fig.tight_layout()
    save_fig(fig, out_dir, "fig08_asi_diagnostic")


def fig09_ablation_forest(root: Path, out_dir: Path):
    entries = []
    base = 54.68

    def add(name, path):
        if not path.exists():
            return
        r = json.load(open(path))
        if isinstance(r, dict) and "rows" in r:
            for row in r["rows"]:
                if row.get("frac", 0.01) == 0.01:
                    entries.append((row.get("probe", name), row.get("delta_miou", 0)))
                    return
        v = r.get("miou_global", 0)
        if v <= 1:
            v *= 100
        entries.append((name, v - base))

    add("CB-PEFT", root / "outputs/probe_ab/cb_peft/probe_cb_summary.json")
    add("DSA-FPN", root / "outputs/probe_ab/dsa_fpn/probe_dsa_summary.json")
    rad_path = root / "outputs/rad_probes/probe_rad_summary.json"
    if rad_path.exists():
        for row in json.load(open(rad_path)).get("rows", []):
            if row.get("frac", 0.01) == 0.01:
                entries.append((row["probe"].title() + " loss", row.get("delta_miou", 0)))
    ds_vals = []
    for jf in (root / "outputs/paper_defectsam2/neu_seg").glob("defectsam2_f0010_s*.json"):
        ds_vals.append(json.load(open(jf))["miou_global"] * 100)
    if ds_vals:
        entries.append(("DefectSAM2 (Supp.)", float(np.mean(ds_vals)) - base))

    if not entries:
        print("  skip fig09 — no ablation data")
        return

    names, deltas = zip(*sorted(entries, key=lambda x: x[1]))
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    colors = [COLORS["positive"] if d >= 0 else COLORS["negative"] for d in deltas]
    ax.barh(names, deltas, color=colors, height=0.55, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=COLORS["text"], lw=0.8)
    ax.set_xlabel("Δ mIoU @1% vs SAM2-LER (pp)")
    ax.set_title("Ablation under 1% labels (NEU-Seg)")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig09_ablation_forest")


def figS4_significance(csv_path: Path, out_dir: Path):
    supp = out_dir.parent / "supplementary" / "figures"
    methods, fracs, mat = [], ["1%", "5%", "10%", "25%", "100%"], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = row.get("Method (vs SAM2-LER)") or row.get("Method (vs SAM2+LoRA)")
            if not key:
                continue
            methods.append(key)
            mat.append([float(row[c].split()[0]) for c in fracs])
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu", vmin=-5, vmax=35)
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels(fracs)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m[:22] for m in methods], fontsize=6)
    ax.set_xlabel("Label fraction")
    ax.set_title("SAM2-LER advantage (pp mIoU)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="pp")
    fig.tight_layout()
    save_fig(fig, supp, "figS4_significance_heatmap")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="release/manuscript_sensors/figures")
    p.add_argument("--exp_dir", default="outputs/paper_de")
    p.add_argument("--asi_report", default="outputs/thesis_validation/thesis_validation_report.json")
    p.add_argument("--root", default=str(ROOT))
    args = p.parse_args()

    setup_style()
    out = Path(args.out_dir)
    root = Path(args.root)
    exp = Path(args.exp_dir)
    tab = Path("release/sam2_ler_github/results/paper_de/tables/table_main_neu_seg.csv")
    sig = Path("release/manuscript_sensors/tables/table3_significance_neu.csv")

    print(f"[paper_sensors_figures] -> {out}")
    fig03_fig04_from_paper_de(exp, out)
    if tab.exists():
        fig05_pareto(tab, out)
    fig07_perclass_heatmap(exp, out)
    fig08_asi(Path(args.asi_report), out)
    fig09_ablation_forest(root, out)
    if sig.exists():
        figS4_significance(sig, out)
    print("Done.")


if __name__ == "__main__":
    main()
