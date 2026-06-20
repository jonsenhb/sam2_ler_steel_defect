"""
paper_de_report.py — 汇总 paper_de 实验, 生成 Nature 风格图 + 论文表格 (CSV/LaTeX)

输出:
  figures/fig_data_efficiency.(pdf|png)   主图: 各数据集数据效率曲线 (mean±s.d.)
  figures/fig_lowdata_bar.(pdf|png)        低数据(1/5/10%)条形对比
  tables/table_main.(csv|tex)              各方法×标注比例 mIoU(mean±sd)
  tables/table_lowdata.(csv|tex)           低数据汇总 + 相对最优CNN增益
  tables/table_published_sota.(csv|tex)    published SOTA 参照 (带论文出处)
  tables/methods_citations.md              方法-论文出处清单
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from paper_de_pipeline import PAPER_METHODS, PUBLISHED_SOTA

try:
    from paper_external_sota import EXTERNAL_METHODS
except ImportError:
    EXTERNAL_METHODS = {}

ALL_METHODS = {**PAPER_METHODS, **EXTERNAL_METHODS}

# ---- Nature 风格全局设置 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # TrueType (可编辑文字)
    "ps.fonttype": 42,
})

# Wong 色盲友好调色板 (Nature 推荐)
PALETTE = {
    "sam2_lora": "#0072B2",     # blue
    "sam2_linear": "#56B4E9",   # sky blue
    "unet": "#D55E00",          # vermillion
    "deeplabv3plus": "#E69F00", # orange
    "pspnet": "#009E73",        # green
    "segformer": "#CC79A7",     # pink
    "ddsnet_approx": "#882255",      # wine
    "mff_metal_approx": "#332288",   # indigo
    "sme_dlv3p_approx": "#88CCEE",   # cyan
    "hybrid_trans_approx": "#44AA99", # teal
}
MARKERS = {
    "sam2_lora": "o", "sam2_linear": "D", "unet": "s",
    "deeplabv3plus": "^", "pspnet": "v", "segformer": "P",
    "ddsnet_approx": "X", "mff_metal_approx": "d",
    "sme_dlv3p_approx": "<", "hybrid_trans_approx": ">",
}


def load_all(exp_dir: Path):
    """返回 data[dataset][method][frac] = [miou per seed]."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    perclass = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for jf in exp_dir.rglob("*.json"):
        try:
            r = json.load(open(jf))
        except Exception:
            continue
        if "miou_global" not in r:
            continue
        data[r["dataset"]][r["method"]][r["frac"]].append(r["miou_global"])
        if r.get("per_class"):
            perclass[r["dataset"]][r["method"]][r["frac"]].append(r["per_class"])
    return data, perclass


def agg(vals):
    a = np.array(vals, dtype=float)
    return (float(a.mean()) if len(a) else float("nan"),
            float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def fig_data_efficiency(data, out_dir, fracs):
    datasets = [d for d in ["neu_seg", "severstal"] if d in data]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.8 * len(datasets), 3.2), squeeze=False)
    titles = {"neu_seg": "NEU-Seg", "severstal": "Severstal"}
    for ax, ds in zip(axes[0], datasets):
        for method in ALL_METHODS:
            if method not in data[ds]:
                continue
            xs, ys, es = [], [], []
            for f in fracs:
                if f in data[ds][method] and data[ds][method][f]:
                    m, s = agg(data[ds][method][f])
                    xs.append(f * 100); ys.append(m * 100); es.append(s * 100)
            if not xs:
                continue
            is_ours = method.startswith("sam2")
            is_ext = method in EXTERNAL_METHODS
            color = PALETTE.get(method, "#999999")
            marker = MARKERS.get(method, "x")
            lbl = ALL_METHODS[method]["label"]
            ax.errorbar(xs, ys, yerr=es, marker=marker, ms=4,
                        lw=2.0 if is_ours else (1.5 if is_ext else 1.2),
                        capsize=2.5, color=color, label=lbl,
                        zorder=5 if is_ours else (4 if is_ext else 3),
                        linestyle="-" if is_ours else ("-." if is_ext else "--"))
        ax.set_xscale("log")
        ax.set_xlabel("Training labels (%)")
        ax.set_ylabel("Val mIoU (%)")
        ax.set_title(titles.get(ds, ds), fontweight="bold")
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.set_xticks([1, 5, 10, 25, 100])
        ax.set_xticklabels(["1", "5", "10", "25", "100"])
    for i, ax in enumerate(axes[0]):
        ax.text(-0.12, 1.05, chr(97 + i), transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")
    axes[0][0].legend(loc="lower right", ncol=1, fontsize=5.5, framealpha=0.92)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_data_efficiency.{ext}")
    plt.close(fig)
    print(f"  图: {out_dir/'fig_data_efficiency.pdf'}")


def fig_lowdata_bar(data, out_dir, low_fracs=(0.01, 0.05, 0.10)):
    datasets = [d for d in ["neu_seg", "severstal"] if d in data]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.2), squeeze=False)
    titles = {"neu_seg": "NEU-Seg", "severstal": "Severstal"}
    all_m = [m for m in ALL_METHODS if any(m in data[ds] for ds in datasets)]
    for ax, ds in zip(axes[0], datasets):
        present = [m for m in all_m if m in data[ds]]
        x = np.arange(len(low_fracs)); w = 0.8 / max(1, len(present))
        for k, method in enumerate(present):
            ys, es = [], []
            for f in low_fracs:
                m, s = agg(data[ds][method].get(f, [])) if data[ds][method].get(f) else (np.nan, 0)
                ys.append(m * 100 if not np.isnan(m) else 0); es.append(s * 100)
            color = PALETTE.get(method, "#999999")
            ax.bar(x + k * w, ys, w, yerr=es, capsize=2, color=color,
                   label=ALL_METHODS[method]["label"], edgecolor="white", linewidth=0.4)
        ax.set_xticks(x + w * (len(present) - 1) / 2)
        ax.set_xticklabels([f"{int(f*100)}%" for f in low_fracs])
        ax.set_xlabel("Training labels"); ax.set_ylabel("Val mIoU (%)")
        ax.set_title(titles.get(ds, ds), fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    for i, ax in enumerate(axes[0]):
        ax.text(-0.12, 1.05, chr(97 + i), transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")
    axes[0][-1].legend(loc="upper left", ncol=1, fontsize=5.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_lowdata_bar.{ext}")
    plt.close(fig)
    print(f"  图: {out_dir/'fig_lowdata_bar.pdf'}")


def table_main(data, out_dir, fracs):
    for ds in data:
        # find best mIoU per frac to bold
        best_per_frac = {}
        for f in fracs:
            bval = -1.0
            for method in ALL_METHODS:
                if method in data[ds] and data[ds][method].get(f):
                    m, _ = agg(data[ds][method][f])
                    if m > bval:
                        bval = m
            best_per_frac[f] = bval

        rows_ours, rows_base, rows_ext = [], [], []
        for method in ALL_METHODS:
            if method not in data[ds]:
                continue
            cells = {}; cells_tex = {}
            for f in fracs:
                if data[ds][method].get(f):
                    m, s = agg(data[ds][method][f])
                    txt = f"{m*100:.2f}±{s*100:.2f}"
                    cells[f] = txt
                    is_best = abs(m - best_per_frac[f]) < 1e-6
                    cells_tex[f] = f"\\textbf{{{txt}}}" if is_best else txt
                else:
                    cells[f] = "-"; cells_tex[f] = "-"
            lbl = ALL_METHODS[method]["label"]
            row = (lbl, cells, cells_tex)
            if method.startswith("sam2"):
                rows_ours.append(row)
            elif method in EXTERNAL_METHODS:
                rows_ext.append(row)
            else:
                rows_base.append(row)

        all_rows = rows_ours + rows_base + rows_ext
        hdr = ["Method"] + [f"{int(f*100)}\\%" for f in fracs]
        hdr_csv = ["Method"] + [f"{int(f*100)}%" for f in fracs]

        with open(out_dir / f"table_main_{ds}.csv", "w") as fcsv:
            fcsv.write(",".join(hdr_csv) + "\n")
            for lbl, cells, _ in all_rows:
                fcsv.write(",".join([lbl] + [cells[f] for f in fracs]) + "\n")

        with open(out_dir / f"table_main_{ds}.tex", "w") as ftex:
            ftex.write("\\begin{tabular}{l" + "c" * len(fracs) + "}\n\\toprule\n")
            ftex.write(" & ".join(hdr) + " \\\\\n\\midrule\n")
            for i, (lbl, _, ct) in enumerate(all_rows):
                ftex.write(" & ".join([lbl] + [ct[f] for f in fracs]) + " \\\\\n")
                if i == len(rows_ours) - 1 or i == len(rows_ours) + len(rows_base) - 1:
                    ftex.write("\\midrule\n")
            ftex.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  表: {out_dir}/table_main_*.csv|tex")


def table_perclass(data, perclass, out_dir):
    """Per-class IoU table at 100% data for each dataset."""
    for ds in perclass:
        rows = []
        for method in ALL_METHODS:
            if method not in perclass[ds]:
                continue
            pc_list = perclass[ds][method].get(1.0, [])
            if not pc_list:
                continue
            all_classes = sorted({k for d in pc_list for k in d})
            avg = {}
            for c in all_classes:
                vals = [d[c] for d in pc_list if c in d]
                avg[c] = np.mean(vals) * 100 if vals else 0.0
            rows.append((ALL_METHODS[method]["label"], avg, all_classes))
        if not rows:
            continue
        classes = rows[0][2]
        hdr = ["Method"] + list(classes) + ["mIoU"]
        with open(out_dir / f"table_perclass_{ds}.csv", "w") as f:
            f.write(",".join(hdr) + "\n")
            for lbl, avg, _ in rows:
                miou = np.mean([avg[c] for c in classes])
                f.write(",".join([lbl] + [f"{avg[c]:.2f}" for c in classes] + [f"{miou:.2f}"]) + "\n")
        with open(out_dir / f"table_perclass_{ds}.tex", "w") as f:
            f.write("\\begin{tabular}{l" + "c" * (len(classes) + 1) + "}\n\\toprule\n")
            f.write(" & ".join(hdr) + " \\\\\n\\midrule\n")
            for lbl, avg, _ in rows:
                miou = np.mean([avg[c] for c in classes])
                f.write(" & ".join([lbl] + [f"{avg[c]:.1f}" for c in classes] + [f"{miou:.1f}"]) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  表: {out_dir}/table_perclass_*.csv|tex")


def table_significance(data, out_dir, fracs):
    """Statistical significance table: SAM2+LoRA vs every other method."""
    from scipy import stats as sp
    ref = "sam2_lora"
    for ds in data:
        if ref not in data[ds]:
            continue
        rows = []
        for method in ALL_METHODS:
            if method == ref or method not in data[ds]:
                continue
            cells = {}
            for f in fracs:
                ref_v = data[ds][ref].get(f, [])
                cmp_v = data[ds][method].get(f, [])
                if len(ref_v) >= 3 and len(cmp_v) >= 3:
                    delta = np.mean(ref_v) - np.mean(cmp_v)
                    _, p = sp.ttest_ind(ref_v, cmp_v)
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    cells[f] = f"+{delta*100:.1f} ({sig})"
                else:
                    cells[f] = "-"
            rows.append((ALL_METHODS[method]["label"], cells))
        hdr = ["Method (vs SAM2+LoRA)"] + [f"{int(f*100)}%" for f in fracs]
        with open(out_dir / f"table_significance_{ds}.csv", "w") as fcsv:
            fcsv.write(",".join(hdr) + "\n")
            for lbl, cells in rows:
                fcsv.write(",".join([lbl] + [cells[f] for f in fracs]) + "\n")
    print(f"  表: {out_dir}/table_significance_*.csv")


def table_published(out_dir):
    for ds, items in PUBLISHED_SOTA.items():
        with open(out_dir / f"table_published_sota_{ds}.csv", "w") as f:
            f.write("Method,mIoU(%),Source/Paper\n")
            for it in items:
                f.write(f"{it['method']},{it['miou']:.2f},\"{it['paper']}\"\n")
    print(f"  表: {out_dir}/table_published_sota_*.csv (参照, 原协议)")


def methods_citations(out_dir):
    with open(out_dir / "methods_citations.md", "w") as f:
        f.write("# 对比方法与论文出处\n\n## Ours + Basic Baselines\n\n| 方法 | 出处 |\n|---|---|\n")
        for m, spec in PAPER_METHODS.items():
            f.write(f"| {spec['label']} | {spec['paper']} |\n")
        if EXTERNAL_METHODS:
            f.write("\n## External SOTA (reproduced under our protocol)\n\n| 方法 | 出处 |\n|---|---|\n")
            for m, spec in EXTERNAL_METHODS.items():
                f.write(f"| {spec['label']} | {spec['paper']} |\n")
        f.write("\n## 数据集\n")
        f.write("- NEU-Seg: Song & Yan, Applied Surface Science 2013 (NEU surface defect database)\n")
        f.write("- Severstal: Severstal Steel Defect Detection, Kaggle 2019\n")
    print(f"  引用清单: {out_dir/'methods_citations.md'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", default="outputs/paper_de")
    p.add_argument("--fracs", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.25, 1.0])
    args = p.parse_args()
    exp_dir = Path(args.exp_dir)
    fig_dir = exp_dir / "figures"; tab_dir = exp_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)

    data, perclass = load_all(exp_dir)
    if not data:
        print(f"未找到结果: {exp_dir}"); return

    # 进度概览
    print("已完成实验统计 (method × frac × #seeds):")
    for ds in data:
        print(f"  [{ds}]")
        for m in PAPER_METHODS:
            if m in data[ds]:
                cov = {int(f*100): len(data[ds][m][f]) for f in sorted(data[ds][m])}
                print(f"    {PAPER_METHODS[m]['label']:32s} {cov}")

    fig_data_efficiency(data, fig_dir, args.fracs)
    fig_lowdata_bar(data, fig_dir)
    table_main(data, tab_dir, args.fracs)
    table_perclass(data, perclass, tab_dir)
    table_significance(data, tab_dir, args.fracs)
    table_published(tab_dir)
    methods_citations(tab_dir)
    print("\n✅ 图表生成完成.")


if __name__ == "__main__":
    main()
