"""
paper_defectsam2_report.py — 汇总 DefectSAM2 vs SAM2+LoRA baseline
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_runs(root: Path, method: str):
    data = defaultdict(lambda: defaultdict(list))
    perclass = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if not root.exists():
        return data, perclass
    for jf in root.rglob("*.json"):
        try:
            r = json.load(open(jf))
        except Exception:
            continue
        if "miou_global" not in r:
            continue
        if r.get("method") != method:
            continue
        ds = r["dataset"]
        frac = r["frac"]
        data[ds][frac].append(r["miou_global"])
        if r.get("per_class"):
            for k, v in r["per_class"].items():
                perclass[ds][frac][k].append(float(v))
    return data, perclass


def agg(vals):
    a = np.array(vals, dtype=float)
    if len(a) == 0:
        return float("nan"), 0.0, 0
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, len(a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--defectsam2_dir", default="outputs/paper_defectsam2")
    p.add_argument("--baseline_dir", default="outputs/paper_de")
    p.add_argument("--out_dir", default="outputs/paper_defectsam2")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fracs = [0.01, 0.05, 0.10, 0.25, 1.0]

    d_new, pc_new = load_runs(Path(args.defectsam2_dir), "defectsam2")
    d_base, pc_base = load_runs(Path(args.baseline_dir), "sam2_lora")

    lines = [
        "# DefectSAM2 vs SAM2+LoRA+FPN\n\n",
        "## Global mIoU\n\n",
        "| Dataset | Frac | DefectSAM2 mIoU | Baseline mIoU | Δ (pp) | n |\n",
        "|---------|------|-----------------|---------------|--------|---|\n",
    ]

    for ds in ["neu_seg", "severstal"]:
        ds_label = "NEU-Seg" if ds == "neu_seg" else "Severstal"
        for frac in fracs:
            mn, sn, nn = agg(d_new.get(ds, {}).get(frac, []))
            mb, sb, nb = agg(d_base.get(ds, {}).get(frac, []))
            delta = (mn - mb) * 100 if not (np.isnan(mn) or np.isnan(mb)) else float("nan")
            lines.append(
                f"| {ds_label} | {frac*100:.0f}% | {mn*100:.2f}±{sn*100:.2f} | "
                f"{mb*100:.2f}±{sb*100:.2f} | {delta:+.2f} | {nn}/{nb} |\n"
            )

    lines.append("\n## Per-class IoU (NEU-Seg)\n\n")
    lines.append("| Frac | Class | DefectSAM2 | Baseline | Δ (pp) |\n")
    lines.append("|------|-------|------------|----------|--------|\n")
    for frac in fracs:
        for cls in ["patches", "inclusion", "scratches"]:
            mn, sn, _ = agg(pc_new.get("neu_seg", {}).get(frac, {}).get(cls, []))
            mb, sb, _ = agg(pc_base.get("neu_seg", {}).get(frac, {}).get(cls, []))
            delta = (mn - mb) * 100 if not (np.isnan(mn) or np.isnan(mb)) else float("nan")
            lines.append(
                f"| {frac*100:.0f}% | {cls} | {mn*100:.2f}±{sn*100:.2f} | "
                f"{mb*100:.2f}±{sb*100:.2f} | {delta:+.2f} |\n"
            )

    # GO check @1% NEU
    mn_1, _, _ = agg(d_new.get("neu_seg", {}).get(0.01, []))
    mb_1, _, _ = agg(d_base.get("neu_seg", {}).get(0.01, []))
    mp_1, _, _ = agg(pc_new.get("neu_seg", {}).get(0.01, {}).get("patches", []))
    bp_1, _, _ = agg(pc_base.get("neu_seg", {}).get(0.01, {}).get("patches", []))
    go_global = (mn_1 - mb_1) * 100 >= 3.0 if not (np.isnan(mn_1) or np.isnan(mb_1)) else False
    go_patches = (mp_1 - bp_1) * 100 >= 5.0 if not (np.isnan(mp_1) or np.isnan(bp_1)) else False
    go_str = "GO" if go_global and go_patches else "NO-GO"

    lines.append("\n## GO Criteria @1% NEU-Seg\n\n")
    lines.append(f"- Global mIoU: DefectSAM2 {mn_1*100:.2f}% vs baseline {mb_1*100:.2f}% "
                 f"(Δ {(mn_1-mb_1)*100:+.2f} pp, need ≥+3 pp) → {'PASS' if go_global else 'FAIL'}\n")
    lines.append(f"- Patches IoU: DefectSAM2 {mp_1*100:.2f}% vs baseline {bp_1*100:.2f}% "
                 f"(Δ {(mp_1-bp_1)*100:+.2f} pp, need ≥+5 pp) → {'PASS' if go_patches else 'FAIL'}\n")
    lines.append(f"\n**Overall: {go_str}**\n")

    report = "".join(lines)
    rpath = out / "comparison_report.md"
    rpath.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {rpath}")


if __name__ == "__main__":
    main()
