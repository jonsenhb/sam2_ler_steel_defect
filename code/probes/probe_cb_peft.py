"""
probe_cb_peft.py — Probe-A: Class-Balanced PEFT (CB-PEFT)

训练时对「含稀有缺陷类」的图像做过采样 (importance sampling on images),
不改 loss / 不改 LoRA 结构 — 针对低标注下难类梯度 starvation.

Go @1%:  patches >= baseline+5pp  AND  mIoU >= baseline-1pp
Go @10%: patches >= baseline+3pp  AND  mIoU >= baseline-0.5pp
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dataset import get_dataset, get_dataset_meta
from paper_de_pipeline import make_pool_val, frac_subset
from probe_de_common import (
    compute_cb_image_weights,
    defect_iou,
    load_baseline,
    train_sam2_probe,
)


def verdict_cb(frac: float, miou: float, pc: dict, base: dict) -> str:
    b_miou = base["miou_global"] * 100
    b_pat = defect_iou(base.get("per_class", {}), "patches")
    p_pat = defect_iou(pc, "patches")
    d_miou = miou - b_miou
    d_pat = (p_pat - b_pat) if (p_pat is not None and b_pat is not None) else None

    if frac <= 0.02:
        if d_pat is not None and d_pat >= 5.0 and d_miou >= -1.0:
            return "GO"
        if d_pat is not None and d_pat >= 2.0 and d_miou >= -2.0:
            return "MAYBE"
        return "NO-GO"
    if d_pat is not None and d_pat >= 3.0 and d_miou >= -0.5:
        return "GO"
    if d_pat is not None and d_pat >= 1.5:
        return "MAYBE"
    return "NO-GO"


def run(args):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = get_dataset_meta(args.dataset)
    num_classes = len(meta["class_id_to_name"])
    c2n = meta["class_id_to_name"]
    defect_ids = meta["defect_class_ids"]

    ds_tr = get_dataset(args.dataset, args.data_dir, split="train", img_size=1024)
    ds_va = get_dataset(args.dataset, args.data_dir, split="val", img_size=1024)
    pool, val_idx = make_pool_val(len(ds_tr), args.pool_size, len(ds_va), args.val_size)

    rows = []
    print(f"\n{'='*72}\n  Probe-A: CB-PEFT  dataset={args.dataset}  seed={args.seed}\n{'='*72}")

    for frac in args.fracs:
        train_idx, n_img = frac_subset(pool, frac, args.seed)
        tag = f"f{int(round(frac * 1000)):04d}"
        base = load_baseline(Path(args.baseline_dir), args.dataset, frac, args.seed)
        w, cb_probs, present, img_count = compute_cb_image_weights(ds_tr, train_idx, defect_ids)

        print(f"\n[frac={frac:.0%}  n={n_img}]  CB image weights: min={w.min():.2f} max={w.max():.2f}")
        for c in defect_ids:
            print(f"  class {c2n[c]}: {img_count[c]}/{n_img} images in subset")

        run_name = f"sam2_lora_cb_{tag}_s{args.seed}"
        json_path = out_dir / f"{run_name}.json"
        if json_path.exists() and not args.force:
            with open(json_path) as f:
                rec = json.load(f)
            miou = rec["miou_global"] * 100
            pc = rec.get("per_class", {})
            ep = rec.get("best_epoch", 0)
            elapsed = rec.get("time_s", 0)
        else:
            t0 = time.time()
            print(f">>> CB-PEFT training  frac={frac:.0%}")
            best, pc_raw, ep, head_p, _ = train_sam2_probe(
                ds_tr, ds_va, train_idx, val_idx, device, num_classes, c2n,
                use_dsa=False, sampler="cb", cb_probs=cb_probs,
                epochs=args.epochs, patience=args.patience, seed=args.seed,
            )
            elapsed = time.time() - t0
            miou = best * 100
            pc = {k: float(v) for k, v in pc_raw.items()}
            rec = {
                "dataset": args.dataset,
                "method": run_name,
                "label": "SAM2+LoRA+FPN + CB-PEFT (Probe-A)",
                "probe": "cb_peft",
                "frac": frac,
                "n_images": n_img,
                "seed": args.seed,
                "miou_global": best,
                "per_class": pc,
                "best_epoch": ep,
                "cb_weights": w.tolist(),
                "img_class_counts": {c2n[c]: img_count[c] for c in defect_ids},
                "head_params": head_p,
                "time_s": round(elapsed, 1),
                "paper_de_protocol": True,
            }
            with open(json_path, "w") as f:
                json.dump(rec, f, indent=2)
            print(f"    done  mIoU={miou:.2f}%  patches={defect_iou(pc,'patches') or 0:.1f}%  {elapsed:.0f}s")

        v = verdict_cb(frac, miou, pc, base) if base else "—"
        d_miou = d_pat = None
        if base:
            d_miou = miou - base["miou_global"] * 100
            bp, pp = defect_iou(base.get("per_class", {}), "patches"), defect_iou(pc, "patches")
            if bp is not None and pp is not None:
                d_pat = pp - bp
        rows.append({"frac": frac, "miou": miou, "delta_miou": d_miou, "delta_patches": d_pat, "verdict": v})

    print(f"\n{'Frac':<8}{'mIoU':>8}{'ΔmIoU':>8}{'ΔPat':>8}{'Verdict':>10}")
    for r in rows:
        dm = f"{r['delta_miou']:+.1f}" if r["delta_miou"] is not None else "—"
        dp = f"{r['delta_patches']:+.1f}" if r["delta_patches"] is not None else "—"
        print(f"{r['frac']*100:>5.0f}%  {r['miou']:>7.1f}{dm:>8}{dp:>8}{r['verdict']:>10}")

    summary = out_dir / "probe_cb_summary.json"
    with open(summary, "w") as f:
        json.dump({"probe": "cb_peft", "rows": rows}, f, indent=2)
    print(f"\n  Wrote {summary}")
    print(f"  Overall: {'GO/MAYBE — 值得扩展实验' if any(r['verdict'] in ('GO','MAYBE') for r in rows) else 'NO-GO'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Probe-A: Class-Balanced PEFT")
    p.add_argument("--dataset", default="neu_seg")
    p.add_argument("--data_dir", default="data/NEU-Seg")
    p.add_argument("--baseline_dir", default="outputs/paper_de")
    p.add_argument("--output_dir", default="outputs/probe_ab/cb_peft")
    p.add_argument("--pool_size", type=int, default=1200)
    p.add_argument("--val_size", type=int, default=400)
    p.add_argument("--fracs", type=float, nargs="+", default=[0.01, 0.10])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--force", action="store_true")
    run(p.parse_args())
