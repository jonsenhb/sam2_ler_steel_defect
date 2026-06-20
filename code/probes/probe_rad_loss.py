"""
probe_rad_loss.py — RAD 理论探针: 头级/损失级轻量增强 (Probe-1/2)

在 paper_de 同一协议下快速验证:
  Probe-1  rarity   — 逆平方根像素频率加权 CE (长尾 / RAD Δ_PEFT)
  Probe-2  boundary — DDSNet-style 边界加权 CE (仅 FPN 头, 不改 encoder)

用法:
  python probe_rad_loss.py                    # 跑 Probe-1/2 @ 1% & 10%, seed=0
  python probe_rad_loss.py --probes rarity    # 只跑 Probe-1
  python probe_rad_loss.py --fracs 0.01       # 只跑 1%

Go 标准 (相对已有 sam2_lora JSON):
  Probe-1: mIoU@1% >= baseline+2pp  OR  patches@1% >= baseline+3pp
  Probe-2: patches@1% >= baseline+3pp
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from dataset import get_dataset, get_dataset_meta
from paper_de_pipeline import make_pool_val, frac_subset, eval_sam2_head, PAPER_METHODS
from train import build_sam2_model, inject_lora_to_model
from train_multiclass import encoder_forward, MultiClassFPNHead, MultiClassLoss


PROBE_SPECS = {
    "baseline": {"loss_mode": "baseline", "label": "SAM2+LoRA (baseline)"},
    "rarity": {"loss_mode": "rarity", "label": "SAM2+LoRA + Rarity-weighted CE (Probe-1)"},
    "boundary": {"loss_mode": "boundary", "label": "SAM2+LoRA + Boundary-weighted CE (Probe-2)"},
}


def train_sam2_lora_probe(
    ds_train,
    ds_val,
    train_idx,
    val_idx,
    device,
    num_classes,
    c2n,
    loss_mode="baseline",
    epochs=30,
    lr=1e-4,
    patience=7,
    rank=4,
    alpha=4,
    seed=0,
    boundary_weight=0.2,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, predictor = build_sam2_model(
        "checkpoints/sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml", device=device,
    )
    for p in model.parameters():
        p.requires_grad = False
    lora_params = inject_lora_to_model(model.image_encoder, rank=rank, alpha=alpha)
    head = MultiClassFPNHead(embed_dim=256, hr_dims=[32, 64], num_classes=num_classes).to(device)

    class_weights = None
    if loss_mode in ("rarity", "rarity_boundary"):
        class_weights = MultiClassLoss.compute_rarity_weights(ds_train, train_idx, num_classes)

    lossf = MultiClassLoss(
        num_classes=num_classes,
        loss_mode=loss_mode,
        class_weights=class_weights,
        boundary_weight=boundary_weight,
    )
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": lr}, {"params": head.parameters(), "lr": lr * 2}],
        weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    scaler = GradScaler("cuda")
    rng = np.random.default_rng(seed)
    best, best_pc, best_ep, wait = 0.0, {}, 0, 0

    for ep in range(1, epochs + 1):
        model.train()
        head.train()
        for j in rng.permutation(len(train_idx)):
            i = train_idx[j]
            s = ds_train[i]
            img = s["image"].to(device)
            gt = s["mask"].to(device)
            opt.zero_grad()
            with autocast("cuda"):
                embed, hr = encoder_forward(model, predictor, img)
                logit = head(embed, hr)
                gt_i = F.interpolate(
                    gt.unsqueeze(0).unsqueeze(0).float(),
                    size=logit.shape[-2:],
                    mode="nearest",
                ).squeeze(1).long()
                loss = lossf(logit, gt_i)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        m = eval_sam2_head(model, predictor, head, ds_val, val_idx, device, num_classes, c2n)
        if m["miou_global"] > best + 1e-4:
            best, best_pc, best_ep, wait = m["miou_global"], m["per_class_global"], ep, 0
        elif ep >= 3:
            wait += 1
            if wait >= patience:
                break

    del model, head
    torch.cuda.empty_cache()
    return best, best_pc, best_ep, class_weights


def load_baseline(output_dir: Path, dataset: str, frac: float, seed: int) -> dict | None:
    tag = f"f{int(round(frac * 1000)):04d}"
    path = output_dir / dataset / f"sam2_lora_{tag}_s{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def defect_iou(per_class: dict, name: str) -> float | None:
    v = per_class.get(name)
    return float(v) * 100 if v is not None else None


def verdict(probe: str, frac: float, miou: float, pc: dict, base: dict) -> str:
    b_miou = base["miou_global"] * 100
    b_patches = defect_iou(base.get("per_class", {}), "patches")
    p_patches = defect_iou(pc, "patches")
    d_miou = miou - b_miou
    d_pat = (p_patches - b_patches) if (p_patches is not None and b_patches is not None) else None

    if frac <= 0.02 and probe == "rarity":
        if d_miou >= 2.0 or (d_pat is not None and d_pat >= 3.0):
            return "GO"
        if d_miou >= 0.5 or (d_pat is not None and d_pat >= 1.0):
            return "MAYBE"
        return "NO-GO"
    if frac <= 0.02 and probe == "boundary":
        if d_pat is not None and d_pat >= 3.0:
            return "GO"
        if d_pat is not None and d_pat >= 1.0:
            return "MAYBE"
        return "NO-GO"
    if frac <= 0.11:
        if d_miou >= 1.0:
            return "GO"
        if d_miou >= 0.3:
            return "MAYBE"
        return "NO-GO"
    return "—"


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = get_dataset_meta(args.dataset)
    num_classes = len(meta["class_id_to_name"])
    c2n = meta["class_id_to_name"]

    ds_tr = get_dataset(args.dataset, args.data_dir, split="train", img_size=1024)
    ds_va = get_dataset(args.dataset, args.data_dir, split="val", img_size=1024)
    pool, val_idx = make_pool_val(len(ds_tr), args.pool_size, len(ds_va), args.val_size)

    probes = [args.probes] if args.probes != "all" else ["rarity", "boundary"]
    rows = []

    print(f"\n{'='*72}")
    print("  RAD Loss Probes (paper_de protocol)")
    print(f"  dataset={args.dataset}  seed={args.seed}  fracs={args.fracs}")
    print(f"{'='*72}\n")

    for frac in args.fracs:
        train_idx, n_img = frac_subset(pool, frac, args.seed)
        tag = f"f{int(round(frac * 1000)):04d}"
        base = load_baseline(Path(args.baseline_dir), args.dataset, frac, args.seed)
        if base:
            print(
                f"[baseline JSON] frac={frac:.0%}  mIoU={base['miou_global']*100:.2f}%  "
                f"patches={defect_iou(base.get('per_class',{}),'patches') or float('nan'):.1f}%"
            )
        else:
            print(f"[baseline JSON] frac={frac:.0%}  — missing, skip delta")

        for probe in probes:
            spec = PROBE_SPECS[probe]
            run_name = f"sam2_lora_{probe}_{tag}_s{args.seed}"
            json_path = out_dir / f"{run_name}.json"
            if json_path.exists() and not args.force:
                with open(json_path) as f:
                    rec = json.load(f)
                miou = rec["miou_global"] * 100
                pc = rec.get("per_class", {})
                ep = rec.get("best_epoch", 0)
                elapsed = rec.get("time_s", 0)
                cw = rec.get("class_weights")
            else:
                t0 = time.time()
                print(f"\n>>> {spec['label']}  frac={frac:.0%}  n={n_img}  seed={args.seed}")
                if probe != "baseline":
                    cw_preview = (
                        MultiClassLoss.compute_rarity_weights(ds_tr, train_idx, num_classes)
                        if probe == "rarity"
                        else None
                    )
                    if cw_preview:
                        print(f"    rarity weights: {[f'{w:.3f}' for w in cw_preview]}")
                best, pc_raw, ep, cw = train_sam2_lora_probe(
                    ds_tr, ds_va, train_idx, val_idx, device, num_classes, c2n,
                    loss_mode=spec["loss_mode"],
                    epochs=args.epochs,
                    patience=args.patience,
                    seed=args.seed,
                    boundary_weight=args.boundary_weight,
                )
                elapsed = time.time() - t0
                miou = best * 100
                pc = {k: float(v) for k, v in pc_raw.items()}
                rec = {
                    "dataset": args.dataset,
                    "method": run_name,
                    "label": spec["label"],
                    "loss_mode": spec["loss_mode"],
                    "frac": frac,
                    "n_images": n_img,
                    "seed": args.seed,
                    "miou_global": best,
                    "per_class": pc,
                    "best_epoch": ep,
                    "class_weights": cw,
                    "time_s": round(elapsed, 1),
                    "paper_de_protocol": True,
                }
                with open(json_path, "w") as f:
                    json.dump(rec, f, indent=2)
                print(f"    done  mIoU={miou:.2f}%  ep={ep}  {elapsed:.0f}s")

            d_miou = d_pat = None
            v = "—"
            if base:
                d_miou = miou - base["miou_global"] * 100
                bp = defect_iou(base.get("per_class", {}), "patches")
                pp = defect_iou(pc, "patches")
                if bp is not None and pp is not None:
                    d_pat = pp - bp
                v = verdict(probe, frac, miou, pc, base)

            rows.append({
                "probe": probe,
                "frac": frac,
                "miou": miou,
                "patches": defect_iou(pc, "patches"),
                "delta_miou": d_miou,
                "delta_patches": d_pat,
                "verdict": v,
                "json": str(json_path),
            })

    # summary table
    print(f"\n{'='*72}")
    print("  Summary vs sam2_lora baseline")
    print(f"{'='*72}")
    print(f"{'Probe':<10}{'Frac':<8}{'mIoU':>8}{'ΔmIoU':>8}{'Patches':>10}{'ΔPat':>8}{'Verdict':>10}")
    for r in rows:
        dm = f"{r['delta_miou']:+.1f}" if r["delta_miou"] is not None else "—"
        dp = f"{r['delta_patches']:+.1f}" if r["delta_patches"] is not None else "—"
        pat = f"{r['patches']:.1f}" if r["patches"] is not None else "—"
        print(
            f"{r['probe']:<10}{r['frac']*100:>5.0f}%  {r['miou']:>7.1f}{dm:>8}"
            f"{pat:>10}{dp:>8}{r['verdict']:>10}"
        )

    summary_path = out_dir / "probe_rad_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"rows": rows, "baseline_dir": args.baseline_dir}, f, indent=2)
    print(f"\n  Wrote {summary_path}")

    any_go = any(r["verdict"] == "GO" for r in rows)
    print(f"\n  Overall: {'至少一项 GO → 值得全协议实验' if any_go else '无 GO → RAD 理论仍可写「预训练已编码先验, 边际递减」'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RAD loss probes (Probe-1 rarity / Probe-2 boundary)")
    p.add_argument("--dataset", default="neu_seg")
    p.add_argument("--data_dir", default="data/NEU-Seg")
    p.add_argument("--baseline_dir", default="outputs/paper_de")
    p.add_argument("--output_dir", default="outputs/rad_probes")
    p.add_argument("--pool_size", type=int, default=1200)
    p.add_argument("--val_size", type=int, default=400)
    p.add_argument("--fracs", type=float, nargs="+", default=[0.01, 0.10])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--boundary_weight", type=float, default=0.2)
    p.add_argument("--probes", default="all", choices=["all", "rarity", "boundary"])
    p.add_argument("--force", action="store_true")
    run(p.parse_args())
