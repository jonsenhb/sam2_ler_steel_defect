"""
paper_defectsam2_pipeline.py — DefectSAM2 全协议实验 (与 paper_de 相同划分/指标)

方法: SAM2(frozen) + LoRA + DefectSAM2Head (CGFA + MGFA + FPN decode)
输出: outputs/paper_defectsam2/{dataset}/defectsam2_f{frac}_s{seed}.json

默认: neu_seg + severstal × fracs {1,5,10,25,100}% × seeds {0,1,2} → 30 runs
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
from defect_sam2_adapt import DefectSAM2Head, count_head_params
from paper_de_pipeline import eval_sam2_head, frac_subset, make_pool_val
from train import build_sam2_model, inject_lora_to_model
from train_multiclass import MultiClassLoss, encoder_forward

METHOD_LABEL = "SAM2(frozen)+LoRA+DefectSAM2 (Ours)"
METHOD_KEY = "defectsam2"
PAPER_REF = (
    "DefectSAM2: CGFA+MGFA hierarchical adaptation; "
    "SAM2: Ravi et al. 2024; LoRA: Hu et al. ICLR 2022; "
    "DefectSAM: Yan et al. TNNLS 2025"
)


def train_defectsam2(
    ds_train,
    ds_val,
    train_idx,
    val_idx,
    device,
    num_classes,
    c2n,
    *,
    epochs=30,
    lr=1e-4,
    patience=7,
    rank=4,
    alpha=4,
    seed=0,
    coarse_weight=0.25,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, predictor = build_sam2_model(
        "checkpoints/sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml", device=device,
    )
    for p in model.parameters():
        p.requires_grad = False
    lora_params = inject_lora_to_model(model.image_encoder, rank=rank, alpha=alpha)
    head = DefectSAM2Head(embed_dim=256, hr_dims=[32, 64], num_classes=num_classes).to(device)
    head_params = count_head_params(head)

    lossf = MultiClassLoss(num_classes=num_classes)
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": lr}, {"params": head.parameters(), "lr": lr * 2}],
        weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    scaler = GradScaler("cuda")
    g = np.random.default_rng(seed)
    best, best_pc, best_ep, wait = 0.0, {}, 0, 0

    for ep in range(1, epochs + 1):
        model.train()
        head.train()
        order = g.permutation(len(train_idx))
        for j in order:
            i = train_idx[j]
            s = ds_train[i]
            img = s["image"].to(device)
            gt = s["mask"].to(device)
            opt.zero_grad()
            with autocast("cuda"):
                embed, hr = encoder_forward(model, predictor, img)
                logit, coarse = head.forward_with_aux(embed, hr)
                gt_i = F.interpolate(
                    gt.unsqueeze(0).unsqueeze(0).float(),
                    size=logit.shape[-2:],
                    mode="nearest",
                ).squeeze(1).long()
                gt_c = F.interpolate(
                    gt.unsqueeze(0).unsqueeze(0).float(),
                    size=coarse.shape[-2:],
                    mode="nearest",
                ).squeeze(1).long()
                loss = lossf(logit, gt_i) + coarse_weight * lossf(coarse, gt_c)
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
    return best, best_pc, best_ep, head_params


def run_dataset(dsname, data_dir, args, device):
    meta = get_dataset_meta(dsname)
    c2n = meta["class_id_to_name"]
    num_classes = len(c2n)
    out_dir = Path(args.output_dir) / dsname
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_tr = get_dataset(dsname, data_dir, split="train", img_size=1024)
    ds_va = get_dataset(dsname, data_dir, split="val", img_size=1024)
    pool, val_idx = make_pool_val(len(ds_tr), args.pool_size, len(ds_va), args.val_size)
    print(f"[{dsname}] pool={len(pool)} val={len(val_idx)} classes={num_classes}")

    for frac in args.fracs:
        for seed in args.seeds:
            tag = f"{METHOD_KEY}_f{int(round(frac * 1000)):04d}_s{seed}"
            jpath = out_dir / f"{tag}.json"
            if jpath.exists() and not args.force:
                print(f"  skip {tag} (exists)")
                continue

            t0 = time.time()
            tr_idx, n_img = frac_subset(pool, frac, seed)
            try:
                miou, pc, ep, head_p = train_defectsam2(
                    ds_tr, ds_va, tr_idx, val_idx, device, num_classes, c2n,
                    epochs=args.epochs, patience=args.patience, seed=seed,
                    coarse_weight=args.coarse_weight,
                )
            except Exception as e:
                print(f"  !! {tag} FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue

            rec = {
                "dataset": dsname,
                "method": METHOD_KEY,
                "label": METHOD_LABEL,
                "paper": PAPER_REF,
                "frac": frac,
                "n_images": n_img,
                "seed": seed,
                "miou_global": float(miou),
                "per_class": pc,
                "best_epoch": ep,
                "head_params": head_p,
                "time_s": round(time.time() - t0, 1),
                "paper_de_protocol": True,
                "architecture": "DefectSAM2Head(CGFA+MGFA+FPN)",
            }
            with open(jpath, "w") as f:
                json.dump(rec, f, indent=2, default=float)
            print(
                f"  {tag:34s} mIoU={miou * 100:5.2f}%  n={n_img:<5} ep={ep:<3} "
                f"head={head_p / 1e6:.2f}M  {rec['time_s']:.0f}s"
            )


def main():
    p = argparse.ArgumentParser(description="DefectSAM2 full protocol experiments")
    p.add_argument("--datasets", nargs="+", default=["neu_seg", "severstal"])
    p.add_argument("--neu_dir", default="data/NEU-Seg")
    p.add_argument("--severstal_dir", default="data/severstal")
    p.add_argument("--fracs", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.25, 1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--pool_size", type=int, default=1200)
    p.add_argument("--val_size", type=int, default=400)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--coarse_weight", type=float, default=0.25)
    p.add_argument("--force", action="store_true")
    p.add_argument("--output_dir", default="outputs/paper_defectsam2")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = {"neu_seg": args.neu_dir, "severstal": args.severstal_dir}

    print("=" * 72)
    print(" DefectSAM2 Pipeline — full paper_de protocol")
    print(f" datasets={args.datasets} fracs={args.fracs} seeds={args.seeds}")
    print(f" output={args.output_dir}")
    print("=" * 72)

    for ds in args.datasets:
        torch.cuda.empty_cache()
        run_dataset(ds, dirs[ds], args, device)

    print("\n✅ DefectSAM2 pipeline complete.")


if __name__ == "__main__":
    main()
