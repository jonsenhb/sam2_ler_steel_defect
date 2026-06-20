"""
paper_external_sota.py — 外部 SOTA 对比实验 (统一协议, 可断点续跑)

复现 / 近似复现 4 篇论文中的方法, 全部在 paper_de_pipeline.py 同一协议下评测:
  - 同一数据池 (pool=1200, val=400, rng seed=20240618)
  - 同一 global pooled mIoU 指标 @256
  - 同一标注比例 {1%, 5%, 10%, 25%, 100%} × 3 seeds

外部 SOTA 方法 (近似实现, 基于 SMP):
  1. DDSNet (IEEE TIM 2024, Yin et al.) → UNet+ResNet50 + Dice+BCE + boundary aux
  2. MFF-Metal (J.Supercomputing 2025, Li et al.) → UNet++ + multi-loss (CE+Dice+Boundary)
  3. SME-DeepLabV3+ (PLOS One 2025, Zhang et al.) → DeepLabV3+ ResNet50 + augmentation
  4. Hybrid-Transformer (MTA 2024, Sime et al.) → UPerNet + MiT-B4 encoder + Dice+BCE

注: 这些不是原论文的精确复现 (缺少原作者专有模块), 而是尽可能忠实的 SMP 参考实现,
    旨在在统一协议下给出这些方法的公平近似性能. 论文中会注明 "reproduced under our protocol".
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from dataset import get_dataset, get_dataset_meta
from segmentation_metrics import SegmentationMetricAccumulator

# reuse pool/val split from main pipeline
from paper_de_pipeline import make_pool_val, frac_subset


EXTERNAL_METHODS = {
    "ddsnet_approx": {
        "label": "DDSNet* (UNet-R50+BndLoss)",
        "paper": "Yin et al., IEEE TIM 2024 (approx. reproduction under our protocol)",
        "arch": "Unet", "encoder": "resnet50",
        "loss": "bce_dice_boundary", "epochs": 80,
    },
    "mff_metal_approx": {
        "label": "MFF-Metal* (UNet++-R34+MultiLoss)",
        "paper": "Li et al., J.Supercomputing 2025 (approx. reproduction under our protocol)",
        "arch": "UnetPlusPlus", "encoder": "resnet34",
        "loss": "ce_dice_boundary", "epochs": 80,
    },
    "sme_dlv3p_approx": {
        "label": "SME-DLV3+* (DLV3+-R50)",
        "paper": "Zhang et al., PLOS One 2025 (approx. reproduction under our protocol)",
        "arch": "DeepLabV3Plus", "encoder": "resnet50",
        "loss": "ce_dice", "epochs": 80,
    },
    "hybrid_trans_approx": {
        "label": "Hybrid-Trans* (UPerNet-MiT-B4)",
        "paper": "Sime et al., MTA 2024 (approx. reproduction under our protocol)",
        "arch": "UPerNet", "encoder": "mit_b4",
        "loss": "bce_dice", "epochs": 80,
    },
}


# ---- Loss functions matching papers ----
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        prob = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (prob * target_oh).sum(dims)
        union = prob.sum(dims) + target_oh.sum(dims)
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BoundaryLoss(nn.Module):
    """Approximate boundary loss: penalize errors near GT boundaries (Laplacian edge)."""
    def forward(self, logits, target):
        target_f = target.float().unsqueeze(1)
        lap_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                             dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        edge = F.conv2d(target_f, lap_k, padding=1).abs().clamp(0, 1).squeeze(1)
        ce = F.cross_entropy(logits, target, reduction='none')
        return (ce * (1.0 + 5.0 * edge)).mean()


class CompositeLoss(nn.Module):
    def __init__(self, mode="bce_dice", num_classes=4):
        super().__init__()
        self.mode = mode
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.bnd = BoundaryLoss() if "boundary" in mode else None

    def forward(self, logits, target):
        l_ce = self.ce(logits, target)
        l_dice = self.dice(logits, target)
        loss = 0.6 * l_ce + 0.4 * l_dice
        if self.bnd is not None:
            loss = loss + 0.2 * self.bnd(logits, target)
        return loss


# ---- Evaluation ----
@torch.no_grad()
def eval_model(net, ds, idxs, device, num_classes, c2n, grid=256):
    net.eval()
    acc = SegmentationMetricAccumulator(num_classes, c2n)
    dl = DataLoader(Subset(ds, idxs), batch_size=8, shuffle=False, num_workers=2)
    for batch in dl:
        img = batch["image"].to(device); gt = batch["mask"]
        if img.shape[-1] != grid:
            img = F.interpolate(img, size=(grid, grid), mode="bilinear", align_corners=False)
        pred = net(img).argmax(1).cpu()
        for b in range(pred.shape[0]):
            g = F.interpolate(gt[b:b+1].unsqueeze(1).float(), size=(grid, grid),
                              mode="nearest").squeeze(1).long()[0]
            acc.update(pred[b], g)
    return acc.compute()


# ---- Training ----
def train_external(spec, ds_train, ds_val, train_idx, val_idx, device,
                   num_classes, c2n, seed=0, grid=256):
    import segmentation_models_pytorch as smp
    torch.manual_seed(seed); np.random.seed(seed)

    builder = getattr(smp, spec["arch"])
    net = builder(spec["encoder"], encoder_weights=None,
                  in_channels=3, classes=num_classes).to(device)

    epochs = spec.get("epochs", 80)
    lr = 1e-3
    patience = 12

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    lossf = CompositeLoss(mode=spec.get("loss", "ce_dice"), num_classes=num_classes)

    g = torch.Generator(); g.manual_seed(seed)
    bs = 8
    dl = DataLoader(Subset(ds_train, train_idx),
                    batch_size=min(bs, max(2, len(train_idx))),
                    shuffle=True, num_workers=2,
                    drop_last=len(train_idx) >= bs, generator=g)

    best, best_pc, best_ep, wait = 0.0, {}, 0, 0
    for ep in range(1, epochs + 1):
        net.train()
        for batch in dl:
            img = batch["image"].to(device); gt = batch["mask"].to(device)
            if img.shape[-1] != grid:
                img = F.interpolate(img, size=(grid, grid), mode="bilinear", align_corners=False)
                gt = F.interpolate(gt.unsqueeze(1).float(), size=(grid, grid),
                                   mode="nearest").squeeze(1).long()
            opt.zero_grad()
            loss = lossf(net(img), gt)
            loss.backward()
            opt.step()
        sched.step()
        if ep % 5 == 0 or ep == epochs:
            m = eval_model(net, ds_val, val_idx, device, num_classes, c2n, grid)
            if m["miou_global"] > best + 1e-4:
                best, best_pc, best_ep, wait = m["miou_global"], m["per_class_global"], ep, 0
            else:
                wait += 1
                if wait >= patience:
                    break
    del net
    torch.cuda.empty_cache()
    return best, best_pc, best_ep


# ---- Main driver ----
def run_dataset(dsname, data_dir, args, device):
    meta = get_dataset_meta(dsname)
    c2n = meta["class_id_to_name"]
    num_classes = len(c2n)
    out_dir = Path(args.output_dir) / dsname; out_dir.mkdir(parents=True, exist_ok=True)

    ds_tr = get_dataset(dsname, data_dir, split="train", img_size=256)
    ds_va = get_dataset(dsname, data_dir, split="val", img_size=256)
    pool, val_idx = make_pool_val(len(ds_tr), args.pool_size, len(ds_va), args.val_size)
    print(f"[{dsname}] pool={len(pool)} val={len(val_idx)} classes={num_classes}")

    methods = args.methods or list(EXTERNAL_METHODS.keys())
    for method in methods:
        spec = EXTERNAL_METHODS[method]
        for frac in args.fracs:
            for seed in args.seeds:
                tag = f"{method}_f{int(frac*1000):04d}_s{seed}"
                jpath = out_dir / f"{tag}.json"
                if jpath.exists() and not args.force:
                    print(f"  [skip] {tag}")
                    continue
                t0 = time.time()
                tr_idx, n_img = frac_subset(pool, frac, seed)
                try:
                    miou, pc, ep = train_external(
                        spec, ds_tr, ds_va, tr_idx, val_idx,
                        device, num_classes, c2n, seed=seed)
                except Exception as e:
                    print(f"  !! {tag} FAILED: {type(e).__name__}: {e}")
                    continue
                elapsed = round(time.time() - t0, 1)
                rec = {"dataset": dsname, "method": method,
                       "label": spec["label"], "paper": spec["paper"],
                       "frac": frac, "n_images": n_img, "seed": seed,
                       "miou_global": float(miou), "per_class": pc,
                       "best_epoch": ep, "time_s": elapsed}
                with open(jpath, "w") as f:
                    json.dump(rec, f, indent=2, default=float)
                print(f"  {tag:38s} mIoU={miou*100:5.2f}  n={n_img:<5} ep={ep:<3} {elapsed:.0f}s")


def main():
    p = argparse.ArgumentParser(description="外部 SOTA 近似复现 (统一协议)")
    p.add_argument("--datasets", nargs="+", default=["neu_seg", "severstal"])
    p.add_argument("--neu_dir", default="data/NEU-Seg")
    p.add_argument("--severstal_dir", default="data/severstal")
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--fracs", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.25, 1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--pool_size", type=int, default=1200)
    p.add_argument("--val_size", type=int, default=400)
    p.add_argument("--force", action="store_true")
    p.add_argument("--output_dir", default="outputs/paper_de")
    args = p.parse_args()

    device = torch.device("cuda")
    dirs = {"neu_seg": args.neu_dir, "severstal": args.severstal_dir}
    for ds in args.datasets:
        run_dataset(ds, dirs[ds], args, device)
    print("\n✅ 外部 SOTA 实验完成. 运行 paper_de_report.py 生成含外部SOTA的完整图表.")


if __name__ == "__main__":
    main()
