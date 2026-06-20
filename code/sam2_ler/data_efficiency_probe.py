"""
data_efficiency_probe.py — 方向①(数据高效缺陷分割)可行性探针

画数据效率曲线: 在 {1%,5%,10%,25%,100%} 训练标注下, 对比
  (a) 冻结 SAM2 + 线性探针 (1×1, 纯表示质量)
  (b) 从零训练的小 U-Net (标准从头训基线)
两者同 val (global pooled IoU @256) 评估.

判定(GO): SAM2 在低数据区大幅领先, 高数据区收敛 → 基础模型先验的数据效率优势成立,
          小样本框架是可赢的赛道.
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
from tqdm import tqdm

from torch.utils.data import Subset, DataLoader
from dataset import get_dataset, get_dataset_meta
from train import build_sam2_model
from train_multiclass import encoder_forward, MultiClassLoss
from run_asi_experiments import pool_feature_maps
from segmentation_metrics import SegmentationMetricAccumulator


# ---------------------------------------------------------------------------
# 小 U-Net (从零训练基线)
# ---------------------------------------------------------------------------

def _cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SmallUNet(nn.Module):
    def __init__(self, num_classes=4, ch=32):
        super().__init__()
        self.e1 = _cbr(3, ch); self.e2 = _cbr(ch, ch * 2); self.e3 = _cbr(ch * 2, ch * 4)
        self.bott = _cbr(ch * 4, ch * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2); self.d3 = _cbr(ch * 8, ch * 4)
        self.u2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2); self.d2 = _cbr(ch * 4, ch * 2)
        self.u1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2); self.d1 = _cbr(ch * 2, ch)
        self.out = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


# ---------------------------------------------------------------------------
# (a) 冻结 SAM2 + 线性探针
# ---------------------------------------------------------------------------

@torch.no_grad()
def sam_train_feats(model, predictor, ds_sam, idxs, device, defect_ids, grid=256, max_px=120):
    """逐图采样平衡像素特征, 返回 per-image 列表 (便于按比例取子集)."""
    classes = [0] + list(defect_ids)
    per_img = []
    for i in tqdm(idxs, desc="SAM2 train特征"):
        s = ds_sam[i]
        embed, hr = encoder_forward(model, predictor, s["image"].to(device))
        fused = pool_feature_maps(embed, hr, target_size=embed.shape[-2:])
        fused = F.interpolate(fused, size=(grid, grid), mode="bilinear", align_corners=False)
        ms = F.interpolate(s["mask"].unsqueeze(0).unsqueeze(0).float().to(device),
                           size=(grid, grid), mode="nearest").squeeze().long().cpu().numpy()
        fv = fused.squeeze(0).permute(1, 2, 0).reshape(-1, fused.shape[1]).float().cpu().numpy()
        lv = ms.reshape(-1)
        rng = np.random.default_rng(i)
        fs, ls = [], []
        for c in classes:
            idx = np.where(lv == c)[0]
            if len(idx) == 0:
                continue
            if len(idx) > max_px:
                idx = rng.choice(idx, max_px, replace=False)
            fs.append(fv[idx]); ls.append(lv[idx])
        if fs:
            per_img.append((np.concatenate(fs), np.concatenate(ls)))
    return per_img


@torch.no_grad()
def sam_eval_val(clf, scaler, model, predictor, ds_sam, idxs, device,
                 num_classes, c2n, grid=256):
    acc = SegmentationMetricAccumulator(num_classes, c2n)
    W = torch.tensor(clf.coef_, dtype=torch.float32, device=device)      # (K, C)
    b = torch.tensor(clf.intercept_, dtype=torch.float32, device=device)  # (K,)
    cls_labels = list(clf.classes_)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    std = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
    for i in idxs:
        s = ds_sam[i]
        embed, hr = encoder_forward(model, predictor, s["image"].to(device))
        fused = pool_feature_maps(embed, hr, target_size=embed.shape[-2:])
        fused = F.interpolate(fused, size=(grid, grid), mode="bilinear", align_corners=False)
        C = fused.shape[1]
        fv = fused.squeeze(0).permute(1, 2, 0).reshape(-1, C)
        fv = (fv - mean) / std
        logits = fv @ W.t() + b                      # (N, K)
        pred_idx = logits.argmax(1).cpu().numpy()
        pred = np.array(cls_labels)[pred_idx].reshape(grid, grid)
        gt = F.interpolate(s["mask"].unsqueeze(0).unsqueeze(0).float(),
                           size=(grid, grid), mode="nearest").squeeze().long().numpy()
        acc.update(torch.from_numpy(pred), torch.from_numpy(gt))
    return acc.compute()["miou_global"]


def fit_linear(per_img, frac, seed, num_classes):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    n = max(1, int(round(len(per_img) * frac)))
    sel = rng.choice(len(per_img), n, replace=False)
    X = np.concatenate([per_img[i][0] for i in sel])
    y = np.concatenate([per_img[i][1] for i in sel])
    if len(np.unique(y)) < 2:
        return None, None, n
    if len(y) > 40000:
        s = rng.choice(len(y), 40000, replace=False); X, y = X[s], y[s]
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(scaler.transform(X), y)
    return clf, scaler, n


# ---------------------------------------------------------------------------
# (b) 从零 U-Net
# ---------------------------------------------------------------------------

def train_cnn(ds_cnn, train_idxs, val_idxs, device, num_classes, c2n,
              epochs=40, bs=8, lr=1e-3, grid=256):
    net = SmallUNet(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = MultiClassLoss(num_classes=num_classes)
    tr = DataLoader(Subset(ds_cnn, train_idxs), batch_size=bs, shuffle=True,
                    num_workers=2, drop_last=len(train_idxs) >= bs)
    best = 0.0
    for ep in range(epochs):
        net.train()
        for batch in tr:
            img = batch["image"].to(device)
            gt = batch["mask"].to(device)
            if img.shape[-1] != grid:
                img = F.interpolate(img, size=(grid, grid), mode="bilinear", align_corners=False)
                gt = F.interpolate(gt.unsqueeze(1).float(), size=(grid, grid), mode="nearest").squeeze(1).long()
            opt.zero_grad()
            logit = net(img)
            loss = lossf(logit, gt)
            loss.backward(); opt.step()
        # 评估 (每若干 epoch)
        if ep >= epochs - 1 or (ep + 1) % 10 == 0:
            miou = eval_cnn(net, ds_cnn, val_idxs, device, num_classes, c2n, grid)
            best = max(best, miou)
    return best


@torch.no_grad()
def eval_cnn(net, ds_cnn, val_idxs, device, num_classes, c2n, grid=256):
    net.eval()
    acc = SegmentationMetricAccumulator(num_classes, c2n)
    dl = DataLoader(Subset(ds_cnn, val_idxs), batch_size=8, shuffle=False, num_workers=2)
    for batch in dl:
        img = batch["image"].to(device); gt = batch["mask"]
        if img.shape[-1] != grid:
            img = F.interpolate(img, size=(grid, grid), mode="bilinear", align_corners=False)
        logit = net(img)
        pred = logit.argmax(1).cpu()
        for b in range(pred.shape[0]):
            g = F.interpolate(gt[b:b+1].unsqueeze(1).float(), size=(grid, grid),
                              mode="nearest").squeeze(1).long()[0]
            acc.update(pred[b], g)
    return acc.compute()["miou_global"]


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="neu_seg")
    p.add_argument("--data_dir", default="data/NEU-Seg")
    p.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt")
    p.add_argument("--model_cfg", default="sam2.1_hiera_b+.yaml")
    p.add_argument("--pool_size", type=int, default=1000, help="训练池大小 (100%=pool)")
    p.add_argument("--val_size", type=int, default=300)
    p.add_argument("--cnn_epochs", type=int, default=40)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--fracs", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.25, 1.0])
    p.add_argument("--output_dir", default="outputs/data_efficiency")
    args = p.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meta = get_dataset_meta(args.dataset)
    defect_ids = meta["defect_class_ids"]; c2n = meta["class_id_to_name"]
    num_classes = len(c2n)

    ds_sam = get_dataset(args.dataset, args.data_dir, split="train", img_size=1024)
    ds_cnn = get_dataset(args.dataset, args.data_dir, split="train", img_size=256)
    ds_val_sam = get_dataset(args.dataset, args.data_dir, split="val", img_size=1024)
    ds_val_cnn = get_dataset(args.dataset, args.data_dir, split="val", img_size=256)

    rng = np.random.default_rng(0)
    pool = rng.choice(len(ds_sam), min(args.pool_size, len(ds_sam)), replace=False).tolist()
    val_idx = rng.choice(len(ds_val_sam), min(args.val_size, len(ds_val_sam)), replace=False).tolist()
    print(f"训练池={len(pool)} 验证={len(val_idx)} | 类别={num_classes} | fracs={args.fracs}")

    model, predictor = build_sam2_model(args.checkpoint, args.model_cfg, device=device)
    for pp in model.parameters(): pp.requires_grad = False

    # 预提取 SAM2 训练池特征 (一次)
    t0 = time.time()
    per_img = sam_train_feats(model, predictor, ds_sam, pool, device, defect_ids)
    print(f"SAM2 训练特征提取完成 ({time.time()-t0:.0f}s), per-image={len(per_img)}")

    results = {"fracs": args.fracs, "sam": {}, "cnn": {}}
    for frac in args.fracs:
        sam_seeds, cnn_seeds = [], []
        for sd in args.seeds:
            clf, scaler, n_img = fit_linear(per_img, frac, sd, num_classes)
            if clf is not None:
                miou_sam = sam_eval_val(clf, scaler, model, predictor, ds_val_sam,
                                        val_idx, device, num_classes, c2n)
                sam_seeds.append(miou_sam)
            # CNN: 同样的图像子集
            rng2 = np.random.default_rng(sd)
            n = max(1, int(round(len(pool) * frac)))
            sub = rng2.choice(len(pool), n, replace=False)
            cnn_train_idx = [pool[i] for i in sub]
            miou_cnn = train_cnn(ds_cnn, cnn_train_idx, val_idx, device, num_classes, c2n,
                                 epochs=args.cnn_epochs)
            cnn_seeds.append(miou_cnn)
        results["sam"][frac] = sam_seeds
        results["cnn"][frac] = cnn_seeds
        print(f"  frac={frac*100:>5.1f}% (n≈{n_img:>4}图) | "
              f"SAM2={np.mean(sam_seeds)*100:5.2f}±{np.std(sam_seeds)*100:.2f} | "
              f"CNN={np.mean(cnn_seeds)*100:5.2f}±{np.std(cnn_seeds)*100:.2f}")

    # 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fr = np.array(args.fracs) * 100
        sam_m = [np.mean(results["sam"][f]) * 100 for f in args.fracs]
        sam_s = [np.std(results["sam"][f]) * 100 for f in args.fracs]
        cnn_m = [np.mean(results["cnn"][f]) * 100 for f in args.fracs]
        cnn_s = [np.std(results["cnn"][f]) * 100 for f in args.fracs]
        plt.figure(figsize=(7, 5))
        plt.errorbar(fr, sam_m, yerr=sam_s, marker="o", lw=2, capsize=4,
                     label="Frozen SAM2 + linear probe")
        plt.errorbar(fr, cnn_m, yerr=cnn_s, marker="s", lw=2, capsize=4,
                     label="From-scratch U-Net")
        plt.xscale("log"); plt.xlabel("Training labels (%)"); plt.ylabel("Val mIoU (%)")
        plt.title(f"Data efficiency on {args.dataset}")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"data_efficiency_{args.dataset}.png", dpi=150)
        print(f"  图: {out_dir/f'data_efficiency_{args.dataset}.png'}")
    except Exception as e:
        print(f"  (画图跳过: {e})")

    # 判定
    lo = args.fracs[0]
    sam_lo = np.mean(results["sam"][lo]) * 100
    cnn_lo = np.mean(results["cnn"][lo]) * 100
    sam_hi = np.mean(results["sam"][args.fracs[-1]]) * 100
    cnn_hi = np.mean(results["cnn"][args.fracs[-1]]) * 100
    low_gap = sam_lo - cnn_lo
    high_gap = sam_hi - cnn_hi
    print(f"\n{'='*70}\n  判定")
    print(f"  低数据({lo*100:.0f}%): SAM2={sam_lo:.1f} vs CNN={cnn_lo:.1f}  → 领先 {low_gap:+.1f}pp")
    print(f"  高数据(100%): SAM2={sam_hi:.1f} vs CNN={cnn_hi:.1f}  → 领先 {high_gap:+.1f}pp")
    if low_gap > 10 and low_gap > high_gap + 3:
        verdict = "GO: SAM2 在低数据区大幅领先且高数据收敛 → 数据高效/小样本框架成立, 可赢"
    elif low_gap > 5:
        verdict = "MAYBE: 低数据有优势但不够强, 可结合更强轻量头"
    else:
        verdict = "NO-GO: 低数据优势不明显, 数据高效叙事不成立"
    print(f"  → {verdict}\n{'='*70}")

    with open(out_dir / f"data_efficiency_{args.dataset}.json", "w") as f:
        json.dump({"results": results, "low_gap": low_gap, "high_gap": high_gap,
                   "verdict": verdict}, f, indent=2, default=float)
    print(f"  报告: {out_dir/f'data_efficiency_{args.dataset}.json'}")


if __name__ == "__main__":
    main()
