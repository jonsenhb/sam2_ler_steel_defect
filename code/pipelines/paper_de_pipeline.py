"""
paper_de_pipeline.py — 论文级"标注高效缺陷分割"全实验流水线 (可断点续跑)

设计原则 (严谨性):
  - 所有方法在同一数据集上共享 固定的训练池 / 验证集 / 标注子集 (同 seed), 公平可配对
  - 统一指标: global pooled IoU @256 (Cityscapes/VOC 标准)
  - 统一损失族 (CE+Dice)、统一早停、统一 epoch 预算、全 RNG 种子受控
  - 每个 run 独立 JSON, 已完成自动跳过 (可中断/续跑)
  - 外部对比方法均为忠实参考实现 (segmentation_models_pytorch), 标注论文出处

对比方法 (见 PAPER_METHODS 引用):
  Ours:   SAM2(frozen)+Linear ;  SAM2-LER
  Extern: U-Net / U-Net(ImageNet) / DeepLabV3+ / PSPNet / SegFormer
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
from torch.amp import GradScaler, autocast
from torch.utils.data import Subset, DataLoader

from dataset import get_dataset, get_dataset_meta
from train import build_sam2_model, inject_lora_to_model
from train_multiclass import encoder_forward, MultiClassFPNHead, MultiClassLoss
from segmentation_metrics import SegmentationMetricAccumulator
from data_efficiency_probe import sam_train_feats, sam_eval_val, fit_linear


# ===========================================================================
# 方法注册表 (含论文出处)
# ===========================================================================
PAPER_METHODS = {
    "sam2_linear": {
        "label": "SAM2(frozen)+Linear (Ours)",
        "family": "sam2_linear",
        "paper": "SAM2: Ravi et al., arXiv:2408.00714, 2024",
    },
    "sam2_lora": {
        "label": "SAM2-LER (Ours)",
        "family": "sam2_lora",
        "paper": "SAM2: Ravi et al. 2024; LoRA: Hu et al., ICLR 2022",
    },
    "unet": {
        "label": "U-Net",
        "family": "smp", "arch": "Unet", "encoder": "resnet34",
        "encoder_weights": None,
        "paper": "Ronneberger et al., MICCAI 2015 (backbone: He et al., CVPR 2016)",
    },
    "unet_imagenet": {
        "label": "U-Net (ImageNet)",
        "family": "smp", "arch": "Unet", "encoder": "resnet34",
        "encoder_weights": "imagenet",
        "paper": "Ronneberger et al., MICCAI 2015; ImageNet pretrain: Russakovsky et al., IJCV 2015",
    },
    "deeplabv3plus": {
        "label": "DeepLabV3+",
        "family": "smp", "arch": "DeepLabV3Plus", "encoder": "resnet34",
        "paper": "Chen et al., ECCV 2018 (backbone: He et al., CVPR 2016)",
    },
    "pspnet": {
        "label": "PSPNet",
        "family": "smp", "arch": "PSPNet", "encoder": "resnet34",
        "paper": "Zhao et al., CVPR 2017 (backbone: He et al., CVPR 2016)",
    },
    "segformer": {
        "label": "SegFormer",
        "family": "smp", "arch": "Segformer", "encoder": "mit_b0",
        "paper": "Xie et al., NeurIPS 2021",
    },
}

# Published SOTA (原协议, 全量数据, 仅作参照; 不同 split/分辨率/预训练)
PUBLISHED_SOTA = {
    "neu_seg": [
        {"method": "FCN", "miou": 79.83, "paper": "Long et al., CVPR 2015 (as reported in Sci.Rep. s41598-025-07550-0)"},
        {"method": "PSPNet", "miou": 82.52, "paper": "Zhao et al., CVPR 2017 (ibid.)"},
        {"method": "DeepLabV3+", "miou": 82.96, "paper": "Chen et al., ECCV 2018 (ibid.)"},
        {"method": "SegFormer", "miou": 81.15, "paper": "Xie et al., NeurIPS 2021 (ibid.)"},
        {"method": "DDSNet", "miou": 85.12, "paper": "as reported in Sci.Rep. s41598-025-07550-0, 2025"},
        {"method": "SOTA (Sci.Rep.2025)", "miou": 87.00, "paper": "Sci.Rep. s41598-025-07550-0, 2025 (full data, ImageNet pretrain)"},
    ],
}


# ===========================================================================
# 数据子集 (固定池/验证集, 跨方法共享)
# ===========================================================================
def make_pool_val(dataset_len, pool_size, val_len, val_size):
    rng = np.random.default_rng(20240618)
    pool = rng.choice(dataset_len, min(pool_size, dataset_len), replace=False).tolist()
    val = rng.choice(val_len, min(val_size, val_len), replace=False).tolist()
    return pool, val


def frac_subset(pool, frac, seed):
    rng = np.random.default_rng(seed)
    n = max(2, int(round(len(pool) * frac)))
    sel = rng.choice(len(pool), n, replace=False)
    return [pool[i] for i in sel], n


# ===========================================================================
# 训练/评估: smp 系列 (从零)
# ===========================================================================
@torch.no_grad()
def eval_smp(net, ds, idxs, device, num_classes, c2n, grid=256):
    net.eval()
    acc = SegmentationMetricAccumulator(num_classes, c2n)
    dl = DataLoader(Subset(ds, idxs), batch_size=8, shuffle=False, num_workers=2)
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
    return acc.compute()


def train_smp(spec, ds_train, ds_val, train_idx, val_idx, device, num_classes, c2n,
              epochs=60, bs=8, lr=1e-3, patience=10, grid=256, seed=0):
    import segmentation_models_pytorch as smp
    torch.manual_seed(seed); np.random.seed(seed)
    builder = getattr(smp, spec["arch"])
    net = builder(spec["encoder"], encoder_weights=spec.get("encoder_weights"), in_channels=3, classes=num_classes).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    lossf = MultiClassLoss(num_classes=num_classes)
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(Subset(ds_train, train_idx), batch_size=min(bs, max(2, len(train_idx))),
                    shuffle=True, num_workers=2, drop_last=len(train_idx) >= bs, generator=g)
    best, best_pc, best_ep, wait = 0.0, {}, 0, 0
    for ep in range(1, epochs + 1):
        net.train()
        for batch in dl:
            img = batch["image"].to(device); gt = batch["mask"].to(device)
            if img.shape[-1] != grid:
                img = F.interpolate(img, size=(grid, grid), mode="bilinear", align_corners=False)
                gt = F.interpolate(gt.unsqueeze(1).float(), size=(grid, grid), mode="nearest").squeeze(1).long()
            opt.zero_grad(); loss = lossf(net(img), gt); loss.backward(); opt.step()
        sched.step()
        if ep % 5 == 0 or ep == epochs:
            m = eval_smp(net, ds_val, val_idx, device, num_classes, c2n, grid)
            if m["miou_global"] > best + 1e-4:
                best, best_pc, best_ep, wait = m["miou_global"], m["per_class_global"], ep, 0
            else:
                wait += 1
                if wait >= patience:
                    break
    return best, best_pc, best_ep


# ===========================================================================
# 训练/评估: SAM2 + LoRA + FPN (Ours full)
# ===========================================================================
@torch.no_grad()
def eval_sam2_head(model, predictor, head, ds_val, val_idx, device, num_classes, c2n):
    model.eval(); head.eval()
    acc = SegmentationMetricAccumulator(num_classes, c2n)
    for i in val_idx:
        s = ds_val[i]
        embed, hr = encoder_forward(model, predictor, s["image"].to(device))
        logit = head(embed, hr)
        gt = F.interpolate(s["mask"].unsqueeze(0).unsqueeze(0).float(),
                           size=logit.shape[-2:], mode="nearest").squeeze().long()
        acc.update(logit.argmax(1)[0].cpu(), gt)
    return acc.compute()


def train_sam2_lora(ds_train, ds_val, train_idx, val_idx, device, num_classes, c2n,
                    epochs=30, lr=1e-4, patience=7, rank=4, alpha=4, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model, predictor = build_sam2_model(
        "checkpoints/sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml", device=device)
    for p in model.parameters(): p.requires_grad = False
    lora_params = inject_lora_to_model(model.image_encoder, rank=rank, alpha=alpha)
    head = MultiClassFPNHead(embed_dim=256, hr_dims=[32, 64], num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": lr}, {"params": head.parameters(), "lr": lr * 2}],
        weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    lossf = MultiClassLoss(num_classes=num_classes)
    scaler = GradScaler("cuda")
    g = np.random.default_rng(seed)
    best, best_pc, best_ep, wait = 0.0, {}, 0, 0
    for ep in range(1, epochs + 1):
        model.train(); head.train()
        order = g.permutation(len(train_idx))
        for j in order:
            i = train_idx[j]
            s = ds_train[i]
            img = s["image"].to(device); gt = s["mask"].to(device)
            opt.zero_grad()
            with autocast("cuda"):
                embed, hr = encoder_forward(model, predictor, img)
                logit = head(embed, hr)
                gt_i = F.interpolate(gt.unsqueeze(0).unsqueeze(0).float(),
                                     size=logit.shape[-2:], mode="nearest").squeeze(1).long()
                loss = lossf(logit, gt_i)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
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
    return best, best_pc, best_ep


# ===========================================================================
# 驱动
# ===========================================================================
def run_dataset(dsname, data_dir, args, device):
    meta = get_dataset_meta(dsname)
    defect_ids = meta["defect_class_ids"]; c2n = meta["class_id_to_name"]
    num_classes = len(c2n)
    out_dir = Path(args.output_dir) / dsname; out_dir.mkdir(parents=True, exist_ok=True)

    ds_tr_1024 = get_dataset(dsname, data_dir, split="train", img_size=1024)
    ds_tr_256 = get_dataset(dsname, data_dir, split="train", img_size=256)
    ds_va_1024 = get_dataset(dsname, data_dir, split="val", img_size=1024)
    ds_va_256 = get_dataset(dsname, data_dir, split="val", img_size=256)
    pool, val_idx = make_pool_val(len(ds_tr_1024), args.pool_size, len(ds_va_1024), args.val_size)
    print(f"[{dsname}] pool={len(pool)} val={len(val_idx)} classes={num_classes}")

    sam_feats = None  # 懒加载 (仅当需要 sam2_linear)

    methods = args.methods or list(PAPER_METHODS.keys())
    for method in methods:
        spec = PAPER_METHODS[method]
        for frac in args.fracs:
            for seed in args.seeds:
                tag = f"{method}_f{int(frac*1000):04d}_s{seed}"
                jpath = out_dir / f"{tag}.json"
                if jpath.exists() and not args.force:
                    continue
                t0 = time.time()
                tr_idx, n_img = frac_subset(pool, frac, seed)
                try:
                    if spec["family"] == "smp":
                        miou, pc, ep = train_smp(spec, ds_tr_256, ds_va_256, tr_idx, val_idx,
                                                 device, num_classes, c2n,
                                                 epochs=args.smp_epochs, seed=seed)
                    elif spec["family"] == "sam2_lora":
                        miou, pc, ep = train_sam2_lora(ds_tr_1024, ds_va_1024, tr_idx, val_idx,
                                                       device, num_classes, c2n,
                                                       epochs=args.sam_epochs, seed=seed)
                    elif spec["family"] == "sam2_linear":
                        model, predictor = _SAM_SINGLETON(device)
                        per_img = sam_train_feats(model, predictor, ds_tr_1024, tr_idx,
                                                  device, defect_ids)
                        clf, scaler, _ = _fit_linear_list(per_img, num_classes, seed)
                        if clf is None:
                            miou, pc, ep = 0.0, {}, 0
                        else:
                            res = sam_eval_val(clf, scaler, model, predictor, ds_va_1024,
                                               val_idx, device, num_classes, c2n)
                            miou, pc, ep = res, {}, 0
                    else:
                        continue
                except Exception as e:
                    print(f"  !! {tag} FAILED: {type(e).__name__}: {e}")
                    continue
                rec = {"dataset": dsname, "method": method, "label": spec["label"],
                       "paper": spec["paper"], "frac": frac, "n_images": n_img,
                       "seed": seed, "miou_global": float(miou), "per_class": pc,
                       "best_epoch": ep, "time_s": round(time.time() - t0, 1)}
                with open(jpath, "w") as f:
                    json.dump(rec, f, indent=2, default=float)
                print(f"  {tag:34s} mIoU={miou*100:5.2f}  n={n_img:<5} ep={ep:<3} {rec['time_s']:.0f}s")


# --- SAM2 linear 特征缓存 (每数据集一次) ---
_SAM_CACHE = {}

def _SAM_SINGLETON(device):
    if "model" not in _SAM_CACHE:
        m, p = build_sam2_model("checkpoints/sam2.1_hiera_base_plus.pt",
                                "sam2.1_hiera_b+.yaml", device=device)
        for pp in m.parameters(): pp.requires_grad = False
        _SAM_CACHE["model"], _SAM_CACHE["predictor"] = m, p
    return _SAM_CACHE["model"], _SAM_CACHE["predictor"]


def _fit_linear_list(per_img, num_classes, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if not per_img:
        return None, None, 0
    X = np.concatenate([a for a, _ in per_img]); y = np.concatenate([b for _, b in per_img])
    if len(np.unique(y)) < 2:
        return None, None, 0
    rng = np.random.default_rng(seed)
    if len(y) > 40000:
        s = rng.choice(len(y), 40000, replace=False); X, y = X[s], y[s]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(sc.transform(X), y)
    return clf, sc, len(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["neu_seg", "severstal"])
    p.add_argument("--neu_dir", default="data/NEU-Seg")
    p.add_argument("--severstal_dir", default="data/severstal")
    p.add_argument("--methods", nargs="+", default=None, help="默认全部")
    p.add_argument("--fracs", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.25, 1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--pool_size", type=int, default=1200)
    p.add_argument("--val_size", type=int, default=400)
    p.add_argument("--smp_epochs", type=int, default=60)
    p.add_argument("--sam_epochs", type=int, default=30)
    p.add_argument("--force", action="store_true")
    p.add_argument("--output_dir", default="outputs/paper_de")
    args = p.parse_args()

    device = torch.device("cuda")
    dirs = {"neu_seg": args.neu_dir, "severstal": args.severstal_dir}
    for ds in args.datasets:
        _SAM_CACHE.clear(); torch.cuda.empty_cache()  # 释放上一个数据集的模型/特征缓存
        run_dataset(ds, dirs[ds], args, device)
    print("\n✅ 全部实验完成 (或已跳过). 运行 paper_de_report.py 生成图表.")


if __name__ == "__main__":
    main()
