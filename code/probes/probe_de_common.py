"""Shared training / eval helpers for paper_de-protocol probes (Probe-A/B)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from paper_de_pipeline import eval_sam2_head
from train import build_sam2_model, inject_lora_to_model
from train_multiclass import encoder_forward, MultiClassFPNHead, MultiClassLoss


def load_baseline(baseline_dir: Path, dataset: str, frac: float, seed: int) -> dict | None:
    tag = f"f{int(round(frac * 1000)):04d}"
    path = baseline_dir / dataset / f"sam2_lora_{tag}_s{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def defect_iou(per_class: dict, name: str) -> float | None:
    v = per_class.get(name)
    return float(v) * 100 if v is not None else None


def compute_cb_image_weights(ds, train_idx, defect_ids):
    """
    Class-balanced image weights (Probe-A / CB-PEFT).
    Images containing rare defect classes (by image frequency) get higher sampling weight.
    """
    img_count = {c: 0 for c in defect_ids}
    present_list: list[set[int]] = []
    for i in train_idx:
        mask = ds[i]["mask"].numpy()
        present = {int(c) for c in np.unique(mask) if c in defect_ids}
        present_list.append(present)
        for c in present:
            img_count[c] += 1

    weights = np.ones(len(train_idx), dtype=np.float64)
    for k, present in enumerate(present_list):
        if present:
            weights[k] = max(1.0 / max(img_count[c], 1) for c in present)
    weights /= weights.mean()
    probs = weights / weights.sum()
    return weights, probs, present_list, img_count


def epoch_train_order(n: int, rng: np.random.Generator, sampler: str, cb_probs=None):
    if sampler == "cb" and cb_probs is not None:
        return rng.choice(n, size=n, replace=True, p=cb_probs)
    return rng.permutation(n)


def train_sam2_probe(
    ds_train,
    ds_val,
    train_idx,
    val_idx,
    device,
    num_classes,
    c2n,
    *,
    use_dsa=False,
    sampler="uniform",
    cb_probs=None,
    epochs=30,
    lr=1e-4,
    patience=7,
    rank=4,
    alpha=4,
    seed=0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, predictor = build_sam2_model(
        "checkpoints/sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml", device=device,
    )
    for p in model.parameters():
        p.requires_grad = False
    lora_params = inject_lora_to_model(model.image_encoder, rank=rank, alpha=alpha)
    head = MultiClassFPNHead(
        embed_dim=256, hr_dims=[32, 64], num_classes=num_classes, use_dsa=use_dsa,
    ).to(device)
    lossf = MultiClassLoss(num_classes=num_classes)
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": lr}, {"params": head.parameters(), "lr": lr * 2}],
        weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    scaler = GradScaler("cuda")
    rng = np.random.default_rng(seed)
    best, best_pc, best_ep, wait = 0.0, {}, 0, 0

    head_params = sum(p.numel() for p in head.parameters())
    dsa_params = sum(p.numel() for p in head.spatial_adapter.parameters()) if use_dsa else 0

    for ep in range(1, epochs + 1):
        model.train()
        head.train()
        order = epoch_train_order(len(train_idx), rng, sampler, cb_probs)
        for j in order:
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
    return best, best_pc, best_ep, head_params, dsa_params
