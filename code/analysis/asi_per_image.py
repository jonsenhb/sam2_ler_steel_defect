"""
Per-image ASI separability vs finetuned validation IoU (100% labels).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm

from dataset import get_dataset, get_dataset_meta
from segmentation_metrics import compute_multiclass_metrics
from train import build_sam2_model, inject_lora_to_model
from train_multiclass import MultiClassFPNHead, encoder_forward


def two_group_fisher(defect_feats: np.ndarray, bg_feats: np.ndarray) -> float:
    if len(defect_feats) < 2 or len(bg_feats) < 2:
        return 0.0
    mu_p = defect_feats.mean(axis=0)
    mu_n = bg_feats.mean(axis=0)
    between = float(np.sum((mu_p - mu_n) ** 2))
    within = float(defect_feats.var(axis=0).sum() + bg_feats.var(axis=0).sum() + 1e-8)
    return between / within


@torch.no_grad()
def compute_per_image_separability(model, predictor, dataset, device, feat_level="fused"):
    meta = get_dataset_meta("neu_seg")
    id_to_name = meta["class_id_to_name"]
    records = []

    for idx in tqdm(range(len(dataset)), desc="Per-image separability"):
        sample = dataset[idx]
        image = sample["image"].to(device)
        mask = sample["mask"]

        defect_vals = mask[mask > 0]
        if len(defect_vals) == 0:
            continue
        class_id = int(defect_vals.mode().values.item())
        class_name = id_to_name[class_id]

        embed, hr = encoder_forward(model, predictor, image)
        if feat_level == "embed":
            feat_map = embed
        elif feat_level == "hr0":
            feat_map = F.interpolate(hr[0], size=embed.shape[-2:], mode="bilinear", align_corners=False)
        else:
            from run_asi_experiments import pool_feature_maps
            feat_map = pool_feature_maps(embed, hr, target_size=embed.shape[-2:])

        mask_small = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=feat_map.shape[-2:],
            mode="nearest",
        ).squeeze().bool()

        if mask_small.sum() < 2 or (~mask_small).sum() < 2:
            continue

        fmap = feat_map.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        defect = fmap[mask_small.cpu().numpy()]
        bg = fmap[(~mask_small).cpu().numpy()]

        records.append({
            "index": idx,
            "class_id": class_id,
            "class_name": class_name,
            "separability": two_group_fisher(defect, bg),
        })
    return records


@torch.no_grad()
def load_sam2_ler_head(checkpoint_path: Path, model, num_classes: int, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    head = MultiClassFPNHead(embed_dim=256, hr_dims=[32, 64], num_classes=num_classes).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    if ckpt.get("encoder_lora"):
        for name, param in model.image_encoder.named_parameters():
            if name in ckpt["encoder_lora"]:
                param.data.copy_(ckpt["encoder_lora"][name])
    head.eval()
    return head


@torch.no_grad()
def compute_per_image_finetuned_iou(model, predictor, head, dataset, device, class_id_to_name):
    records = []
    for idx in tqdm(range(len(dataset)), desc="Per-image finetuned IoU"):
        sample = dataset[idx]
        image = sample["image"].to(device)
        mask = sample["mask"].to(device)

        defect_vals = mask[mask > 0]
        if len(defect_vals) == 0:
            continue
        class_id = int(defect_vals.mode().values.item())
        class_name = class_id_to_name[class_id]

        embed, hr_feats = encoder_forward(model, predictor, image)
        logits = head(embed, hr_feats)
        gt = mask.unsqueeze(0)
        if gt.shape[-2:] != logits.shape[-2:]:
            gt = F.interpolate(gt.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest").squeeze(1).long()

        _, per_class = compute_multiclass_metrics(
            logits, gt, num_classes=len(class_id_to_name),
            class_id_to_name=class_id_to_name, mode="per_image",
        )
        records.append({
            "index": idx,
            "class_id": class_id,
            "class_name": class_name,
            "iou": float(per_class.get(class_name, 0.0)),
        })
    return records


def merge_and_correlate(sep_records, iou_records):
    iou_by_idx = {r["index"]: r for r in iou_records}
    merged = []
    for s in sep_records:
        i = iou_by_idx.get(s["index"])
        if i is None:
            continue
        merged.append({
            "index": s["index"],
            "class_name": s["class_name"],
            "separability": s["separability"],
            "iou": i["iou"],
        })
    if len(merged) < 3:
        return merged, {"rho": float("nan"), "p_value": float("nan"), "n": len(merged)}

    sep = np.array([m["separability"] for m in merged])
    ious = np.array([m["iou"] for m in merged])
    rho, p = spearmanr(sep, ious)
    return merged, {"rho": float(rho), "p_value": float(p), "n": len(merged)}


def run_per_image_analysis(
    data_dir: str,
    dataset_name: str = "neu_seg",
    checkpoint: str = "checkpoints/sam2.1_hiera_base_plus.pt",
    model_cfg: str = "sam2.1_hiera_b+.yaml",
    finetune_ckpt: str | Path = "outputs/thesis_validation/sam2_ler_100/best_model.pth",
    img_size: int = 1024,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = get_dataset_meta(dataset_name)
    val_ds = get_dataset(dataset_name, data_dir, split="val", img_size=img_size)

    model, predictor = build_sam2_model(checkpoint, model_cfg, device=device)
    for p in model.parameters():
        p.requires_grad = False
    inject_lora_to_model(model.image_encoder, rank=4, alpha=4)

    sep_records = compute_per_image_separability(model, predictor, val_ds, device)
    finetune_ckpt = Path(finetune_ckpt)
    if not finetune_ckpt.exists():
        raise FileNotFoundError(f"Missing finetuned checkpoint: {finetune_ckpt}")

    head = load_sam2_ler_head(finetune_ckpt, model, len(meta["class_id_to_name"]), device)
    iou_records = compute_per_image_finetuned_iou(
        model, predictor, head, val_ds, device, meta["class_id_to_name"],
    )
    return merge_and_correlate(sep_records, iou_records)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/NEU-Seg")
    p.add_argument("--finetune_ckpt", default="outputs/thesis_validation/sam2_ler_100/best_model.pth")
    p.add_argument("--output", default="outputs/thesis_validation/per_image_asi.json")
    args = p.parse_args()
    merged, corr = run_per_image_analysis(args.data_dir, finetune_ckpt=args.finetune_ckpt)
    out = {"per_image": merged, "per_image_spearman": corr}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(corr, indent=2))
