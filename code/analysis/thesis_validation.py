"""
thesis_validation.py — 论文命题最快验证 (Go / No-Go)

核心命题 (ASI 理论):
  H1: ASI 越高 → 微调后 per-class IoU 越高
  H2: ASI 越低 → Conv-LoRA 相对 Std-LoRA 的增益越大
  H3: patches (最低 ASI) 的 adaptation gap 最大 (linear_probe_iou - finetuned_iou)

执行顺序 (由快到慢):
  Phase A  analyze   ~2 min  图像级 ASI + 现有 IoU 相关性 (不训练)
  Phase B  train      ~50 min 同架构 Conv vs Std 快速对比 (默认 10 epoch)
  Phase C  verdict    自动输出 GO / MAYBE / NO-GO

用法:
  # 第一步: 仅分析 (立刻判断 H1/H3 是否成立)
  python thesis_validation.py --mode analyze

  # 第二步: 快速训练 + 完整验证 (判断 H2, 论文最关键)
  python thesis_validation.py --mode full --epochs 10

  # 已有训练 log 时跳过训练
  python thesis_validation.py --mode analyze \\
      --conv_log outputs/multiclass_conv_lora/training_log.json \\
      --std_log outputs/multiclass_std_lora/training_log.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from asi_metrics import (
    DEFECT_CLASSES,
    compute_asi_table,
    pearson_correlation,
    spearman_correlation,
)
from dataset import get_dataset, get_dataset_meta
from run_asi_experiments import load_best_per_class_iou, pool_feature_maps
from train import build_sam2_model
from train_multiclass import encoder_forward


# ---------------------------------------------------------------------------
# Image-level ASI (NEU-Seg 每图单类, 比 pixel-level 更干净)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_image_level_features(model, predictor, dataset, device, feat_level="fused"):
    """每张图提取一个 defect mean-pooled 特征向量 + 类别标签."""
    feats, labels, paths = [], [], []

    for idx in tqdm(range(len(dataset)), desc="Image-level features"):
        sample = dataset[idx]
        image = sample["image"].to(device)
        mask = sample["mask"]

        defect_vals = mask[mask > 0]
        if len(defect_vals) == 0:
            continue
        class_id = int(defect_vals.mode().values.item())

        embed, hr = encoder_forward(model, predictor, image)
        if feat_level == "embed":
            feat_map = embed
        elif feat_level == "hr0":
            feat_map = F.interpolate(hr[0], size=embed.shape[-2:], mode="bilinear", align_corners=False)
        else:
            feat_map = pool_feature_maps(embed, hr, target_size=embed.shape[-2:])

        mask_small = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=feat_map.shape[-2:],
            mode="nearest",
        ).squeeze().bool()

        if mask_small.sum() == 0:
            continue

        fmap = feat_map.squeeze(0).permute(1, 2, 0)  # (H, W, C)
        mean_feat = fmap[mask_small].mean(dim=0).float().cpu().numpy()

        feats.append(mean_feat)
        labels.append(class_id)
        paths.append(sample.get("image_path", str(idx)))

    return np.stack(feats), np.array(labels, dtype=np.int64), paths


def load_asi_report(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def asi_dict_from_table(asi_table: dict) -> dict[str, float]:
    return {r["class_name"]: r["asi"] for r in asi_table["per_class"]}


def per_class_delta(conv_iou: dict, std_iou: dict, defect_classes: list[str] | None = None) -> dict[str, float]:
    classes = defect_classes or DEFECT_CLASSES
    return {c: conv_iou[c] - std_iou[c] for c in classes if c in conv_iou and c in std_iou}


def adaptation_gap(linear_probe: dict, finetuned: dict, defect_classes: list[str] | None = None) -> dict[str, float]:
    classes = defect_classes or DEFECT_CLASSES
    return {c: linear_probe[c] - finetuned[c] for c in classes if c in linear_probe and c in finetuned}


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def evaluate_verdict(
    pixel_asi: dict[str, float],
    image_asi: dict[str, float],
    iou: dict[str, float],
    asi_iou_spearman: float,
    image_asi_iou_spearman: float,
    delta_conv_std: dict[str, float] | None,
    adapt_gap: dict[str, float],
) -> dict:
    """
    判定规则 (保守):
      GO    — H1 图像级 ASI-IoU 趋势成立 + H2 patches 的 Conv 增益 ≥ scratches
      MAYBE — 部分成立, 需 ASI-guided 方法或更多实验
      NO-GO — H2 反向 (Std-LoRA 在 patches 上更好) 或 ASI 与 IoU 完全无关
    """
    checks = {}

    # H1a: patches 最低 ASI
    checks["h1_patches_lowest_asi"] = (
        pixel_asi.get("patches", 1) <= pixel_asi.get("inclusion", 0)
        and pixel_asi.get("patches", 1) <= pixel_asi.get("scratches", 0)
    )

    # H1b: patches 最低 IoU
    checks["h1_patches_lowest_iou"] = (
        iou.get("patches", 1) <= iou.get("inclusion", 0)
        and iou.get("patches", 1) <= iou.get("scratches", 0)
    )

    # H1c: 图像级 ASI 与 IoU 正相关 (3 点 Spearman, 阈值放宽)
    checks["h1_image_asi_iou_corr"] = image_asi_iou_spearman >= 0.5

    # H3: patches adaptation gap 最大
    if adapt_gap:
        checks["h3_patches_largest_gap"] = adapt_gap.get("patches", 0) >= max(
            adapt_gap.get("inclusion", 0), adapt_gap.get("scratches", 0)
        )

    # H2: Conv-LoRA 对低 ASI 类增益更大 → delta(patches) >= delta(scratches)
    if delta_conv_std:
        checks["h2_conv_helps_patches_more"] = (
            delta_conv_std.get("patches", -999) >= delta_conv_std.get("scratches", 999)
        )
        checks["h2_conv_positive_on_patches"] = delta_conv_std.get("patches", 0) > 0
        delta_vals = [delta_conv_std[c] for c in DEFECT_CLASSES if c in delta_conv_std]
        asi_vals = [pixel_asi[c] for c in DEFECT_CLASSES if c in delta_conv_std]
        if len(delta_vals) >= 2:
            # 低 ASI → 高 delta: 负相关
            checks["h2_delta_asi_corr"] = spearman_correlation(
                np.array(asi_vals), np.array(delta_vals)
            ) <= -0.5
        else:
            checks["h2_delta_asi_corr"] = None

    # Overall verdict
    n_pass = sum(1 for v in checks.values() if v is True)
    n_fail = sum(1 for v in checks.values() if v is False)

    if delta_conv_std is None:
        if checks.get("h1_patches_lowest_asi") and checks.get("h1_patches_lowest_iou"):
            verdict = "MAYBE"
            reason = "H1/H3 部分成立, 但缺少 Conv vs Std 同架构对比 (H2), 请运行 --mode full"
        else:
            verdict = "NO-GO"
            reason = "基础 ASI-IoU 现象不成立, 需重新审视理论"
    elif checks.get("h2_conv_helps_patches_more") and checks.get("h2_conv_positive_on_patches"):
        if checks.get("h1_image_asi_iou_corr") or checks.get("h1_patches_lowest_iou"):
            verdict = "GO"
            reason = "ASI 预测 IoU 趋势成立, 且 Conv-LoRA 对低 ASI 类 (patches) 增益最大"
        else:
            verdict = "MAYBE"
            reason = "H2 成立但 H1 相关性偏弱, 论文可写但需 image-level ASI + 更多分析"
    elif n_fail >= 2:
        verdict = "NO-GO"
        reason = "Conv-LoRA 未在低 ASI 类上展现预期优势, 需调整方法 (ASI-guided) 或理论"
    else:
        verdict = "MAYBE"
        reason = "结果混合, 建议 ASI-guided Conv-LoRA 或增加 epoch"

    return {
        "verdict": verdict,
        "reason": reason,
        "checks": checks,
        "n_pass": n_pass,
        "n_fail": n_fail,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_thesis_summary(
    output_path: Path,
    pixel_asi: dict,
    image_asi: dict,
    iou: dict,
    delta: dict | None,
    adapt_gap: dict | None,
):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # (0,0) ASI vs IoU
    names = DEFECT_CLASSES
    ax = axes[0, 0]
    img_asi = [image_asi.get(n, 0) for n in names]
    ious = [iou.get(n, 0) for n in names]
    ax.scatter(img_asi, ious, s=120, c="#2563eb", zorder=3)
    for n, a, i in zip(names, img_asi, ious):
        ax.annotate(n, (a, i), xytext=(6, 4), fontsize=9)
    rho = spearman_correlation(np.array(img_asi), np.array(ious))
    ax.set_xlabel("Image-level ASI")
    ax.set_ylabel("Finetuned IoU")
    ax.set_title(f"H1: ASI vs IoU (Spearman ρ={rho:.2f})")
    ax.grid(True, alpha=0.3)

    # (0,1) Per-class ASI pixel vs image
    ax = axes[0, 1]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [pixel_asi.get(n, 0) for n in names], 0.4, label="Pixel ASI")
    ax.bar(x + 0.2, [image_asi.get(n, 0) for n in names], 0.4, label="Image ASI")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("ASI: pixel vs image level")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (1,0) Conv - Std delta
    ax = axes[1, 0]
    if delta:
        dvals = [delta.get(n, 0) for n in names]
        colors = ["#22c55e" if v > 0 else "#ef4444" for v in dvals]
        ax.bar(names, dvals, color=colors)
        ax.axhline(0, color="#64748b", linewidth=0.8)
        ax.set_title("H2: ΔIoU (Conv-LoRA − Std-LoRA)")
    else:
        ax.text(0.5, 0.5, "Run --mode full\nfor H2", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("H2: pending")
    ax.grid(axis="y", alpha=0.3)

    # (1,1) Adaptation gap
    ax = axes[1, 1]
    if adapt_gap:
        ax.bar(names, [adapt_gap.get(n, 0) for n in names], color="#8b5cf6")
        ax.set_title("H3: Linear-probe IoU − Finetuned IoU")
    else:
        ax.set_title("H3: N/A")
    ax.set_ylabel("Gap")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------

def run_multiclass_training(
    lora_type: str,
    output_dir: Path,
    epochs: int,
    data_dir: str,
    python: str,
    dataset: str = "neu_seg",
) -> Path:
    """调用 train_multiclass.py, 返回 training_log.json 路径."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.json"
    if log_path.exists():
        print(f"  [skip] 已存在 {log_path}")
        return log_path

    cmd = [
        python, "train_multiclass.py",
        "--lora_type", lora_type,
        "--output_dir", str(output_dir),
        "--data_dir", data_dir,
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--batch_size", "4",
        "--lr", "1e-4",
    ]
    print(f"\n  >>> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    return log_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    meta = get_dataset_meta(args.dataset)
    defect_classes = meta["defect_classes"]
    class_id_to_name = meta["class_id_to_name"]
    defect_class_ids = meta["defect_class_ids"]

    print(f"\n{'='*65}")
    print("  Thesis Validation (ASI Go/No-Go)")
    print(f"  Mode: {args.mode}")
    print(f"{'='*65}")

    # ---- Load existing pixel ASI + linear probe ----
    asi_report_path = Path(args.asi_report)
    if not asi_report_path.exists():
        raise FileNotFoundError(
            f"缺少 ASI 报告: {asi_report_path}\n"
            f"请先运行: python run_asi_experiments.py --save_features ..."
        )
    asi_report = load_asi_report(asi_report_path)
    pixel_asi = asi_dict_from_table(asi_report["asi_table"])
    linear_probe = asi_report.get("linear_probe", {}).get("per_class", {})

    # ---- Finetuned IoU source ----
    iou_path = Path(args.iou_log)
    iou = load_best_per_class_iou(iou_path)
    print(f"\n[IoU source] {iou_path}")
    for c in defect_classes:
        print(f"  {c}: IoU={iou.get(c, 0):.4f}")

    # ---- Image-level ASI ----
    print("\n[Phase A] 计算 image-level ASI (~2 min)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_ds = get_dataset(args.dataset, args.data_dir, split="val", img_size=args.img_size)
    model, predictor = build_sam2_model(args.checkpoint, args.model_cfg, device=device)
    for p in model.parameters():
        p.requires_grad = False

    img_feats, img_labels, _ = extract_image_level_features(
        model, predictor, val_ds, device, feat_level="fused",
    )
    image_asi_table = compute_asi_table(
        img_feats, img_labels,
        defect_class_ids=defect_class_ids,
        class_id_to_name=class_id_to_name,
    )
    image_asi = asi_dict_from_table(image_asi_table)
    print("  Image-level ASI:")
    for c in defect_classes:
        print(f"    {c}: {image_asi.get(c, 0):.4f}")

    pixel_asi_vals = [pixel_asi[c] for c in defect_classes if c in pixel_asi]
    image_asi_vals = [image_asi[c] for c in defect_classes if c in image_asi]
    iou_vals = [iou[c] for c in defect_classes if c in iou]

    # 只对三类指标都存在的缺陷类做相关性 (如 class4 在 val 像素采样中可能为 0)
    corr_classes = [
        c for c in defect_classes
        if c in pixel_asi and c in image_asi and c in iou
    ]
    corr_pixel = corr_image = float("nan")
    if len(corr_classes) >= 2:
        corr_pixel = spearman_correlation(
            np.array([pixel_asi[c] for c in corr_classes]),
            np.array([iou[c] for c in corr_classes]),
        )
        corr_image = spearman_correlation(
            np.array([image_asi[c] for c in corr_classes]),
            np.array([iou[c] for c in corr_classes]),
        )
    print(f"\n  ASI vs IoU Spearman: pixel={corr_pixel:.3f}, image={corr_image:.3f}")

    adapt_gap = adaptation_gap(linear_probe, iou, defect_classes) if linear_probe else {}
    if adapt_gap:
        print("\n  Adaptation gap (linear_probe - finetuned):")
        for c in defect_classes:
            print(f"    {c}: {adapt_gap.get(c, 0):.4f}")

    # ---- Phase B: Conv vs Std training ----
    delta_conv_std = None
    conv_log = Path(args.conv_log) if args.conv_log else output_dir / "multiclass_conv" / "training_log.json"
    std_log = Path(args.std_log) if args.std_log else output_dir / "multiclass_std" / "training_log.json"

    if args.mode in ("train", "full"):
        print(f"\n[Phase B] Conv vs Std 同架构对比 ({args.epochs} epochs)...")
        run_multiclass_training("conv", conv_log.parent, args.epochs, args.data_dir, python, args.dataset)
        run_multiclass_training("standard", std_log.parent, args.epochs, args.data_dir, python, args.dataset)

    if conv_log.exists() and std_log.exists():
        conv_iou = load_best_per_class_iou(conv_log)
        std_iou = load_best_per_class_iou(std_log)
        delta_conv_std = per_class_delta(conv_iou, std_iou, defect_classes)
        print("\n  H2: ΔIoU (Conv − Std):")
        for c in defect_classes:
            print(f"    {c}: conv={conv_iou.get(c,0):.4f} std={std_iou.get(c,0):.4f} "
                  f"delta={delta_conv_std.get(c,0):+.4f}")
    else:
        print(f"\n  [H2 pending] 缺少 training log:")
        print(f"    conv: {conv_log} ({'OK' if conv_log.exists() else 'MISSING'})")
        print(f"    std:  {std_log} ({'OK' if std_log.exists() else 'MISSING'})")

    # ---- Verdict ----
    verdict_info = evaluate_verdict(
        pixel_asi, image_asi, iou,
        corr_pixel, corr_image,
        delta_conv_std, adapt_gap,
    )

    print(f"\n{'='*65}")
    print(f"  VERDICT: {verdict_info['verdict']}")
    print(f"  {verdict_info['reason']}")
    print(f"  Checks: {verdict_info['checks']}")
    print(f"{'='*65}\n")

    # ---- Save ----
    report = {
        "verdict": verdict_info,
        "pixel_asi": pixel_asi,
        "image_asi": image_asi,
        "image_asi_table": image_asi_table,
        "finetuned_iou": iou,
        "correlation": {"pixel_asi_iou_spearman": corr_pixel, "image_asi_iou_spearman": corr_image},
        "adaptation_gap": adapt_gap,
        "delta_conv_std": delta_conv_std,
        "conv_log": str(conv_log) if conv_log.exists() else None,
        "std_log": str(std_log) if std_log.exists() else None,
    }
    report_path = output_dir / "thesis_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=float)

    plot_thesis_summary(
        output_dir / "fig_thesis_validation.png",
        pixel_asi, image_asi, iou, delta_conv_std, adapt_gap,
    )
    print(f"  报告: {report_path}")
    print(f"  图表: {output_dir / 'fig_thesis_validation.png'}")

    return verdict_info["verdict"]


def main():
    p = argparse.ArgumentParser(description="Thesis Go/No-Go validation")
    p.add_argument("--mode", choices=["analyze", "train", "full"], default="analyze",
                   help="analyze=仅 Phase A; train/full=含 Conv vs Std 训练")
    p.add_argument("--output_dir", default="outputs/thesis_validation")
    p.add_argument("--dataset", default="neu_seg", choices=["neu_seg", "severstal"])
    p.add_argument("--data_dir", default="data/NEU-Seg")
    p.add_argument("--img_size", type=int, default=1024)
    p.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt")
    p.add_argument("--model_cfg", default="sam2.1_hiera_b+.yaml")
    p.add_argument("--epochs", type=int, default=10,
                   help="Conv vs Std 快速对比的训练 epoch (默认 10, 完整可用 30)")
    p.add_argument("--asi_report", default="outputs/asi_analysis/exp1_frozen_fused/asi_report.json")
    p.add_argument("--iou_log", default="outputs/two_stage_conv/training_log.json",
                   help="当前最佳微调 IoU (用于 H1/H3)")
    p.add_argument("--conv_log", default=None, help="已有 conv multiclass log 时指定")
    p.add_argument("--std_log", default=None, help="已有 std multiclass log 时指定")
    run(p.parse_args())


if __name__ == "__main__":
    main()
