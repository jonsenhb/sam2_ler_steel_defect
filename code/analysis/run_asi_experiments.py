"""
run_asi_experiments.py — ASI (Adaptation Sufficiency Index) 完整实验流水线

实验内容:
  1. 从 frozen SAM2 encoder 提取像素级特征
  2. 计算每类 Fisher / Linear-AUC / Silhouette → 综合 ASI
  3. ASI vs 微调后 per-class IoU 相关性分析
  4. 多方法 IoU 对比 (从 training_log.json 读取)
  5. t-SNE 可视化 + 线性 probe 上界 (可选)

用法:
  # 基础 ASI 分析 (frozen SAM2, val 集)
  python run_asi_experiments.py \\
      --data_dir data/NEU-Seg \\
      --output_dir outputs/asi_analysis

  # 指定 IoU 来源做相关性
  python run_asi_experiments.py \\
      --iou_log outputs/two_stage_conv/training_log.json \\
      --compare_logs outputs/lora_r4/training_log.json,outputs/two_stage_conv/training_log.json

  # 对比 frozen vs Conv-LoRA 微调后 encoder 的 ASI (需 multiclass checkpoint)
  python run_asi_experiments.py \\
      --finetuned_ckpt outputs/multiclass_conv_lora/best_model.pth \\
      --lora_type conv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from asi_metrics import (
    DEFECT_CLASSES,
    CLASS_ID_TO_NAME,
    compute_asi_table,
    pearson_correlation,
    spearman_correlation,
)
from dataset import get_dataset, get_dataloader, get_dataset_meta
from train import build_sam2_model
from train_multiclass import encoder_forward, compute_multiclass_metrics, MultiClassFPNHead
from conv_lora import inject_conv_lora
from train import inject_lora_to_model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def pool_feature_maps(image_embed, high_res_feats, target_size=(64, 64)):
    """
    融合多尺度特征到统一空间分辨率 (默认 64×64, 与 image_embed 对齐).

    Returns:
        feat_map: (B, C, H, W)
    """
    embed = F.interpolate(image_embed, size=target_size, mode="bilinear", align_corners=False)
    parts = [embed]
    for hr in high_res_feats:
        parts.append(F.interpolate(hr, size=target_size, mode="bilinear", align_corners=False))
    return torch.cat(parts, dim=1)


@torch.no_grad()
def extract_pixel_features(
    model,
    predictor,
    dataloader,
    device,
    feat_level="fused",
    max_pixels_per_class=200,
    max_images=None,
):
    """
    从验证集提取像素级特征与标签.

    feat_level:
      - 'embed'  : 仅 image_embed (256-d @ 64×64)
      - 'hr0'    : 高分辨率特征 (32-d @ 256×256 → 下采样到 64×64)
      - 'fused'  : embed + hr0 + hr1 拼接 (默认, 推荐)

    每张图每类最多采样 max_pixels_per_class 个像素, 控制内存.
    """
    model.eval()
    all_features = []
    all_labels = []
    n_images = 0

    for batch in tqdm(dataloader, desc="Extracting features"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        for i in range(images.shape[0]):
            if max_images is not None and n_images >= max_images:
                break

            embed, hr = encoder_forward(model, predictor, images[i])
            mask = masks[i]

            if feat_level == "embed":
                feat_map = embed
                feat_h, feat_w = embed.shape[-2:]
            elif feat_level == "hr0":
                feat_map = F.interpolate(hr[0], size=mask.shape[-2:], mode="bilinear", align_corners=False)
                feat_h, feat_w = mask.shape[-2], mask.shape[-1]
            else:
                feat_map = pool_feature_maps(embed, hr, target_size=embed.shape[-2:])

            feat_h, feat_w = feat_map.shape[-2], feat_map.shape[-1]
            mask_small = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(feat_h, feat_w),
                mode="nearest",
            ).squeeze().long()

            feat = feat_map.squeeze(0).permute(1, 2, 0).reshape(-1, feat_map.shape[1])
            lbl = mask_small.reshape(-1).cpu().numpy()
            feat_np = feat.float().cpu().numpy()

            rng = np.random.default_rng(n_images)
            for class_id in [0, 1, 2, 3]:
                idx = np.where(lbl == class_id)[0]
                if len(idx) == 0:
                    continue
                if len(idx) > max_pixels_per_class:
                    idx = rng.choice(idx, max_pixels_per_class, replace=False)
                all_features.append(feat_np[idx])
                all_labels.append(lbl[idx])

            n_images += 1

        if max_images is not None and n_images >= max_images:
            break

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels, n_images


def load_finetuned_encoder(model, ckpt_path, lora_type, lora_rank, lora_alpha, device):
    """加载 multiclass 实验保存的 Conv-LoRA / Std-LoRA 权重."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if lora_type == "conv":
        inject_conv_lora(model.image_encoder, rank=lora_rank, alpha=lora_alpha)
    else:
        inject_lora_to_model(model.image_encoder, rank=lora_rank, alpha=lora_alpha)

    lora_state = ckpt.get("encoder_lora", {})
    if lora_state:
        current = {n: p for n, p in model.image_encoder.named_parameters() if p.requires_grad}
        for name, tensor in lora_state.items():
            if name in current:
                current[name].data.copy_(tensor)
        print(f"[ASI] 已加载微调 encoder LoRA 权重 ({len(lora_state)} tensors)")
    else:
        print("[ASI] ⚠ checkpoint 中无 encoder_lora, 仍使用 frozen encoder")
    return ckpt


# ---------------------------------------------------------------------------
# IoU log utilities
# ---------------------------------------------------------------------------

def load_best_per_class_iou(log_path: str | Path) -> dict:
    """从 training_log.json 读取 best epoch 的 per-class IoU."""
    with open(log_path) as f:
        data = json.load(f)

    # 格式 A: results 列表
    if "results" in data and data["results"]:
        key_miou = "val_miou" if "val_miou" in data["results"][0] else "val_iou"
        best = max(data["results"], key=lambda r: r.get(key_miou, 0))
        if "per_class" in best:
            return {k: float(v) for k, v in best["per_class"].items() if k != "background"}
        if "per_class_iou" in best:
            return {k: float(v) for k, v in best["per_class_iou"].items()}

    # 格式 B: 顶层 per_class
    if "per_class" in data:
        return {k: float(v) for k, v in data["per_class"].items() if k != "background"}

    raise ValueError(f"无法从 {log_path} 解析 per-class IoU")


def load_method_name(log_path: Path) -> str:
    return log_path.parent.name


# ---------------------------------------------------------------------------
# Linear probe upper bound (pixel-wise multi-class)
# ---------------------------------------------------------------------------

@torch.no_grad()
def linear_probe_pixel_miou(
    features, labels, defect_only=True,
    defect_class_ids=None, class_id_to_name=None,
):
    """
    在 frozen 特征上训练线性多类分类器, 估计 pixel-wise mIoU 上界.
    使用 sklearn LogisticRegression, 子采样加速.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    max_samples = 30000
    rng = np.random.default_rng(42)
    n = len(labels)
    idx = np.arange(n) if n <= max_samples else rng.choice(n, max_samples, replace=False)
    x_all = features[idx]
    y_all = labels[idx]

    try:
        x_tr, x_te, y_tr, y_te = train_test_split(
            x_all, y_all, test_size=0.3, random_state=42, stratify=y_all,
        )
    except ValueError:
        return 0.0, {}

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_te = scaler.transform(x_te)

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    y = y_te

    if class_id_to_name is None:
        class_id_to_name = CLASS_ID_TO_NAME
    if defect_class_ids is None:
        defect_class_ids = [1, 2, 3]

    ious = {}
    classes = defect_class_ids if defect_only else [0] + defect_class_ids
    for c in classes:
        pred_c = pred == c
        gt_c = y == c
        inter = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        ious[class_id_to_name[c]] = float((inter + 1e-6) / (union + 1e-6))
    miou = float(np.mean(list(ious.values())))
    return miou, ious


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_asi_vs_iou(asi_table, iou_dict, output_path, title="ASI vs IoU"):
    """散点图: x=ASI, y=IoU (per defect class)."""
    names, asis, ious = [], [], []
    for row in asi_table["per_class"]:
        name = row["class_name"]
        if name not in iou_dict:
            continue
        names.append(name)
        asis.append(row["asi"])
        ious.append(iou_dict[name])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(asis, ious, s=120, c="#2563eb", edgecolors="white", linewidths=1.5, zorder=3)
    for name, a, i in zip(names, asis, ious):
        ax.annotate(name, (a, i), textcoords="offset points", xytext=(8, 6), fontsize=10)

    if len(asis) >= 2:
        z = np.polyfit(asis, ious, 1)
        xs = np.linspace(min(asis) - 0.05, max(asis) + 0.05, 50)
        ax.plot(xs, np.poly1d(z)(xs), "--", color="#94a3b8", linewidth=1.5, label="linear fit")

    r_pearson = pearson_correlation(np.array(asis), np.array(ious))
    r_spearman = spearman_correlation(np.array(asis), np.array(ious))
    ax.set_xlabel("ASI (Adaptation Sufficiency Index)", fontsize=11)
    ax.set_ylabel("Validation IoU (after fine-tuning)", fontsize=11)
    ax.set_title(f"{title}\nPearson r={r_pearson:.3f}, Spearman ρ={r_spearman:.3f}", fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return r_pearson, r_spearman


def plot_asi_bars(asi_table, output_path):
    """每类 ASI 分量柱状图."""
    rows = asi_table["per_class"]
    names = [r["class_name"] for r in rows]
    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, [r["fisher_norm"] for r in rows], width, label="Fisher (norm)")
    ax.bar(x, [r["auc_norm"] for r in rows], width, label="Linear AUC (norm)")
    ax.bar(x + width, [r["silhouette_norm"] for r in rows], width, label="Silhouette (norm)")

    for i, r in enumerate(rows):
        ax.text(i, r["asi"] + 0.02, f"ASI={r['asi']:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Normalized score")
    ax.set_title("Per-class ASI components")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_tsne(features, labels, output_path, max_points=4000):
    """t-SNE 可视化 frozen 特征 (按类着色)."""
    rng = np.random.default_rng(42)
    n = len(labels)
    idx = np.arange(n) if n <= max_points else rng.choice(n, max_points, replace=False)
    x = features[idx].astype(np.float32)
    y = labels[idx]

    x = PCA(n_components=min(50, x.shape[1]), random_state=42).fit_transform(x)
    emb = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto").fit_transform(x)

    colors = {0: "#cbd5e1", 1: "#3b82f6", 2: "#22c55e", 3: "#ef4444"}
    names = {0: "background", 1: "patches", 2: "inclusion", 3: "scratches"}

    fig, ax = plt.subplots(figsize=(7, 6))
    for cid in [0, 1, 2, 3]:
        m = y == cid
        if m.sum() == 0:
            continue
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.5, c=colors[cid], label=names[cid])

    ax.legend(markerscale=2, fontsize=9)
    ax.set_title("t-SNE of frozen SAM2 pixel features (val set)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_method_delta_iou(compare_results, asi_table, output_path):
    """
    对比多种方法的 per-class IoU 增益 vs ASI.
    compare_results: {method_name: {class: iou}}
    """
    if len(compare_results) < 2:
        return

    methods = list(compare_results.keys())
    base_method = methods[0]
    target_method = methods[-1]

    base = compare_results[base_method]
    target = compare_results[target_method]
    asi_map = {r["class_name"]: r["asi"] for r in asi_table["per_class"]}

    names, deltas, asis = [], [], []
    for name in DEFECT_CLASSES:
        if name in base and name in target and name in asi_map:
            names.append(name)
            deltas.append(target[name] - base[name])
            asis.append(asi_map[name])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(names))
    width = 0.25
    axes[0].bar(x, [compare_results[methods[0]].get(n, 0) for n in names], width=width, label=methods[0])
    for j, method in enumerate(methods[1:], 1):
        axes[0].bar(x + width * j, [compare_results[method].get(n, 0) for n in names], width=width, label=method)
    axes[0].set_xticks(x + 0.15)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("IoU")
    axes[0].set_title("Per-class IoU by method")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].scatter(asis, deltas, s=100, c="#7c3aed")
    for name, a, d in zip(names, asis, deltas):
        axes[1].annotate(name, (a, d), xytext=(6, 4), fontsize=9)
    axes[1].axhline(0, color="#94a3b8", linewidth=0.8)
    axes[1].set_xlabel("ASI")
    axes[1].set_ylabel(f"ΔIoU ({target_method} − {base_method})")
    axes[1].set_title("Low ASI → larger gain from stronger adaptation?")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_experiments(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = get_dataset_meta(args.dataset)
    defect_class_ids = meta["defect_class_ids"]
    class_id_to_name = meta["class_id_to_name"]
    defect_classes = meta["defect_classes"]

    print(f"\n{'='*65}")
    print("  ASI Experiment Pipeline")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"{'='*65}\n")

    # Dataset
    val_ds = get_dataset(args.dataset, args.data_dir, split="val", img_size=args.img_size)
    val_loader = get_dataloader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Model (skip heavy load if only re-analyzing cached features)
    model = predictor = None
    if not args.features_npz:
        model, predictor = build_sam2_model(args.checkpoint, args.model_cfg, device=device)
        for p in model.parameters():
            p.requires_grad = False

        encoder_mode = "frozen"
        if args.finetuned_ckpt:
            load_finetuned_encoder(
                model, args.finetuned_ckpt, args.lora_type,
                args.lora_rank, args.lora_alpha, device,
            )
            encoder_mode = f"finetuned_{args.lora_type}_lora"
    else:
        encoder_mode = "frozen"

    # ---- Step 1: Feature extraction ----
    if args.features_npz:
        print(f"\n[Step 1] 加载已保存特征: {args.features_npz}")
        data = np.load(args.features_npz, allow_pickle=True)
        features, labels = data["features"], data["labels"]
        n_images = int(data.get("n_images", -1))
        encoder_mode = str(data.get("encoder_mode", "frozen"))
        print(f"  像素样本: {len(labels):,}, 特征维: {features.shape[1]}, mode={encoder_mode}")
    else:
        print(f"\n[Step 1] 提取 {encoder_mode} encoder 特征 (level={args.feat_level})...")
        features, labels, n_images = extract_pixel_features(
            model, predictor, val_loader, device,
            feat_level=args.feat_level,
            max_pixels_per_class=args.max_pixels_per_class,
            max_images=args.max_images,
        )
        print(f"  图像数: {n_images}, 像素样本: {len(labels):,}, 特征维: {features.shape[1]}")

        for cid in defect_class_ids:
            name = class_id_to_name[cid]
            print(f"  {name}: {(labels == cid).sum():,} pixels")

        if args.save_features:
            np.savez_compressed(
                output_dir / "features_val.npz",
                features=features.astype(np.float32),
                labels=labels,
                n_images=n_images,
                encoder_mode=encoder_mode,
            )
            print(f"  特征已保存: {output_dir / 'features_val.npz'}")

    # ---- Step 2: ASI computation ----
    print("\n[Step 2] 计算 ASI...")
    asi_table = compute_asi_table(
        features, labels,
        defect_class_ids=defect_class_ids,
        class_id_to_name=class_id_to_name,
    )
    for row in asi_table["per_class"]:
        print(f"  {row['class_name']:12s} | ASI={row['asi']:.4f} | "
              f"Fisher={row['fisher']:.2f} AUC={row['auc']:.3f} Sil={row['silhouette']:.3f} | "
              f"n={row['n_pixels']:,}")
    print(f"  Mean ASI: {asi_table['mean_asi']:.4f}")

    # ---- Step 3: Linear probe upper bound ----
    if args.run_linear_probe:
        print("\n[Step 3] Linear probe 上界 (frozen features)...")
        lp_miou, lp_ious = linear_probe_pixel_miou(
            features, labels,
            defect_class_ids=defect_class_ids,
            class_id_to_name=class_id_to_name,
        )
        print(f"  Linear probe mIoU (defect): {lp_miou:.4f}")
        for k, v in lp_ious.items():
            print(f"    {k}: {v:.4f}")
    else:
        lp_miou, lp_ious = None, None

    # ---- Step 4: Correlation with fine-tuned IoU ----
    iou_dict = {}
    correlation = {}
    if args.iou_log:
        iou_path = Path(args.iou_log)
        if not iou_path.exists():
            print(f"\n[Step 4] 跳过 ASI-IoU 相关性 — 尚无 {args.iou_log}")
        else:
            print(f"\n[Step 4] ASI vs IoU 相关性 ({args.iou_log})...")
            iou_dict = load_best_per_class_iou(iou_path)
            for name in defect_classes:
                if name in iou_dict:
                    print(f"  {name}: IoU={iou_dict[name]:.4f}")
            r_p, r_s = plot_asi_vs_iou(
                asi_table, iou_dict,
                output_dir / "fig_asi_vs_iou.png",
                title=f"ASI vs IoU ({encoder_mode})",
            )
            correlation = {"pearson": r_p, "spearman": r_s}
            print(f"  Pearson r={r_p:.4f}, Spearman ρ={r_s:.4f}")

    # ---- Step 5: Multi-method comparison ----
    compare_results = {}
    if args.compare_logs:
        print("\n[Step 5] 多方法 IoU 对比...")
        for log_path in args.compare_logs.split(","):
            log_path = Path(log_path.strip())
            if not log_path.exists():
                print(f"  ⚠ 跳过不存在的 log: {log_path}")
                continue
            try:
                name = load_method_name(log_path)
                compare_results[name] = load_best_per_class_iou(log_path)
            except ValueError as e:
                print(f"  ⚠ 跳过 {log_path.name}: {e}")
                continue
            miou = np.mean(list(compare_results[name].values()))
            print(f"  {name}: mIoU={miou:.4f} | " +
                  " | ".join(f"{k}={v:.3f}" for k, v in compare_results[name].items()))
        if len(compare_results) >= 2:
            plot_method_delta_iou(compare_results, asi_table, output_dir / "fig_method_comparison.png")
        elif compare_results:
            print("  (仅 1 个有效 log, 跳过 ΔIoU 图)")

    # ---- Step 6: Visualizations ----
    print("\n[Step 6] 生成可视化...")
    plot_asi_bars(asi_table, output_dir / "fig_asi_components.png")
    if not args.skip_tsne:
        plot_tsne(features, labels, output_dir / "fig_tsne_features.png")

    # ---- Save JSON report ----
    report = {
        "config": vars(args),
        "encoder_mode": encoder_mode,
        "n_images": n_images,
        "n_pixel_samples": int(len(labels)),
        "feature_dim": int(features.shape[1]),
        "asi_table": asi_table,
        "linear_probe": {"miou": lp_miou, "per_class": lp_ious} if lp_miou else None,
        "finetuned_iou": iou_dict,
        "correlation": correlation,
        "method_comparison": compare_results,
    }
    report_path = output_dir / "asi_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=float)

    print(f"\n{'='*65}")
    print(f"  ✅ ASI 实验完成!")
    print(f"  报告: {report_path}")
    print(f"  图表: {output_dir}/fig_*.png")
    print(f"{'='*65}\n")

    return report


def main():
    p = argparse.ArgumentParser(description="ASI (Adaptation Sufficiency Index) experiments")
    p.add_argument("--dataset", default="neu_seg", choices=["neu_seg", "severstal"])
    p.add_argument("--data_dir", default="data/NEU-Seg")
    p.add_argument("--img_size", type=int, default=1024)
    p.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt")
    p.add_argument("--model_cfg", default="sam2.1_hiera_b+.yaml")
    p.add_argument("--output_dir", default="outputs/asi_analysis")

    p.add_argument("--feat_level", default="fused", choices=["embed", "hr0", "fused"],
                   help="特征层级: embed(256d) / hr0(32d@256) / fused(拼接, 推荐)")
    p.add_argument("--max_pixels_per_class", type=int, default=200,
                   help="每张图每类最多采样像素数")
    p.add_argument("--max_images", type=int, default=None,
                   help="最多处理多少张图 (调试用, 默认全部 val 集)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_features", action="store_true",
                   help="保存 features_val.npz 供后续复用")

    p.add_argument("--features_npz", default=None,
                   help="跳过特征提取, 直接加载已保存的 .npz")

    p.add_argument("--iou_log", default="outputs/two_stage_conv/training_log.json",
                   help="微调结果的 training_log.json, 用于 ASI-IoU 相关性")
    p.add_argument("--compare_logs", default="outputs/lora_r4/training_log.json,outputs/two_stage_conv/training_log.json",
                   help="逗号分隔的多个 log, 对比 per-class IoU")

    p.add_argument("--finetuned_ckpt", default=None,
                   help="multiclass best_model.pth, 对比微调后 encoder ASI")
    p.add_argument("--lora_type", default="conv", choices=["conv", "standard"])
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=4)

    p.add_argument("--run_linear_probe", action="store_true",
                   help="运行 linear probe 上界估计 (默认关闭, 较慢)")
    p.add_argument("--skip_tsne", action="store_true", help="跳过 t-SNE (加速)")

    args = p.parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
