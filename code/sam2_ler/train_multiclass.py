"""
train_multiclass.py — 多类语义分割 (4类: bg/patches/inclusion/scratches)

核心假设验证:
  二值分割中 Conv-LoRA ≈ LoRA (因为不需要区分缺陷类型)
  多类分割中 Conv-LoRA > LoRA (因为区分缺陷类型需要纹理特征)

架构:
  Image → [SAM2 Encoder + Conv-LoRA/LoRA] → multi-scale features
       → [Multi-Class FPN Head] → 4-class prediction

用法:
  # Conv-LoRA (实验组)
  python train_multiclass.py --lora_type conv --output_dir outputs/mc_conv_lora
  # 标准 LoRA (对照组)
  python train_multiclass.py --lora_type standard --output_dir outputs/mc_std_lora
"""

import os, json, time, argparse, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from tqdm import tqdm

from dataset import get_dataset, get_dataloader, get_dataset_meta
from train import build_sam2_model, inject_lora_to_model
from conv_lora import inject_conv_lora, print_mix_weights
from peft_variants import inject_peft
from spectral_head import SpectralResidualHead
from segmentation_metrics import SegmentationMetricAccumulator, compute_multiclass_metrics


def set_seed(seed: int):
    """统一设置所有 RNG, 保证跨运行可复现 (init / shuffle / 采样)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark=True 提速; 不强制 deterministic (AMP+conv 下会极慢).
    # 种子已固定初始化与数据顺序这两个主要方差来源, 跨 seed 提供真实方差估计.
    torch.backends.cudnn.benchmark = True


# ============================================================
#  多类分割头 (替代 SAM2 的二值 mask decoder)
# ============================================================
class SpatialDecodeAdapter(nn.Module):
    """Probe-B (DSA-FPN): 高分辨率解码支路上的轻量空间残差适配 (~8K params)."""

    def __init__(self, channels=64):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.norm(F.gelu(self.pw(self.dw(x))))


class MultiClassFPNHead(nn.Module):
    """
    轻量级 FPN 分割头, 融合 SAM2 encoder 的多尺度特征.

    特征输入:
      image_embed:    (B, 256, 64, 64)    — 低分辨率高语义
      high_res_feat1: (B, 64, 128, 128)   — 中分辨率
      high_res_feat0: (B, 32, 256, 256)   — 高分辨率高细节

    通过逐级上采样 + 侧向融合, 生成 (B, num_classes, 256, 256) 的预测.
    """

    def __init__(self, embed_dim=256, hr_dims=None, num_classes=4, use_dsa=False):
        super().__init__()
        if hr_dims is None:
            hr_dims = [32, 64]
        self.use_dsa = use_dsa

        # Stage 3: embed (256, 64, 64) → (128, 128, 128)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        # 融合 hr1 (64ch) + up3 (128ch) → 128ch
        self.fuse2 = nn.Sequential(
            nn.Conv2d(128 + hr_dims[1], 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # Stage 2: (128, 128, 128) → (64, 256, 256)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        # 融合 hr0 (32ch) + up2 (64ch) → 64ch
        self.fuse1 = nn.Sequential(
            nn.Conv2d(64 + hr_dims[0], 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.spatial_adapter = SpatialDecodeAdapter(64) if use_dsa else None

        # 分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, image_embed, high_res_feats):
        """
        Args:
            image_embed: (B, C, H, W) = (B, 256, 64, 64)
            high_res_feats: [(B, 32, 256, 256), (B, 64, 128, 128)]
        Returns:
            logits: (B, num_classes, 256, 256)
        """
        hr0, hr1 = high_res_feats[0], high_res_feats[1]

        # 64→128, 融合 hr1
        x = self.up3(image_embed)
        x = self.fuse2(torch.cat([x, hr1], dim=1))

        # 128→256, 融合 hr0
        x = self.up2(x)
        x = self.fuse1(torch.cat([x, hr0], dim=1))
        if self.spatial_adapter is not None:
            x = self.spatial_adapter(x)

        return self.classifier(x)


# ============================================================
#  多类损失函数
# ============================================================
class MultiClassLoss(nn.Module):
    """CrossEntropy + Multi-class Dice; 可选 rarity 加权 / boundary 辅助 (Probe/RAD)."""

    _LAP_K = None  # lazy: device-specific Laplacian for boundary loss

    def __init__(
        self,
        num_classes=4,
        ce_weight=0.5,
        dice_weight=0.5,
        loss_mode="baseline",
        class_weights=None,
        boundary_weight=0.2,
        boundary_edge_scale=5.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.loss_mode = loss_mode
        self.boundary_weight = boundary_weight
        self.boundary_edge_scale = boundary_edge_scale
        self.use_boundary = loss_mode in ("boundary", "rarity_boundary")
        self.use_rarity = loss_mode in ("rarity", "rarity_boundary")

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
        elif self.use_rarity:
            w = torch.ones(num_classes)
        else:
            w = None
        self.register_buffer("class_weights", w)

    def _ce(self, logits, targets, reduction="mean"):
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None
        return F.cross_entropy(logits, targets, weight=w, reduction=reduction)

    @staticmethod
    def compute_rarity_weights(dataset, train_idx, num_classes, eps=1e-6):
        """Inverse-sqrt pixel frequency weights (long-tail; RAD Probe-1)."""
        counts = np.zeros(num_classes, dtype=np.float64)
        for i in train_idx:
            mask = dataset[i]["mask"].numpy()
            for c in range(num_classes):
                counts[c] += (mask == c).sum()
        freq = counts / max(counts.sum(), 1.0)
        w = 1.0 / np.sqrt(freq + eps)
        w = w / w.mean()
        return w.tolist()

    def _boundary_weighted_ce(self, logits, targets):
        if MultiClassLoss._LAP_K is None or MultiClassLoss._LAP_K.device != targets.device:
            lap = torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
                device=targets.device,
            ).view(1, 1, 3, 3)
            MultiClassLoss._LAP_K = lap
        target_f = targets.float().unsqueeze(1)
        edge = F.conv2d(target_f, MultiClassLoss._LAP_K, padding=1).abs().clamp(0, 1).squeeze(1)
        ce_map = self._ce(logits, targets, reduction="none")
        return (ce_map * (1.0 + self.boundary_edge_scale * edge)).mean()

    def forward(self, logits, targets):
        """
        logits:  (B, C, H, W)
        targets: (B, H, W) long, 值为 0~num_classes-1
        """
        if self.use_boundary:
            ce_loss = self._boundary_weighted_ce(logits, targets)
        else:
            ce_loss = self._ce(logits, targets)

        probs = F.softmax(logits, dim=1)
        dice_sum = 0.0
        count = 0
        for c in range(1, self.num_classes):
            pred_c = probs[:, c]
            gt_c = (targets == c).float()
            if gt_c.sum() == 0:
                continue
            inter = (pred_c * gt_c).sum()
            dice = (2 * inter + 1e-6) / (pred_c.sum() + gt_c.sum() + 1e-6)
            dice_sum += dice
            count += 1

        dice_loss = 1.0 - dice_sum / max(count, 1)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# ============================================================
#  多类指标 (见 segmentation_metrics.py; 验证用 global pooled)
# ============================================================
# compute_multiclass_metrics 已从 segmentation_metrics 导入


# ============================================================
#  前向传播 (只用 encoder, 不用 SAM2 的 mask decoder)
# ============================================================
def encoder_forward(model, predictor, image_tensor):
    """
    只跑 SAM2 的 image encoder, 返回多尺度特征.
    Conv-LoRA 在这里自动生效.
    """
    device = image_tensor.device
    img_batch = image_tensor.unsqueeze(0) * 255.0
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
    input_image = (img_batch - pixel_mean) / pixel_std

    backbone_out = model.forward_image(input_image)
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
    if getattr(model, "directly_add_no_mem_embed", False):
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

    bb_feat_sizes = predictor._bb_feat_sizes
    feats = [
        feat.permute(1, 2, 0).view(1, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
    ][::-1]

    image_embed = feats[-1][-1].unsqueeze(0)    # (1, 256, 64, 64)
    hr0 = feats[0][-1].unsqueeze(0)             # (1, 32, 256, 256)
    hr1 = feats[1][-1].unsqueeze(0) if len(feats) > 2 else None  # (1, 64, 128, 128)

    high_res = [hr0]
    if hr1 is not None:
        high_res.append(hr1)

    return image_embed, high_res


# ============================================================
#  统一前向 (兼容基线 FPN 头与 SpectralResidualHead)
# ============================================================
def run_head(head, embed, hr_feats, image=None):
    """返回 (logits, aux). FPN 头 aux=None; Spectral 头返回 (logits, aux)."""
    if isinstance(head, SpectralResidualHead):
        logits, aux = head(embed, hr_feats, image=image)
        return logits, aux
    return head(embed, hr_feats), None


# ============================================================
#  训练/验证
# ============================================================
def train_one_epoch(model, predictor, head, dataloader, optimizer, scaler, loss_fn, device, epoch,
                    num_classes=4, class_id_to_name=None, aux_weight=0.0):
    model.train()
    head.train()
    total_loss = total_miou = 0.0
    n = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch["image"].to(device)
        gt_masks = batch["mask"].to(device)  # (B, H, W) 多类标签

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            batch_loss = 0.0
            batch_miou = 0.0
            for i in range(images.shape[0]):
                embed, hr_feats = encoder_forward(model, predictor, images[i])
                logits, aux = run_head(head, embed, hr_feats, image=images[i].unsqueeze(0))

                # Resize gt_mask to match logits
                gt_i = gt_masks[i].unsqueeze(0)  # (1, H, W)
                if gt_i.shape[-2:] != logits.shape[-2:]:
                    gt_i = F.interpolate(
                        gt_i.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest"
                    ).squeeze(1).long()

                loss = loss_fn(logits, gt_i)
                if aux is not None and aux_weight > 0:
                    loss = loss + aux_weight * loss_fn(aux, gt_i)
                batch_loss += loss
                miou, _ = compute_multiclass_metrics(
                    logits.detach(), gt_i, num_classes=num_classes, class_id_to_name=class_id_to_name,
                )
                batch_miou += miou

            batch_loss /= images.shape[0]

        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += batch_loss.item()
        total_miou += batch_miou / images.shape[0]
        n += 1
        pbar.set_postfix(loss=f"{total_loss/n:.4f}", miou=f"{total_miou/n:.4f}")
    return total_loss / n, total_miou / n


@torch.no_grad()
def validate(model, predictor, head, dataloader, loss_fn, device, num_classes=4, class_id_to_name=None):
    model.eval()
    head.eval()
    total_loss = 0.0
    num = 0
    acc = SegmentationMetricAccumulator(num_classes, class_id_to_name)

    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["image"].to(device)
        gt_masks = batch["mask"].to(device)
        for i in range(images.shape[0]):
            embed, hr_feats = encoder_forward(model, predictor, images[i])
            logits, _ = run_head(head, embed, hr_feats, image=images[i].unsqueeze(0))

            gt_i = gt_masks[i].unsqueeze(0)
            if gt_i.shape[-2:] != logits.shape[-2:]:
                gt_i = F.interpolate(
                    gt_i.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest"
                ).squeeze(1).long()

            total_loss += loss_fn(logits, gt_i).item()
            acc.update_logits(logits, gt_i)
            num += 1

    metrics = acc.compute()
    per_class = metrics["per_class_global"]
    vl_miou = metrics["miou_global"]
    return total_loss / max(num, 1), vl_miou, per_class, metrics


# ============================================================
#  主函数
# ============================================================
def main(args):
    device = torch.device("cuda")
    set_seed(args.seed)
    meta = get_dataset_meta(args.dataset)
    num_classes = args.num_classes or len(meta["class_id_to_name"])
    class_id_to_name = meta["class_id_to_name"]
    defect_names = meta["defect_classes"]

    print(f"\n{'='*60}")
    print(f"  多类语义分割: SAM2 Encoder + PEFT={args.lora_type.upper()} + HEAD={args.head_type.upper()}")
    print(f"  数据集: {args.dataset} | 类别: background / {' / '.join(defect_names)}")
    print(f"  seed: {args.seed} | GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data (val_loader 不 shuffle, 完全确定; train_loader 用 seed 固定 shuffle 顺序)
    train_ds = get_dataset(args.dataset, args.data_dir, split="train", img_size=args.img_size)
    val_ds = get_dataset(args.dataset, args.data_dir, split="val", img_size=args.img_size)
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = get_dataloader(val_ds, batch_size=1, shuffle=False)

    # Model
    model, predictor = build_sam2_model(args.checkpoint, args.model_cfg, device=device)
    for p in model.parameters(): p.requires_grad = False

    # PEFT 变体 (none = 零 PEFT 控制组)
    lora_params = inject_peft(
        model.image_encoder, args.lora_type, rank=args.lora_rank, alpha=args.lora_alpha,
    )

    # 分割头: fpn (基线) 或 spectral (FMRA 频率创新)
    if args.head_type == "spectral":
        head = SpectralResidualHead(
            embed_dim=256, hr_dims=[32, 64], num_classes=num_classes,
        ).to(device)
    else:
        head = MultiClassFPNHead(embed_dim=256, hr_dims=[32, 64], num_classes=num_classes).to(device)
    head_params = list(head.parameters())
    aux_weight = args.aux_weight if args.head_type == "spectral" else 0.0

    # 统计
    lc = sum(p.numel() for p in lora_params)
    hc = sum(p.numel() for p in head_params)
    total_trainable = lc + hc
    print(f"\n  PEFT({args.lora_type}): {lc:,} | Head({args.head_type}): {hc:,} | "
          f"Total: {total_trainable:,} ({total_trainable/1e6:.2f}M)\n")

    # 守护检查: 含 conv 的变体必须真正激活 conv 路径 (fail-fast, 防止死代码)
    if args.lora_type in ("conv", "map", "phase", "rank_adapt"):
        from conv_lora import verify_conv_active
        with torch.no_grad(), torch.amp.autocast("cuda"):
            sample = val_ds[0]["image"].to(device)
            _ = encoder_forward(model, predictor, sample)
        verify_conv_active(model.image_encoder, raise_on_fail=True)

    # Optimizer (零 PEFT 时不加入空的 lora 参数组)
    param_groups = [{"params": head_params, "lr": args.lr * 2.0}]
    if len(lora_params) > 0:
        param_groups.insert(0, {"params": lora_params, "lr": args.lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)
    class_weights = None
    if args.loss_mode in ("rarity", "rarity_boundary"):
        class_weights = MultiClassLoss.compute_rarity_weights(
            train_ds, list(range(len(train_ds))), num_classes,
        )
        print(f"  Rarity class weights: {[f'{w:.3f}' for w in class_weights]}")
    loss_fn = MultiClassLoss(
        num_classes=num_classes,
        loss_mode=args.loss_mode,
        class_weights=class_weights,
        boundary_weight=args.boundary_weight,
    )
    scaler = GradScaler("cuda")

    best_miou = 0.0
    best_epoch = 0
    patience_counter = 0
    results = []
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_miou = train_one_epoch(
            model, predictor, head, train_loader, optimizer, scaler, loss_fn, device, epoch,
            num_classes=num_classes, class_id_to_name=class_id_to_name, aux_weight=aux_weight,
        )
        vl_loss, vl_miou, pc, val_metrics = validate(
            model, predictor, head, val_loader, loss_fn, device,
            num_classes=num_classes, class_id_to_name=class_id_to_name,
        )
        scheduler.step()

        pc_str = " | ".join(
            f"{k}: {v:.4f}" for k, v in pc.items()
            if k != "background" and not (isinstance(v, float) and np.isnan(v))
        )
        print(
            f"Epoch {epoch}/{args.epochs} | Train mIoU: {tr_miou:.4f} | "
            f"Val mIoU(global): {vl_miou:.4f} | present: {val_metrics['miou_present']:.4f} | "
            f"{pc_str} | {time.time()-t0:.0f}s"
        )

        results.append({
            "epoch": epoch,
            "train_miou": tr_miou,
            "val_miou": vl_miou,
            "val_miou_present": val_metrics["miou_present"],
            "per_class": pc,
            "per_class_present": val_metrics["per_class_present"],
        })

        improved = vl_miou > best_miou + args.early_stop_min_delta
        if improved:
            best_miou = vl_miou
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch, "val_miou": vl_miou, "per_class": pc,
                "val_metrics": val_metrics,
                "head_state_dict": head.state_dict(),
                "encoder_lora": {
                    n: p.data.clone()
                    for n, p in model.image_encoder.named_parameters() if p.requires_grad
                },
            }, output_dir / "best_model.pth")
            print(f"  ✅ Best mIoU (global): {vl_miou:.4f} @ epoch {epoch}")
        elif epoch >= args.early_stop_min_epochs:
            patience_counter += 1
            if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                print(
                    f"  ⏹ Early stopping: {patience_counter} epochs无提升 "
                    f"(best={best_miou:.4f} @ ep{best_epoch})"
                )
                stopped_early = True
                break

    if args.lora_type in ("conv", "map", "phase", "rank_adapt"):
        print_mix_weights(model.image_encoder)

    with open(output_dir / "training_log.json", "w") as f:
        json.dump({
            "config": vars(args),
            "best_miou": best_miou,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "metric_mode": "global_pooled_iou",
            "trainable_params": total_trainable,
            "results": results,
        }, f, indent=2, default=float)

    print(f"\n{'='*60}")
    print(f"  完成! Best Val mIoU (global pooled): {best_miou:.4f} @ epoch {best_epoch}")
    print(f"  PEFT={args.lora_type} | Head={args.head_type} | seed={args.seed}")
    if stopped_early:
        print(f"  (早停于 epoch {len(results)})")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="neu_seg", choices=["neu_seg", "severstal"])
    p.add_argument("--data_dir", default="data/NEU-Seg")
    p.add_argument("--num_classes", type=int, default=None,
                   help="默认从 dataset meta 推断")
    p.add_argument("--img_size", type=int, default=1024)
    p.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt")
    p.add_argument("--model_cfg", default="sam2.1_hiera_b+.yaml")
    p.add_argument("--lora_type", default="standard",
                   choices=["none", "standard", "conv", "map", "phase", "rank_adapt"],
                   help="none=零PEFT控制组 / Std / Conv / MAP / Phase / Rank")
    p.add_argument("--head_type", default="fpn", choices=["fpn", "spectral"],
                   help="fpn=基线分割头 / spectral=FMRA 频率创新头")
    p.add_argument("--aux_weight", type=float, default=0.4,
                   help="spectral 头的辅助(深监督)损失权重")
    p.add_argument("--loss_mode", default="baseline",
                   choices=["baseline", "rarity", "boundary", "rarity_boundary"],
                   help="baseline | rarity (Probe-1) | boundary (Probe-2) | both")
    p.add_argument("--boundary_weight", type=float, default=0.2,
                   help="boundary 模式下的额外 CE 权重 (DDSNet-style)")
    p.add_argument("--seed", type=int, default=0, help="全局随机种子 (可复现 + 多seed方差)")
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--early_stop_patience", type=int, default=7,
                   help="验证 mIoU 无提升则停止; 0=禁用")
    p.add_argument("--early_stop_min_epochs", type=int, default=3,
                   help="至少训练 N epoch 再允许早停")
    p.add_argument("--early_stop_min_delta", type=float, default=0.001,
                   help="视为提升的最小 mIoU 增量")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--output_dir", default="outputs/multiclass_conv_lora")
    main(p.parse_args())
