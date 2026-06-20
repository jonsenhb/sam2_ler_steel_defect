"""
train.py — SAM2 + LoRA 微调训练脚本
用于钢材表面缺陷分割的快速可行性验证

核心思路:
  1. 冻结 SAM2 的 image_encoder 和 prompt_encoder
  2. 通过 LoRA 在 image_encoder 的注意力层注入少量可训练参数
  3. 完全微调 mask_decoder
  4. 训练时使用从 GT mask 中采样的 point prompt
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from tqdm import tqdm

# ---- 项目内模块 ----
from dataset import get_dataset, get_dataloader


# ============================================================
#  LoRA 注入工具
# ============================================================
class LoRALinear(nn.Module):
    """LoRA: Low-Rank Adaptation for a linear layer"""

    def __init__(self, original_linear, rank=4, alpha=4):
        super().__init__()
        self.original_linear = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA 矩阵 — 必须与原始权重在同一设备和相同 dtype
        device = original_linear.weight.device
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, device=device, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device, dtype=dtype))

        # 冻结原始权重
        for p in self.original_linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        original_out = self.original_linear(x)
        # 确保 LoRA 计算在正确的 dtype 下进行
        lora_out = (x.to(self.lora_A.dtype) @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out.to(original_out.dtype)


def inject_lora_to_model(model, rank=4, alpha=4, target_modules=None):
    """
    将 LoRA 注入到模型的线性层中
    target_modules: 匹配的模块名列表 (默认注入所有 attention 的 qkv 和 proj)
    """
    if target_modules is None:
        target_modules = ["qkv", "proj", "q_proj", "k_proj", "v_proj", "out_proj"]

    lora_params = []
    injected_count = 0

    for name, module in model.named_modules():
        # 只处理叶子线性层
        if not isinstance(module, nn.Linear):
            continue
        # 检查是否匹配目标模块名
        if not any(t in name for t in target_modules):
            continue

        # 获取父模块和属性名
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            attr_name = parts[0]
            parent = model

        # 替换为 LoRA 版本
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
        setattr(parent, attr_name, lora_layer)
        lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
        injected_count += 1

    print(f"[LoRA] 注入了 {injected_count} 个 LoRA 层, rank={rank}, alpha={alpha}")
    total_lora_params = sum(p.numel() for p in lora_params)
    print(f"[LoRA] 可训练 LoRA 参数: {total_lora_params:,} ({total_lora_params/1e6:.2f}M)")
    return lora_params


# ============================================================
#  SAM2 模型构建
# ============================================================
def build_sam2_model(checkpoint_path, model_cfg="sam2.1_hiera_b+.yaml", device="cuda"):
    """
    加载 SAM2 模型 — 手动管理 Hydra 初始化, 避免配置路径问题
    """
    import sam2
    from pathlib import Path

    sam2_pkg_dir = Path(sam2.__file__).parent
    configs_dir = sam2_pkg_dir / "configs"

    # ---- Step 1: 找到配置文件的实际路径 ----
    base_name = model_cfg.replace(".yaml", "")
    config_file = None
    # 优先精确匹配, 其次部分匹配
    all_yamls = sorted(configs_dir.rglob("*.yaml"))
    for yaml_path in all_yamls:
        if yaml_path.stem == base_name:
            config_file = str(yaml_path.relative_to(configs_dir)).replace(".yaml", "")
            print(f"[SAM2] 找到配置 (精确匹配): {yaml_path}")
            break
    if config_file is None:
        for yaml_path in all_yamls:
            if base_name in yaml_path.stem and "training" not in str(yaml_path):
                config_file = str(yaml_path.relative_to(configs_dir)).replace(".yaml", "")
                print(f"[SAM2] 找到配置 (部分匹配): {yaml_path}")
                break

    if config_file is None:
        raise FileNotFoundError(
            f"找不到配置文件 '{model_cfg}'.\n"
            f"configs 目录: {configs_dir}\n"
            f"可用文件: {[str(p.relative_to(configs_dir)) for p in configs_dir.rglob('*.yaml')]}"
        )

    # ---- Step 2: 正确初始化 Hydra 并加载配置 ----
    import hydra
    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    # 必须先清理旧的 Hydra 状态
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # 用 initialize_config_dir 指定配置目录的绝对路径
    with hydra.initialize_config_dir(config_dir=str(configs_dir.resolve()), version_base="1.2"):
        cfg = compose(config_name=config_file)

    print(f"[SAM2] 配置加载成功: {OmegaConf.to_container(cfg).get('model', {}).get('_target_', 'unknown')}")

    # ---- Step 3: 从配置构建模型 ----
    from hydra.utils import instantiate
    model = instantiate(cfg.model, _recursive_=True)

    # 加载权重
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        # 过滤掉不匹配的 key
        model_dict = model.state_dict()
        filtered_ckpt = {k: v for k, v in ckpt.items() if k in model_dict and v.shape == model_dict[k].shape}
        missing = set(model_dict.keys()) - set(filtered_ckpt.keys())
        if missing:
            print(f"[SAM2] 权重中缺少 {len(missing)} 个 key (通常是正常的)")
        model.load_state_dict(filtered_ckpt, strict=False)
        print(f"[SAM2] 加载权重: {len(filtered_ckpt)}/{len(model_dict)} 个参数匹配")

    model = model.to(device).eval()

    # ---- Step 4: 创建 Predictor ----
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[SAM2] ✅ 模型就绪! 总参数: {total_params/1e6:.1f}M, Device: {device}")
    return model, predictor


# ============================================================
#  损失函数
# ============================================================
class SegmentationLoss(nn.Module):
    """Dice Loss + BCE Loss 组合"""

    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred_masks, gt_masks):
        """
        pred_masks: (B, 1, H, W) sigmoid 之前的 logits
        gt_masks:   (B, H, W) 二值 mask (0/1)
        """
        pred = pred_masks.squeeze(1)  # (B, H, W)
        gt = gt_masks.float()

        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, gt)

        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * gt).sum(dim=(1, 2))
        union = pred_sigmoid.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ============================================================
#  指标计算
# ============================================================
def compute_iou(pred_mask, gt_mask, threshold=0.5):
    """计算 IoU"""
    pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
    intersection = (pred_binary * gt_mask).sum()
    union = pred_binary.sum() + gt_mask.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def compute_dice(pred_mask, gt_mask, threshold=0.5):
    """计算 Dice coefficient"""
    pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
    intersection = (pred_binary * gt_mask).sum()
    dice = (2.0 * intersection + 1e-6) / (pred_binary.sum() + gt_mask.sum() + 1e-6)
    return dice.item()


# ============================================================
#  SAM2 前向传播 (训练模式, 支持梯度回传)
# ============================================================
def sam2_forward_with_grad(model, predictor, image_tensor, point_coords_np, point_labels_np):
    """
    绕过 predictor.set_image() 的 @no_grad 限制,
    直接调用 SAM2 内部组件, 使梯度可以流过 LoRA 层.
    """
    device = image_tensor.device

    # ---- 1. 图像编码 (WITH 梯度) ----
    img_batch = image_tensor.unsqueeze(0) * 255.0  # (1, 3, H, W), [0, 255]

    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
    input_image = (img_batch - pixel_mean) / pixel_std

    backbone_out = model.forward_image(input_image)
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)

    # 安全检查 directly_add_no_mem_embed
    if getattr(model, "directly_add_no_mem_embed", False):
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

    # 整理 feature 格式
    bb_feat_sizes = predictor._bb_feat_sizes
    feats = [
        feat.permute(1, 2, 0).view(1, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
    ][::-1]

    image_embed = feats[-1]
    high_res_feats = feats[:-1]

    # ---- 2. Prompt 编码 ----
    # 设置 predictor 内部状态 (让 _prep_prompts 正常工作)
    predictor._orig_hw = [(image_tensor.shape[1], image_tensor.shape[2])]
    predictor._is_image_set = True
    predictor._is_batch = False
    # 存储 features (某些版本的 _prep_prompts 可能需要)
    predictor._features = {"image_embed": image_embed, "high_res_feats": high_res_feats}

    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
        point_coords_np, point_labels_np,
        box=None, mask_logits=None, normalize_coords=True,
    )
    # 确保 prompt tensor 在正确的设备上
    unnorm_coords = unnorm_coords.to(device)
    labels = labels.to(device)

    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=(unnorm_coords, labels),
        boxes=None,
        masks=None,
    )

    # ---- 3. Mask 解码 ----
    high_res_features = [
        feat_level[-1].unsqueeze(0) for feat_level in high_res_feats
    ]

    low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embed[-1].unsqueeze(0),
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )

    return low_res_masks, iou_predictions


# ============================================================
#  训练函数
# ============================================================
def train_one_epoch(model, predictor, dataloader, optimizer, scaler, loss_fn, device, epoch):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch["image"].to(device)                   # (B, 3, H, W)
        gt_masks = batch["mask"].to(device)                  # (B, H, W)
        point_coords = batch["point_coords"]                 # (B, 1, 2) numpy-compatible
        point_labels = batch["point_labels"]                 # (B, 1)

        gt_binary = (gt_masks > 0).float()

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            batch_loss = 0.0
            batch_iou = 0.0

            for i in range(images.shape[0]):
                # 用自定义前向传播 (带梯度)
                low_res_masks, iou_pred = sam2_forward_with_grad(
                    model, predictor,
                    images[i],
                    point_coords[i:i+1].numpy(),
                    point_labels[i:i+1].numpy(),
                )
                # low_res_masks: (1, 1, H', W') — SAM2 输出的低分辨率 mask

                # Resize 到原图尺寸
                logits = F.interpolate(
                    low_res_masks,
                    size=gt_binary[i].shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)  # (H, W)

                loss = loss_fn(logits.unsqueeze(0).unsqueeze(0), gt_binary[i].unsqueeze(0))
                batch_loss += loss
                batch_iou += compute_iou(logits, gt_binary[i])

            batch_loss = batch_loss / images.shape[0]

        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += batch_loss.item()
        total_iou += batch_iou / images.shape[0]
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{total_loss/num_batches:.4f}",
            "iou": f"{total_iou/num_batches:.4f}",
        })

    return total_loss / num_batches, total_iou / num_batches


@torch.no_grad()
def validate(model, predictor, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0

    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["image"].to(device)
        gt_masks = batch["mask"].to(device)
        point_coords = batch["point_coords"]
        point_labels = batch["point_labels"]
        gt_binary = (gt_masks > 0).float()

        for i in range(images.shape[0]):
            # 验证时用 numpy → set_image (无梯度, 更快)
            img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            predictor.set_image(img_np)

            pred_masks, scores, logits = predictor.predict(
                point_coords=point_coords[i:i+1].numpy(),
                point_labels=point_labels[i:i+1].numpy(),
                multimask_output=False,
                return_logits=True,
            )

            if isinstance(logits, np.ndarray):
                logits_tensor = torch.from_numpy(logits).to(device).float()
            else:
                logits_tensor = logits.float()

            if logits_tensor.shape[-2:] != gt_binary[i].shape:
                logits_tensor = F.interpolate(
                    logits_tensor.unsqueeze(0),
                    size=gt_binary[i].shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            loss = loss_fn(logits_tensor.unsqueeze(0), gt_binary[i].unsqueeze(0))
            total_loss += loss.item()
            total_iou += compute_iou(logits_tensor.squeeze(), gt_binary[i])
            total_dice += compute_dice(logits_tensor.squeeze(), gt_binary[i])
            num_samples += 1

    return (
        total_loss / max(num_samples, 1),
        total_iou / max(num_samples, 1),
        total_dice / max(num_samples, 1),
    )


# ============================================================
#  主函数
# ============================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SAM2 + LoRA 钢材缺陷分割 — 快速可行性验证")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    # ---- 输出目录 ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载数据 ----
    print("[DATA] 加载数据集...")
    data_root = {
        "neu_seg": "data/NEU-Seg",
        "severstal": "data/severstal",
    }
    root_dir = args.data_dir if args.data_dir else data_root.get(args.dataset, "data")

    train_dataset = get_dataset(args.dataset, root_dir, split="train", img_size=args.img_size)
    val_dataset = get_dataset(args.dataset, root_dir, split="val", img_size=args.img_size)

    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=1, shuffle=False)

    # ---- 加载 SAM2 模型 ----
    print("\n[MODEL] 加载 SAM2 模型...")
    model, predictor = build_sam2_model(
        checkpoint_path=args.checkpoint,
        model_cfg=args.model_cfg,
        device=device,
    )

    # ---- 冻结所有参数 ----
    for param in model.parameters():
        param.requires_grad = False

    # ---- 注入 LoRA ----
    print("\n[LoRA] 注入 LoRA 到 image_encoder...")
    lora_params = inject_lora_to_model(
        model.image_encoder,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )

    # ---- 解冻 mask_decoder ----
    # SAM2 中 mask decoder 的属性名是 sam_mask_decoder
    mask_decoder = model.sam_mask_decoder
    print("[MODEL] 解冻 sam_mask_decoder...")
    decoder_params = []
    for param in mask_decoder.parameters():
        param.requires_grad = True
        decoder_params.append(param)
    num_decoder_params = sum(p.numel() for p in decoder_params)
    print(f"[MODEL] sam_mask_decoder 可训练参数: {num_decoder_params:,}")

    # ---- 统计参数量 ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[PARAMS] 总参数: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"[PARAMS] 可训练: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"[PARAMS] 训练比例: {trainable_params/total_params*100:.2f}%")

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr},
            {"params": decoder_params, "lr": args.lr * 0.1},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    loss_fn = SegmentationLoss(dice_weight=0.5, bce_weight=0.5)
    scaler = torch.amp.GradScaler("cuda")

    # ---- 训练循环 ----
    print(f"\n{'='*60}")
    print(f"  开始训练: {args.epochs} epochs")
    print(f"  Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"  LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"{'='*60}\n")

    best_iou = 0.0
    results = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_iou = train_one_epoch(
            model, predictor, train_loader, optimizer, scaler, loss_fn, device, epoch
        )

        # Validate
        val_loss, val_iou, val_dice = validate(
            model, predictor, val_loader, loss_fn, device
        )

        scheduler.step()

        elapsed = time.time() - t0

        # Log
        log_msg = (
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f} | "
            f"Time: {elapsed:.1f}s"
        )
        print(log_msg)

        results.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
        })

        # Save best
        if val_iou > best_iou:
            best_iou = val_iou
            # 只保存可训练参数
            save_dict = {
                "epoch": epoch,
                "val_iou": val_iou,
                "val_dice": val_dice,
                "lora_state_dict": {
                    name: param.data.clone()
                    for name, param in model.image_encoder.named_parameters()
                    if param.requires_grad
                },
                "decoder_state_dict": model.sam_mask_decoder.state_dict(),
            }
            save_path = output_dir / "best_model.pth"
            torch.save(save_dict, save_path)
            print(f"  ✅ 保存最佳模型: IoU={val_iou:.4f} → {save_path}")

    # ---- 保存训练日志 ----
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "config": vars(args),
            "best_iou": best_iou,
            "results": results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  训练完成!")
    print(f"  最佳 Val IoU: {best_iou:.4f}")
    print(f"  模型保存: {output_dir / 'best_model.pth'}")
    print(f"  日志保存: {log_path}")
    print(f"{'='*60}")


# ============================================================
#  参数解析
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 + LoRA 钢材缺陷分割训练")

    # 数据
    parser.add_argument("--dataset", default="neu_seg", choices=["neu_seg", "severstal"])
    parser.add_argument("--data_dir", default=None, help="数据集根目录")
    parser.add_argument("--img_size", type=int, default=1024)

    # 模型
    parser.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--model_cfg", default="sam2.1_hiera_b+.yaml")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=4, help="LoRA alpha")

    # 训练
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # 输出
    parser.add_argument("--output_dir", default="outputs")

    args = parser.parse_args()
    main(args)
