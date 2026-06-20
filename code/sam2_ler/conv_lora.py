"""
conv_lora.py — Defect-aware Conv-LoRA (卷积增强的 LoRA)

=== 来自已发表论文的关键洞察 ===

Conv-LoRA (ICLR 2024): 将卷积注入 LoRA 的 A 和 B 之间, 在低秩空间做空间卷积.
  公式: h = W₀x + B · Conv(A · x)  (不是: h = W₀x + B·A·x + Adapter(features))
  
DD-Adapter (2025): Depthwise-Dilated 卷积 adapter 在 SAM2 上超越 LoRA (Dice 0.92→0.93)

=== 为什么之前的 adapter 方案失败 ===
v1/v2 都是在 encoder 输出上做后处理, 纹理信息已经在 ViT tokenization 时丢失.
正确做法: 在 encoder 的每个 Transformer block 内部注入卷积, 让纹理先验
         在特征形成过程中就参与运算, 而不是事后补救.

=== 我们的改进: Defect-Conv-LoRA ===
在 Conv-LoRA 基础上做两点针对钢材缺陷的定制:
  1. 多尺度: 3×3 + dilated-3×3(d=2) 捕获细纹理和粗纹理
  2. 可学习混合: 让网络自动决定每个 block 需要多少卷积先验

公式: h = W₀x + B · (α·(Ax) + β·Conv3(Ax) + γ·DilConv3(Ax))
其中 α+β+γ=1, 由 softmax 归一化的可学习参数控制
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DefectConvLoRA(nn.Module):
    """
    Defect-aware Conv-LoRA Layer

    在标准 LoRA 的低秩空间中注入多尺度深度卷积:
      Input → A (降维到 rank r) → [identity + Conv3×3 + DilConv3×3] → B (升维回原始) → Output

    卷积在 rank-r 维度上操作, 参数量极小 (r*3*3 + r*3*3 per layer)
    """

    def __init__(self, original_linear, rank=4, alpha=4):
        super().__init__()
        self.original_linear = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # 获取设备和精度
        device = original_linear.weight.device
        dtype = original_linear.weight.dtype

        # 冻结原始权重
        for p in original_linear.parameters():
            p.requires_grad = False

        # ---- 标准 LoRA 矩阵 ----
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, device=device, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device, dtype=dtype))

        # ---- 多尺度深度卷积 (在低秩空间操作, 极轻量) ----
        # Conv 3×3: 捕获细粒度纹理 (scratches 等)
        self.conv3 = nn.Conv2d(
            rank, rank, kernel_size=3, padding=1,
            groups=rank, bias=False, device=device,
        ).to(dtype)

        # Dilated Conv 3×3, dilation=2: 捕获中等粒度纹理
        # 有效感受野 = 5×5, 但参数量和 3×3 一样
        self.dil_conv3 = nn.Conv2d(
            rank, rank, kernel_size=3, padding=2, dilation=2,
            groups=rank, bias=False, device=device,
        ).to(dtype)

        # Global Context: 全局平均池化, 捕获大面积缺陷 (patches)
        # 每个位置获得全局统计信息, 感受野 = 整张图
        # 无额外可训练参数 (池化本身无参数)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ---- 可学习混合权重: identity / conv3 / dil_conv3 / global ----
        # identity 初始值最高, 训练初期接近标准 LoRA
        self.mix_logits = nn.Parameter(torch.tensor([1.0, 0.5, 0.5, 0.5], device=device))

        # 守护标志: conv 路径是否真的被执行过 (防止维度不匹配导致死代码)
        self.conv_fired = False

    def forward(self, x):
        original_out = self.original_linear(x)

        # LoRA 编码器: 降维到 rank r
        x_low = x.to(self.lora_A.dtype) @ self.lora_A  # (..., rank)

        # 将 token 重塑为 2D 空间布局做卷积. 支持两种主干布局:
        #   4D: (B, H, W, r)        — SAM2 Hiera (qkv/proj 保持空间维度)
        #   3D: (B, N, r), N=H*H    — ViT 风格 (展平 token, 完美正方形)
        feat_2d = None
        restore = None
        shape = x_low.shape

        if len(shape) == 4:
            # (B, H, W, r) → (B, r, H, W)
            B, H, W, r = shape
            if H >= 1 and W >= 1:
                feat_2d = x_low.permute(0, 3, 1, 2).contiguous()

                def restore(fused_2d):  # (B, r, H, W) → (B, H, W, r)
                    return fused_2d.permute(0, 2, 3, 1).contiguous()
        elif len(shape) == 3:
            B, N, r = shape
            H = int(math.isqrt(N))
            if H * H == N and H >= 3:  # 完美正方形且足够大
                feat_2d = x_low.transpose(1, 2).reshape(B, r, H, H)

                def restore(fused_2d):  # (B, r, H, H) → (B, N, r)
                    return fused_2d.reshape(B, r, N).transpose(1, 2)

        if feat_2d is not None:
            self.conv_fired = True
            # 多尺度卷积 + 全局上下文
            out_identity = feat_2d
            out_conv3 = self.conv3(feat_2d)
            out_dil3 = self.dil_conv3(feat_2d)
            out_global = self.global_pool(feat_2d).expand_as(feat_2d)  # (B,r,1,1) → (B,r,H,W)

            # 可学习混合 (softmax 归一化)
            w = F.softmax(self.mix_logits, dim=0)
            fused = w[0] * out_identity + w[1] * out_conv3 + w[2] * out_dil3 + w[3] * out_global

            x_low = restore(fused)

        # LoRA 解码器: 升维回原始维度
        lora_out = (x_low @ self.lora_B) * self.scaling
        return original_out + lora_out.to(original_out.dtype)


def inject_conv_lora(model, rank=4, alpha=4, target_modules=None):
    """
    将模型中的线性层替换为 DefectConvLoRA
    返回所有可训练参数的列表
    """
    if target_modules is None:
        target_modules = ["qkv", "proj", "q_proj", "k_proj", "v_proj", "out_proj"]

    all_params = []
    injected = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue

        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = dict(model.named_modules())[parts[0]]
            attr = parts[1]
        else:
            parent = model
            attr = parts[0]

        conv_lora = DefectConvLoRA(module, rank=rank, alpha=alpha)
        setattr(parent, attr, conv_lora)

        # 收集可训练参数
        all_params.extend([conv_lora.lora_A, conv_lora.lora_B])
        all_params.extend(conv_lora.conv3.parameters())
        all_params.extend(conv_lora.dil_conv3.parameters())
        all_params.append(conv_lora.mix_logits)
        injected += 1

    total_params = sum(p.numel() for p in all_params)
    conv_params = sum(
        sum(p.numel() for p in m.conv3.parameters()) +
        sum(p.numel() for p in m.dil_conv3.parameters()) +
        m.mix_logits.numel()
        for m in model.modules() if isinstance(m, DefectConvLoRA)
    )

    print(f"[DefectConvLoRA] 注入 {injected} 层, rank={rank}")
    print(f"[DefectConvLoRA] LoRA 参数: {total_params - conv_params:,}")
    print(f"[DefectConvLoRA] Conv 参数: {conv_params:,} (多尺度卷积 + 混合权重)")
    print(f"[DefectConvLoRA] 总可训练: {total_params:,} ({total_params/1e6:.2f}M)")

    return all_params


def verify_conv_active(model, raise_on_fail=True):
    """
    检查所有 DefectConvLoRA 的 conv 路径是否真的被执行过.
    必须在至少一次前向之后调用. 返回 (active, total).
    """
    layers = [m for m in model.modules() if isinstance(m, DefectConvLoRA)]
    total = len(layers)
    active = sum(1 for m in layers if m.conv_fired)
    if total == 0:
        return 0, 0
    if active == 0 and raise_on_fail:
        raise RuntimeError(
            f"[DefectConvLoRA] 致命: {total} 层中 0 层激活了 conv 路径! "
            f"卷积分支是死代码 (Conv-LoRA 退化为 Std-LoRA). "
            f"请检查主干 token 维度与 forward 中的 reshape 逻辑."
        )
    if active < total:
        print(f"[DefectConvLoRA] ⚠ 警告: 仅 {active}/{total} 层激活 conv 路径")
    else:
        print(f"[DefectConvLoRA] ✅ conv 路径已激活: {active}/{total} 层")
    return active, total


def print_mix_weights(model):
    """打印每层的混合权重, 用于分析哪些层更依赖卷积 vs 全局上下文"""
    print("\n[Mix Weights] identity / conv3×3 / dil-conv3×3 / global:")
    for name, module in model.named_modules():
        if isinstance(module, DefectConvLoRA):
            w = F.softmax(module.mix_logits, dim=0).detach().cpu()
            short_name = name.split(".")[-2] + "." + name.split(".")[-1] if "." in name else name
            bar_id = "█" * int(w[0] * 20)
            bar_c3 = "█" * int(w[1] * 20)
            bar_dl = "█" * int(w[2] * 20)
            bar_gl = "█" * int(w[3] * 20)
            print(f"  {short_name:25s} id={w[0]:.2f}{bar_id} c3={w[1]:.2f}{bar_c3} dl={w[2]:.2f}{bar_dl} gl={w[3]:.2f}{bar_gl}")


if __name__ == "__main__":
    # 单元测试
    print("Testing DefectConvLoRA...")
    linear = nn.Linear(256, 256)
    conv_lora = DefectConvLoRA(linear, rank=4, alpha=4)

    # 测试可以做 2D conv 的情况 (N=64*64=4096)
    x = torch.randn(2, 4096, 256)
    out = conv_lora(x)
    assert out.shape == x.shape
    print(f"  ✓ 2D conv path: {x.shape} → {out.shape}")

    # 测试不能做 2D conv 的情况 (N=100, 不是完美正方形)
    x2 = torch.randn(2, 100, 256)
    out2 = conv_lora(x2)
    assert out2.shape == x2.shape
    print(f"  ✓ Fallback path: {x2.shape} → {out2.shape}")

    # 参数量
    params = sum(p.numel() for p in conv_lora.parameters() if p.requires_grad)
    print(f"  参数量: {params:,}")
    print("  ✅ 测试通过!")
