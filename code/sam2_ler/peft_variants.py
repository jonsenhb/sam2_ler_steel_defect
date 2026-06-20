"""
peft_variants.py — Conv-LoRA 改进方向

  A. MAP-LoRA (Morphology-Aware PEFT)
     按层深度初始化 morphology 路由: 浅层偏 local conv, 深层偏 global (patches).

  B. Layer-Phase Adaptive PEFT
     前半层 Std-LoRA (稳定语义), 后半层 Conv-LoRA (纹理/形态).

  C. Rank-Adaptive PEFT
     浅层 rank=2, 中层 rank=4, 深层 rank=8 (容量随语义深度递增).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from conv_lora import DefectConvLoRA, print_mix_weights
from train import LoRALinear, inject_lora_to_model


def _collect_target_linears(model, target_modules=None):
    if target_modules is None:
        target_modules = ["qkv", "proj", "q_proj", "k_proj", "v_proj", "out_proj"]
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            layers.append((name, module))
    return layers


def _replace_module(model, name, new_module):
    parts = name.rsplit(".", 1)
    parent = dict(model.named_modules())[parts[0]] if len(parts) == 2 else model
    attr = parts[1] if len(parts) == 2 else parts[0]
    setattr(parent, attr, new_module)


def _rank_for_depth(layer_idx: int, num_layers: int, base_rank: int = 4) -> int:
    """浅→深: 2 → 4 → 8 (至少 base_rank)."""
    if num_layers <= 1:
        return base_rank
    t = layer_idx / (num_layers - 1)
    if t < 0.34:
        return max(2, base_rank // 2)
    if t < 0.67:
        return base_rank
    return min(8, base_rank * 2)


def _morphology_mix_init(layer_idx: int, num_layers: int) -> list[float]:
    """
    Morphology-aware 初始混合 [identity, conv3, dil_conv, global].
    浅层: 细纹理 (scratches) → conv3
    中层: 均衡
    深层: 大面积 (patches) → global
    """
    if num_layers <= 1:
        return [1.0, 1.0, 1.0, 1.0]
    t = layer_idx / (num_layers - 1)
    if t < 0.33:
        return [0.6, 2.0, 1.2, 0.3]
    if t < 0.66:
        return [1.0, 1.2, 1.2, 0.8]
    return [0.7, 0.6, 1.0, 2.0]


class MorphologyAwareConvLoRA(DefectConvLoRA):
    """MAP-LoRA: morphology 先验初始化 + 可学习微调."""

    def __init__(self, original_linear, rank=4, alpha=4, layer_idx=0, num_layers=1):
        super().__init__(original_linear, rank=rank, alpha=alpha)
        init = _morphology_mix_init(layer_idx, num_layers)
        with torch.no_grad():
            self.mix_logits.copy_(torch.tensor(init, device=self.mix_logits.device))


def inject_map_lora(model, rank=4, alpha=4, target_modules=None):
    layers = _collect_target_linears(model, target_modules)
    n = len(layers)
    all_params = []
    for i, (name, module) in enumerate(layers):
        layer = MorphologyAwareConvLoRA(module, rank=rank, alpha=alpha, layer_idx=i, num_layers=n)
        _replace_module(model, name, layer)
        all_params.extend([layer.lora_A, layer.lora_B])
        all_params.extend(layer.conv3.parameters())
        all_params.extend(layer.dil_conv3.parameters())
        all_params.append(layer.mix_logits)

    total = sum(p.numel() for p in all_params)
    print(f"[MAP-LoRA] 注入 {n} 层 (morphology-aware mix init), rank={rank}")
    print(f"[MAP-LoRA] 总可训练: {total:,} ({total/1e6:.2f}M)")
    return all_params


def inject_phase_adaptive_lora(model, rank=4, alpha=4, target_modules=None, conv_fraction=0.5):
    """
    Layer-Phase Adaptive: 前 (1-f) 层 Std-LoRA, 后 f 层 Conv-LoRA.
    """
    layers = _collect_target_linears(model, target_modules)
    n = len(layers)
    split = int(n * (1.0 - conv_fraction))
    all_params = []
    for i, (name, module) in enumerate(layers):
        if i < split:
            layer = LoRALinear(module, rank=rank, alpha=alpha)
        else:
            layer = DefectConvLoRA(module, rank=rank, alpha=alpha)
        _replace_module(model, name, layer)
        all_params.extend([layer.lora_A, layer.lora_B])
        if isinstance(layer, DefectConvLoRA):
            all_params.extend(layer.conv3.parameters())
            all_params.extend(layer.dil_conv3.parameters())
            all_params.append(layer.mix_logits)

    total = sum(p.numel() for p in all_params)
    print(f"[Phase-Adaptive] 注入 {n} 层: Std×{split} + Conv×{n - split}, rank={rank}")
    print(f"[Phase-Adaptive] 总可训练: {total:,} ({total/1e6:.2f}M)")
    return all_params


def inject_rank_adaptive_lora(model, base_rank=4, alpha=4, target_modules=None, use_conv=True):
    """Rank-Adaptive: 每层 rank 随深度 2→4→8."""
    layers = _collect_target_linears(model, target_modules)
    n = len(layers)
    all_params = []
    ranks_used = []
    for i, (name, module) in enumerate(layers):
        r = _rank_for_depth(i, n, base_rank=base_rank)
        ranks_used.append(r)
        if use_conv:
            layer = DefectConvLoRA(module, rank=r, alpha=alpha)
            _replace_module(model, name, layer)
            all_params.extend([layer.lora_A, layer.lora_B])
            all_params.extend(layer.conv3.parameters())
            all_params.extend(layer.dil_conv3.parameters())
            all_params.append(layer.mix_logits)
        else:
            layer = LoRALinear(module, rank=r, alpha=alpha)
            _replace_module(model, name, layer)
            all_params.extend([layer.lora_A, layer.lora_B])

    total = sum(p.numel() for p in all_params)
    uniq = sorted(set(ranks_used))
    print(f"[Rank-Adaptive] 注入 {n} 层, ranks={uniq}, conv={use_conv}")
    print(f"[Rank-Adaptive] 总可训练: {total:,} ({total/1e6:.2f}M)")
    return all_params


def inject_peft(model, lora_type: str, rank=4, alpha=4):
    """统一入口."""
    if lora_type == "none":
        # 零 PEFT 控制组: 编码器完全冻结, 仅训练分割头. 用于定位性能天花板.
        print("[Zero-PEFT] 编码器完全冻结, 无可训练 LoRA 参数 (控制组)")
        return []
    if lora_type == "standard":
        return inject_lora_to_model(model, rank=rank, alpha=alpha)
    if lora_type == "conv":
        from conv_lora import inject_conv_lora
        return inject_conv_lora(model, rank=rank, alpha=alpha)
    if lora_type == "map":
        return inject_map_lora(model, rank=rank, alpha=alpha)
    if lora_type == "phase":
        return inject_phase_adaptive_lora(model, rank=rank, alpha=alpha)
    if lora_type == "rank_adapt":
        return inject_rank_adaptive_lora(model, base_rank=rank, alpha=alpha, use_conv=True)
    raise ValueError(f"未知 lora_type: {lora_type}")
