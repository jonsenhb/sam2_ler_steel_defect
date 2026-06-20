"""
defect_sam2_adapt.py — DefectSAM2 hierarchical adaptation for SAM2 steel defect segmentation.

Inspired by DefectSAM (TNNLS 2025): CGFA + MGFA between encoder scales before decode.
  CGFA: cross-correlation spatial gating between ViT scale and reference scale.
  MGFA: coarse semantic mask guides fine-scale feature modulation.

Used with frozen SAM2 encoder + LoRA (same as SAM2+LoRA+FPN baseline).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CGFA(nn.Module):
    """Cross-Correlation Gated Feature Adaptation."""

    def __init__(self, c_feat: int, c_ref: int, c_out: int | None = None):
        super().__init__()
        c_out = c_out or c_feat
        self.proj_feat = nn.Sequential(
            nn.Conv2d(c_feat, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )
        self.proj_ref = nn.Sequential(
            nn.Conv2d(c_ref, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(c_out, c_out, 3, padding=1, groups=c_out, bias=False),
            nn.BatchNorm2d(c_out),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )

    def forward(self, feat: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if ref.shape[-2:] != feat.shape[-2:]:
            ref = F.interpolate(ref, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        a = self.proj_feat(feat)
        b = self.proj_ref(ref)
        g = self.gate(a * b)
        return self.fuse(torch.cat([a, g * b], dim=1))


class MGFA(nn.Module):
    """Mask-Guided Feature Adaptation using coarse semantic logits."""

    def __init__(self, channels: int):
        super().__init__()
        self.fg = nn.Conv2d(channels, channels, 1, bias=False)
        self.bg = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, feat: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(coarse_logits, dim=1)
        fg_w = 1.0 - probs[:, :1]
        bg_w = probs[:, :1]
        fg_w = F.interpolate(fg_w, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        bg_w = F.interpolate(bg_w, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        mod = self.fg(feat) * fg_w + self.bg(feat) * bg_w
        return feat + self.norm(mod)


class DefectSAM2Head(nn.Module):
    """
    Hierarchical CGFA + MGFA + FPN decode (drop-in replacement for MultiClassFPNHead).

    forward(image_embed, high_res_feats) -> logits (B, num_classes, 256, 256)
    """

    def __init__(self, embed_dim: int = 256, hr_dims=None, num_classes: int = 4):
        super().__init__()
        if hr_dims is None:
            hr_dims = [32, 64]
        self.num_classes = num_classes

        self.coarse_cls = nn.Conv2d(embed_dim, num_classes, 1)

        self.cgfa_hr1 = CGFA(hr_dims[1], embed_dim, hr_dims[1])
        self.cgfa_hr0 = CGFA(hr_dims[0], hr_dims[1], hr_dims[0])
        self.mgfa = MGFA(hr_dims[0])

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(128 + hr_dims[1], 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(64 + hr_dims[0], 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, num_classes, 1),
        )

    def _decode(self, image_embed: torch.Tensor, hr0_a: torch.Tensor, hr1_a: torch.Tensor) -> torch.Tensor:
        x = self.up3(image_embed)
        x = self.fuse2(torch.cat([x, hr1_a], dim=1))
        x = self.up2(x)
        x = self.fuse1(torch.cat([x, hr0_a], dim=1))
        return self.classifier(x)

    def forward(self, image_embed: torch.Tensor, high_res_feats: list[torch.Tensor]) -> torch.Tensor:
        hr0, hr1 = high_res_feats[0], high_res_feats[1]
        coarse_logits = self.coarse_cls(image_embed)
        hr1_a = self.cgfa_hr1(hr1, image_embed)
        hr0_a = self.cgfa_hr0(hr0, hr1_a)
        hr0_a = self.mgfa(hr0_a, coarse_logits)
        return self._decode(image_embed, hr0_a, hr1_a)

    def forward_with_aux(self, image_embed: torch.Tensor, high_res_feats: list[torch.Tensor]):
        """Training: return (fine_logits, coarse_logits) for deep supervision."""
        hr0, hr1 = high_res_feats[0], high_res_feats[1]
        coarse_logits = self.coarse_cls(image_embed)
        hr1_a = self.cgfa_hr1(hr1, image_embed)
        hr0_a = self.cgfa_hr0(hr0, hr1_a)
        hr0_a = self.mgfa(hr0_a, coarse_logits)
        return self._decode(image_embed, hr0_a, hr1_a), coarse_logits


def count_head_params(head: nn.Module) -> int:
    return sum(p.numel() for p in head.parameters())
