"""
segmentation_metrics.py — 语义分割公平指标

标准做法 (Cityscapes / Pascal VOC / MMSegmentation):
  对验证集 **全局累加** 每类 intersection / union, 再算 IoU.
  避免 per-image 平均在「GT 无该类」时因 epsilon 平滑虚高为 1.0.

同时提供 per-image-present 均值, 仅对 GT 含该类的图像平均 (辅助分析).
"""

from __future__ import annotations

import numpy as np
import torch


class SegmentationMetricAccumulator:
    """在验证集上累加全局 inter/union (per-class)."""

    def __init__(self, num_classes: int, class_id_to_name: dict | None = None):
        self.num_classes = num_classes
        self.class_id_to_name = class_id_to_name or {
            i: f"c{i}" for i in range(num_classes)
        }
        self.inter = np.zeros(num_classes, dtype=np.int64)
        self.union = np.zeros(num_classes, dtype=np.int64)
        # 仅对 GT 含该类的图像记录 IoU (辅助)
        self._present_ious: dict[int, list[float]] = {c: [] for c in range(num_classes)}

    def reset(self):
        self.inter.fill(0)
        self.union.fill(0)
        self._present_ious = {c: [] for c in range(self.num_classes)}

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        """pred, gt: (H, W) long tensors."""
        pred = pred.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        for c in range(self.num_classes):
            pred_c = pred == c
            gt_c = gt == c
            inter = int(np.logical_and(pred_c, gt_c).sum())
            union = int(np.logical_or(pred_c, gt_c).sum())
            self.inter[c] += inter
            self.union[c] += union
            if gt_c.any():
                iou = inter / union if union > 0 else 1.0
                self._present_ious[c].append(float(iou))

    def update_logits(self, logits: torch.Tensor, targets: torch.Tensor):
        pred = logits.argmax(dim=1)
        if pred.dim() == 3 and pred.shape[0] == 1:
            self.update(pred[0], targets[0] if targets.dim() == 3 else targets)
        elif pred.dim() == 2:
            self.update(pred, targets)
        else:
            for b in range(pred.shape[0]):
                self.update(pred[b], targets[b])

    def compute(self) -> dict:
        per_class_global: dict[str, float] = {}
        for c in range(self.num_classes):
            name = self.class_id_to_name.get(c, f"c{c}")
            if self.union[c] > 0:
                per_class_global[name] = float(self.inter[c] / self.union[c])
            else:
                per_class_global[name] = float("nan")

        defect_global = [
            v for k, v in per_class_global.items()
            if k != "background" and not np.isnan(v)
        ]
        miou_global = float(np.mean(defect_global)) if defect_global else 0.0

        per_class_present: dict[str, float] = {}
        for c in range(self.num_classes):
            name = self.class_id_to_name.get(c, f"c{c}")
            vals = self._present_ious[c]
            per_class_present[name] = float(np.mean(vals)) if vals else float("nan")

        defect_present = [
            v for k, v in per_class_present.items()
            if k != "background" and not np.isnan(v)
        ]
        miou_present = float(np.mean(defect_present)) if defect_present else 0.0

        return {
            "miou_global": miou_global,
            "miou_present": miou_present,
            "per_class_global": per_class_global,
            "per_class_present": per_class_present,
        }


def compute_multiclass_metrics(
    logits,
    targets,
    num_classes=4,
    class_id_to_name=None,
    mode: str = "global_batch",
):
    """
    兼容旧接口; mode='global_batch' 对当前 batch 做 pooled IoU.
    mode='per_image' 保留旧 per-image 行为 (不推荐用于 model selection).
    """
    if class_id_to_name is None:
        class_id_to_name = {0: "background", 1: "patches", 2: "inclusion", 3: "scratches"}

    pred = logits.argmax(dim=1)
    if mode == "per_image":
        return _per_image_metrics(pred, targets, num_classes, class_id_to_name)

    acc = SegmentationMetricAccumulator(num_classes, class_id_to_name)
    if pred.dim() == 3:
        for b in range(pred.shape[0]):
            acc.update(pred[b], targets[b] if targets.dim() == 3 else targets)
    else:
        acc.update(pred, targets)
    out = acc.compute()
    return out["miou_global"], out["per_class_global"]

def _per_image_metrics(pred, targets, num_classes, class_id_to_name):
    class_iou = {}
    if pred.dim() == 3:
        pred = pred[0]
        targets = targets[0] if targets.dim() == 3 else targets
    for c in range(num_classes):
        pred_c = pred == c
        gt_c = targets == c
        inter = (pred_c & gt_c).sum().float()
        union = (pred_c | gt_c).sum().float()
        if union.item() == 0 and not gt_c.any():
            iou = float("nan")
        elif union.item() == 0:
            iou = 1.0
        else:
            iou = (inter / union).item()
        class_iou[class_id_to_name.get(c, f"c{c}")] = iou
    defect = [v for k, v in class_iou.items() if k != "background" and not np.isnan(v)]
    miou = float(np.mean(defect)) if defect else 0.0
    return miou, class_iou
