"""
render_qualitative.py — Fig.4 qualitative panel for SAM2-LER manuscript.

Shows validation samples: Input | GT | SAM2-LER | U-Net (quick-trained @1% NEU).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

NEU_COLORS = np.array([
    [0.2, 0.2, 0.2],
    [0.9, 0.3, 0.2],
    [0.3, 0.7, 0.9],
    [0.1, 0.7, 0.5],
])


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return NEU_COLORS[np.clip(mask.astype(int), 0, 3)]


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
    })


@torch.no_grad()
def predict_sam2_ler(model, predictor, head, ds_va, val_idx, device):
    from train_multiclass import encoder_forward

    preds = {}
    model.eval()
    head.eval()
    for i in val_idx:
        s = ds_va[i]
        embed, hr = encoder_forward(model, predictor, s["image"].to(device))
        logit = head(embed, hr)
        preds[i] = logit.argmax(1)[0].cpu().numpy()
    return preds


def train_and_predict_sam2_ler(ds_tr, ds_va, tr_idx, val_idx, device, epochs, seed):
    from train import build_sam2_model, inject_lora_to_model
    from train_multiclass import encoder_forward, MultiClassFPNHead, MultiClassLoss

    torch.manual_seed(seed)
    np.random.seed(seed)
    model, predictor = build_sam2_model(
        "checkpoints/sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml", device=device,
    )
    for p in model.parameters():
        p.requires_grad = False
    lora_params = inject_lora_to_model(model.image_encoder, rank=4, alpha=4)
    head = MultiClassFPNHead(embed_dim=256, hr_dims=[32, 64], num_classes=4).to(device)
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": 1e-4}, {"params": head.parameters(), "lr": 2e-4}],
        weight_decay=0.01,
    )
    lossf = MultiClassLoss(num_classes=4)
    scaler = GradScaler("cuda")
    g = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        head.train()
        for j in g.permutation(len(tr_idx)):
            s = ds_tr[tr_idx[j]]
            img = s["image"].to(device)
            gt = s["mask"].to(device)
            opt.zero_grad()
            with autocast("cuda"):
                embed, hr = encoder_forward(model, predictor, img)
                logit = head(embed, hr)
                gt_i = F.interpolate(
                    gt.unsqueeze(0).unsqueeze(0).float(),
                    size=logit.shape[-2:], mode="nearest",
                ).squeeze(1).long()
                loss = lossf(logit, gt_i)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    preds = predict_sam2_ler(model, predictor, head, ds_va, val_idx, device)
    del model, head
    torch.cuda.empty_cache()
    return preds


def train_and_predict_unet(ds_tr, ds_va, tr_idx, val_idx, device, epochs, seed):
    import segmentation_models_pytorch as smp
    from train_multiclass import MultiClassLoss

    torch.manual_seed(seed)
    net = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=4).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    lossf = MultiClassLoss(num_classes=4)
    dl = DataLoader(Subset(ds_tr, tr_idx), batch_size=4, shuffle=True)
    for _ in range(epochs):
        net.train()
        for batch in dl:
            img = batch["image"].to(device)
            gt = batch["mask"].to(device)
            if img.shape[-1] != 256:
                img = F.interpolate(img, size=(256, 256), mode="bilinear", align_corners=False)
                gt = F.interpolate(gt.unsqueeze(1).float(), size=(256, 256), mode="nearest").squeeze(1).long()
            opt.zero_grad()
            lossf(net(img), gt).backward()
            opt.step()
    preds = {}
    net.eval()
    for i in val_idx:
        s = ds_va[i]
        img = s["image"].to(device).unsqueeze(0)
        if img.shape[-1] != 256:
            img = F.interpolate(img, size=(256, 256), mode="bilinear", align_corners=False)
        preds[i] = net(img).argmax(1)[0].cpu().numpy()
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="release/manuscript_sensors/figures")
    p.add_argument("--data_dir", default="data/NEU-Seg")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--gt_only", action="store_true")
    args = p.parse_args()

    setup_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from dataset import get_dataset
    from paper_de_pipeline import make_pool_val, frac_subset

    ds_tr = get_dataset("neu_seg", args.data_dir, split="train", img_size=1024)
    ds_va = get_dataset("neu_seg", args.data_dir, split="val", img_size=1024)
    pool, val_pool = make_pool_val(len(ds_tr), 1200, len(ds_va), 400)
    tr_idx, _ = frac_subset(pool, 0.01, 0)
    val_idx = val_pool[: args.n_samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam_preds, unet_preds = {}, {}
    if not args.gt_only and device.type == "cuda":
        print(f"Quick-training SAM2-LER ({args.epochs} ep) and U-Net for qualitative panel...")
        sam_preds = train_and_predict_sam2_ler(ds_tr, ds_va, tr_idx, val_idx, device, args.epochs, 0)
        ds_tr256 = get_dataset("neu_seg", args.data_dir, split="train", img_size=256)
        ds_va256 = get_dataset("neu_seg", args.data_dir, split="val", img_size=256)
        unet_preds = train_and_predict_unet(ds_tr256, ds_va256, tr_idx, val_idx, device, args.epochs, 0)

    n_cols = 4 if sam_preds else 2
    fig, axes = plt.subplots(args.n_samples, n_cols, figsize=(2.2 * n_cols, 2.2 * args.n_samples))
    if args.n_samples == 1:
        axes = axes.reshape(1, -1)
    col_titles = ["Input", "Ground truth", "SAM2-LER", "U-Net"][:n_cols]

    for row, vi in enumerate(val_idx):
        s = ds_va[vi]
        img = s["image"].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        gt = s["mask"].numpy()
        if gt.ndim == 3 or (hasattr(gt, "shape") and len(gt.shape) == 2 and gt.shape[0] == 256):
            pass
        gt_t = torch.from_numpy(gt.astype(np.float32))
        if gt_t.ndim == 2:
            gt_t = gt_t.unsqueeze(0)
        gt = F.interpolate(gt_t.unsqueeze(0), size=(256, 256), mode="nearest").squeeze().numpy().astype(int)

        panels = [img, mask_to_rgb(gt)]
        if sam_preds:
            panels.append(mask_to_rgb(sam_preds[vi]))
            panels.append(mask_to_rgb(unet_preds[vi]))

        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=9, fontweight="bold")

    fig.suptitle("Qualitative segmentation (@1% labels, NEU-Seg)", fontsize=10, y=1.01)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig04_qualitative_panel.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out_dir / 'fig04_qualitative_panel.pdf'}")


if __name__ == "__main__":
    main()
