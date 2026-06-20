# Label-efficiency evaluation protocol

All methods in `results/paper_de/` share **identical** data splits.

## Parameters

| Parameter | Value |
|-----------|-------|
| Training pool | 1200 images (fixed RNG seed `20240618`) |
| Validation set | 400 images (same seed) |
| Label fractions | 1%, 5%, 10%, 25%, 100% of pool |
| Subset sampling | Per-fraction seed `{0,1,2}` → 3 runs |
| Input size (SAM2) | 1024 × 1024 |
| Metric | Global pooled mIoU @ 256 × 256 |
| Loss | CE + Dice (multiclass) |
| SAM2-LER epochs | 30, patience 7, AdamW lr=1e-4 (head 2×) |
| LoRA | rank=4, alpha=4 on attention qkv/proj |

## Label counts @ 1%

~12 training images from 1200 pool (NEU & Severstal).

## File naming

```
{method}_f{frac*1000:04d}_s{seed}.json
```

Example: `sam2_lora_f0010_s0.json` = SAM2-LER, 1%, seed 0.

## Methods key

| JSON `method` | Paper name |
|---------------|------------|
| `sam2_lora` | SAM2-LER (Ours) |
| `sam2_linear` | SAM2 frozen + linear probe |
| `unet`, `deeplabv3plus`, ... | CNN baselines |
