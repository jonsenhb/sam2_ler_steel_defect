# 对比方法与论文出处

## Ours + Basic Baselines

| 方法 | 出处 |
|---|---|
| SAM2(frozen)+Linear (Ours) | SAM2: Ravi et al., arXiv:2408.00714, 2024 |
| SAM2(frozen)+LoRA+FPN (Ours) | SAM2: Ravi et al. 2024; LoRA: Hu et al., ICLR 2022 |
| U-Net | Ronneberger et al., MICCAI 2015 (backbone: He et al., CVPR 2016) |
| U-Net (ImageNet) | Ronneberger et al., MICCAI 2015; ImageNet pretrain: Russakovsky et al., IJCV 2015 |
| DeepLabV3+ | Chen et al., ECCV 2018 (backbone: He et al., CVPR 2016) |
| PSPNet | Zhao et al., CVPR 2017 (backbone: He et al., CVPR 2016) |
| SegFormer | Xie et al., NeurIPS 2021 |

## External SOTA (reproduced under our protocol)

| 方法 | 出处 |
|---|---|
| DDSNet* (UNet-R50+BndLoss) | Yin et al., IEEE TIM 2024 (approx. reproduction under our protocol) |
| MFF-Metal* (UNet++-R34+MultiLoss) | Li et al., J.Supercomputing 2025 (approx. reproduction under our protocol) |
| SME-DLV3+* (DLV3+-R50) | Zhang et al., PLOS One 2025 (approx. reproduction under our protocol) |
| Hybrid-Trans* (UPerNet-MiT-B4) | Sime et al., MTA 2024 (approx. reproduction under our protocol) |

## 数据集
- NEU-Seg: Song & Yan, Applied Surface Science 2013 (NEU surface defect database)
- Severstal: Severstal Steel Defect Detection, Kaggle 2019
