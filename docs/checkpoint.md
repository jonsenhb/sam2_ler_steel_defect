# SAM2.1 checkpoint

**Not included in Git** (~309 MB).

## Download

```bash
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_base_plus.pt \
  "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_base_plus.pt"
```

## Config

Use [`configs/sam2.1_hiera_b+.yaml`](../configs/sam2.1_hiera_b+.yaml) or the copy inside `segment-anything-2/sam2/configs/sam2.1/`.

## Reference

Ravi et al., *SAM 2: Segment Anything in Images and Videos*, arXiv:2408.00714, 2024.
