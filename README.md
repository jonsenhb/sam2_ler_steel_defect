# SAM2-LER — Code & Experiment Results

**Repository:** https://github.com/jonsenhb/sam2_ler_steel_defect


## Directory tree

```
sam2_ler_github/
├── code/
│   ├── env.sh              # PYTHONPATH setup — source before running
│   ├── sam2_ler/           # Core: train_multiclass.py (SAM2-LER / HRH)
│   ├── pipelines/          # paper_de_pipeline, reports, figures
│   ├── probes/             # Ablation probes (CB-PEFT, DSA, RAD loss)
│   ├── analysis/           # ASI / thesis validation
│   └── scripts/            # Shell runners
├── configs/                # SAM2.1 Hiera-B+ yaml
├── results/                # 364 JSON result files + tables
├── datasets/               # Download instructions + custom schema
└── docs/                   # Protocol, reproducibility, checkpoints
```

## Installation

```bash
conda create -n sam2_ler python=3.11 -y
conda activate sam2_ler
pip install -r requirements.txt

# SAM2 source (required)
git submodule add https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e . && cd ..

# Checkpoint (~309 MB) — NOT in repo
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_base_plus.pt \
  "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_base_plus.pt"
```

See [`docs/checkpoint.md`](docs/checkpoint.md) for details.

## Data

| Dataset | Link | Local path |
|---------|------|------------|
| NEU-Seg | [Applied Surface Science 2013](https://doi.org/10.1016/j.apsusc.2013.04.097) | `datasets/neu_seg/` |
| Severstal | [Kaggle 2019](https://www.kaggle.com/c/severstal-steel-defect-detection) | `datasets/severstal/` |

Run `python prepare_severstal_data.py` (from full repo) after downloading Kaggle CSVs.

## Reproduce main experiment (300 runs)

```bash
source code/env.sh
python code/pipelines/paper_de_pipeline.py \
  --datasets neu_seg severstal \
  --fracs 0.01 0.05 0.10 0.25 1.0 \
  --seeds 0 1 2 \
  --output_dir results/paper_de
```

Completed runs are skipped automatically. Full log: ~5–10 GPU hours.

## Method: SAM2-LER

- **Frozen** SAM2.1 Hiera-B+ encoder
- **LoRA** (rank=4) on attention → Δ_enc (~0.26M params)
- **Hiera-aligned Readout Head (HRH)** replaces mask decoder → Δ_dec (~0.46M params)
- **Total trainable**: ~0.72M

Protocol: [`docs/protocol.md`](docs/protocol.md)

## Results manifest

[`results/MANIFEST.json`](results/MANIFEST.json) lists all JSON files with SHA256 checksums.
