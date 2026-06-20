# Reproducibility guide

## Environment

```bash
conda create -n sam2_ler python=3.11 -y && conda activate sam2_ler
cd sam2_ler_github
pip install -r requirements.txt
pip install -e ../segment-anything-2   # clone separately
source code/env.sh
```

## Regenerate main tables

```bash
python code/pipelines/paper_de_report.py --exp_dir results/paper_de
# Output: results/paper_de/tables/table_main_*.csv
```

## Regenerate manuscript figures

```bash
python code/pipelines/paper_sensors_figures.py \
  --exp_dir results/paper_de \
  --asi_report results/thesis_validation/thesis_validation_report.json \
  --out_dir ../manuscript_sensors/figures
```

## Expected runtime (single NVIDIA GPU)

| Task | Runs | Time |
|------|------|------|
| SAM2-LER @1% NEU, 1 seed | 1 | ~14 min |
| Full paper_de (300 runs) | 300 | ~5–10 h |
| Figure generation | — | <2 min |

## Randomness

- Pool/val: fixed `numpy.default_rng(20240618)`
- Subset: `frac_subset(pool, frac, seed)`
- Training shuffle: per-run seed
