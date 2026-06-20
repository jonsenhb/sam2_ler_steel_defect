# Dataset download and layout

## NEU-Seg

- **Paper:** Song & Yan, *Applied Surface Science*, 2013  
- **Classes:** background, patches, inclusion, scratches (4-class semantic segmentation)  
- **Expected layout** (compatible with `dataset.py`):

```
datasets/neu_seg/
  images/training/*.jpg
  images/test/*.jpg
  annotations/training/*.png
  annotations/test/*.png
```

Download from the authors' NEU surface defect database or academic mirrors.  
Do **not** commit raw images to Git (use Zenodo or Git LFS for releases).

## Severstal

- **Source:** [Kaggle Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) (2019)  
- **Classes:** 4 defect types + background (5-class)  
- **Preprocessing:** use `prepare_severstal_data.py` from the full repository after placing Kaggle files under `data/severstal/raw/`.

```
datasets/severstal/
  train_images/
  train_masks/
  val_images/
  val_masks/
```

## Custom datasets (future release)

Place self-collected steel defect data under [`custom/`](custom/) following [`custom/schema.json`](custom/schema.json) and [`custom/README.md`](custom/README.md).

Recommended publication path:
1. GitHub repo → metadata + download script  
2. **Zenodo** DOI for image archives  
3. Optional **Git LFS** for small pilot sets (<2 GB)
