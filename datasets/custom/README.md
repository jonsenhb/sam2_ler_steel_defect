# Custom steel defect dataset (placeholder)

This directory is reserved for **author-collected** hot-rolled / cold-rolled steel surface images.

## Naming convention

```
custom/
  {steel_grade}_{line_id}/
    images/{split}/*.jpg
    masks/{split}/*.png    # single-channel class IDs
    meta.json
```

## meta.json template

See [`schema.json`](schema.json). Required fields:

- `dataset_name`, `class_id_to_name`, `defect_class_ids`
- `split`: `train` / `val` / `test` file lists or ratios
- `capture`: camera model, resolution, lighting (for Sensors reproducibility)

## Annotation format

- PNG masks, pixel values = class ID (0 = background)  
- Same convention as NEU-Seg in [`code/sam2_ler/dataset.py`](../../code/sam2_ler/dataset.py)

## Train with SAM2-LER

```bash
source code/env.sh
python code/pipelines/paper_de_pipeline.py \
  --datasets custom --custom_dir datasets/custom/my_line \
  --methods sam2_lora --fracs 0.01 0.10 1.0 --seeds 0 1 2
```

*(Extend `dataset.py` with `custom` entry when data is ready.)*
