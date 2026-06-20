#!/bin/bash
# ============================================================
# run_all_paper_experiments.sh — 一键运行全部论文实验 + 生成图表
#
# 用法: nohup bash run_all_paper_experiments.sh > outputs/paper_all.log 2>&1 &
#
# 步骤:
#   1. 运行外部 SOTA 实验 (DDSNet*/MFF-Metal*/SME-DLV3+*/Hybrid-Trans*)
#      在 paper_de_pipeline.py 同一协议 (pool/val/seed/metric) 下
#   2. 生成含外部 SOTA 的完整论文级图表和表格
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

PYTHON=/home/jonsen/miniconda3/envs/sam2_steel/bin/python
OUT_DIR=outputs/paper_de

echo "========================================"
echo " Step 1: External SOTA experiments"
echo " (4 methods × 5 fracs × 3 seeds × 2 datasets = 120 runs)"
echo "========================================"
echo "[$(date)] Starting external SOTA..."

$PYTHON paper_external_sota.py \
    --datasets neu_seg severstal \
    --fracs 0.01 0.05 0.10 0.25 1.0 \
    --seeds 0 1 2 \
    --pool_size 1200 \
    --val_size 400 \
    --output_dir $OUT_DIR

echo ""
echo "[$(date)] External SOTA done."
echo ""

echo "========================================"
echo " Step 2: Generate publication figures & tables"
echo "========================================"
$PYTHON paper_de_report.py --exp_dir $OUT_DIR

echo ""
echo "========================================"
echo " Summary of completed experiments"
echo "========================================"
echo "JSON files:"
find $OUT_DIR -name '*.json' | wc -l
echo "Figures:"
ls -la $OUT_DIR/figures/ 2>/dev/null
echo "Tables:"
ls -la $OUT_DIR/tables/ 2>/dev/null

echo ""
echo "✅ [$(date)] All done!"
echo "Figures: $OUT_DIR/figures/"
echo "Tables:  $OUT_DIR/tables/"
