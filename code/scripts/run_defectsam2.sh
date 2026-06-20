#!/usr/bin/env bash
# DefectSAM2 全协议实验 (30 runs) + 对比报告
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/jonsen/miniconda3/envs/sam2_steel/bin/python}"
OUT_LOG="outputs/paper_defectsam2/pipeline.log"
mkdir -p outputs/paper_defectsam2

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$OUT_LOG"; }

log "======== DefectSAM2 full protocol start ========"
log "Python: $PYTHON"

# 冒烟: 单 run 验证
log "Smoke test (NEU 1% seed=0)..."
"$PYTHON" paper_defectsam2_pipeline.py \
    --datasets neu_seg \
    --fracs 0.01 \
    --seeds 0 \
    --force \
    2>&1 | tee -a "$OUT_LOG"

log "Smoke OK — launching full 30-run protocol"

"$PYTHON" paper_defectsam2_pipeline.py \
    --datasets neu_seg severstal \
    --fracs 0.01 0.05 0.10 0.25 1.0 \
    --seeds 0 1 2 \
    2>&1 | tee -a "$OUT_LOG"

log "Generating comparison report vs paper_de baseline..."
"$PYTHON" paper_defectsam2_report.py 2>&1 | tee -a "$OUT_LOG"

log "✅ All DefectSAM2 experiments done."
log "  Results: outputs/paper_defectsam2/"
log "  Report:  outputs/paper_defectsam2/comparison_report.md"
