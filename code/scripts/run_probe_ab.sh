#!/usr/bin/env bash
# Probe-A (CB-PEFT) + Probe-B (DSA-FPN) — paper_de 协议快速验证
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/jonsen/miniconda3/envs/sam2_steel/bin/python}"
OUT_LOG="outputs/probe_ab/pipeline.log"
mkdir -p outputs/probe_ab/cb_peft outputs/probe_ab/dsa_fpn

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT_LOG"; }

log "========================================"
log " Probe-A: CB-PEFT  (class-balanced image sampling)"
log "========================================"
"$PYTHON" probe_cb_peft.py \
    --dataset neu_seg --data_dir data/NEU-Seg \
    --fracs 0.01 0.10 --seed 0 \
    2>&1 | tee -a "$OUT_LOG"

log "========================================"
log " Probe-B: DSA-FPN  (decode spatial adapter)"
log "========================================"
"$PYTHON" probe_dsa_fpn.py \
    --dataset neu_seg --data_dir data/NEU-Seg \
    --fracs 0.01 0.10 --seed 0 \
    2>&1 | tee -a "$OUT_LOG"

log "✅ Probe-A/B 全部完成"
log "  CB:  outputs/probe_ab/cb_peft/probe_cb_summary.json"
log "  DSA: outputs/probe_ab/dsa_fpn/probe_dsa_summary.json"
