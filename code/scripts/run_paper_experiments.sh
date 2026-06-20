#!/usr/bin/env bash
# =============================================================================
# run_paper_experiments.sh — 论文实验一键流水线 (建议挂 overnight)
#
# 内容:
#   Phase 0  确保 ASI 报告存在
#   Phase 1  Std-LoRA  30 epoch
#   Phase 2  Conv-LoRA 30 epoch
#   Phase 3  ASI-Guided Hybrid 30 epoch
#   Phase 4  论文表格 (CSV / LaTeX / Markdown)
#   Phase 5  Nature 风格配图 (PDF + PNG)
#
# 用法:
#   conda activate sam2_steel
#   cd ~/sam_research/sam2_steel_defect
#   bash run_paper_experiments.sh
#
# 可选环境变量:
#   EPOCHS=30          训练轮数
#   BATCH_SIZE=4       batch size
#   DATA_DIR=data/NEU-Seg
#   OUT=outputs/paper_exp
#   FORCE=1            强制重训 (删除已有 log)
#   SKIP_TRAIN=1       跳过训练, 只生成表格/图
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ---- 配置 ----
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
DATA_DIR="${DATA_DIR:-data/NEU-Seg}"
OUT="${OUT:-outputs/paper_exp}"
ASI_REPORT="${ASI_REPORT:-outputs/thesis_validation/thesis_validation_report.json}"
LOG_DIR="${OUT}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" &>/dev/null; then
    PYTHON=python3
fi

log "============================================================"
log "  Paper Experiment Pipeline"
log "  OUT=${OUT}  EPOCHS=${EPOCHS}  BATCH_SIZE=${BATCH_SIZE}"
log "  Master log: ${MASTER_LOG}"
log "============================================================"

maybe_clean() {
    local dir="$1"
    if [[ "${FORCE:-0}" == "1" && -d "$dir" ]]; then
        log "  FORCE=1 → 清理 ${dir}"
        rm -rf "$dir"
    fi
}

run_train() {
    local name="$1"
    local cmd="$2"
    local out_sub="$3"
    local out_path="${OUT}/${out_sub}"
    local train_log="${LOG_DIR}/${out_sub}.log"

    if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
        log "[SKIP] ${name} (SKIP_TRAIN=1)"
        return 0
    fi

    if [[ -f "${out_path}/training_log.json" && "${FORCE:-0}" != "1" ]]; then
        log "[SKIP] ${name} — 已有 ${out_path}/training_log.json"
        return 0
    fi

    maybe_clean "$out_path"
    log ">>> START ${name}"
    log "    cmd: ${cmd}"
    log "    log: ${train_log}"

    set +e
    eval "$cmd" 2>&1 | tee -a "$train_log"
    local rc=${PIPESTATUS[0]}
    set -e

    if [[ $rc -ne 0 ]]; then
        log "!!! FAILED ${name} (exit ${rc})"
        exit $rc
    fi
    log "<<< DONE ${name}"
}

# ---- Phase 0: ASI 报告 ----
if [[ ! -f "$ASI_REPORT" ]]; then
    log "[Phase 0] 生成 ASI 报告..."
    $PYTHON thesis_validation.py --mode analyze \
        --output_dir outputs/thesis_validation \
        2>&1 | tee -a "$MASTER_LOG"
else
    log "[Phase 0] ASI 报告已存在: ${ASI_REPORT}"
fi

# ---- Phase 1–3: 训练 ----
run_train "Std-LoRA ${EPOCHS}ep" \
    "$PYTHON train_multiclass.py \
        --lora_type standard \
        --output_dir ${OUT}/multiclass_std_30ep \
        --data_dir ${DATA_DIR} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR}" \
    "multiclass_std_30ep"

run_train "Conv-LoRA ${EPOCHS}ep" \
    "$PYTHON train_multiclass.py \
        --lora_type conv \
        --output_dir ${OUT}/multiclass_conv_30ep \
        --data_dir ${DATA_DIR} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR}" \
    "multiclass_conv_30ep"

run_train "ASI-Guided ${EPOCHS}ep" \
    "$PYTHON train_asi_guided.py \
        --asi_report ${ASI_REPORT} \
        --output_dir ${OUT}/asi_guided_30ep \
        --data_dir ${DATA_DIR} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR}" \
    "asi_guided_30ep"

# ---- Phase 4: 表格 ----
log "[Phase 4] 生成论文表格..."
$PYTHON paper_tables.py \
    --exp_dir "$OUT" \
    --asi_report "$ASI_REPORT" \
    2>&1 | tee -a "$MASTER_LOG"

# ---- Phase 5: 配图 ----
log "[Phase 5] 生成 Nature 风格配图..."
$PYTHON paper_figures.py \
    --exp_dir "$OUT" \
    --asi_report "$ASI_REPORT" \
    2>&1 | tee -a "$MASTER_LOG"

# ---- 汇总 ----
log ""
log "============================================================"
log "  ✅ Pipeline 完成!"
log "  训练结果:"
log "    ${OUT}/multiclass_std_30ep/"
log "    ${OUT}/multiclass_conv_30ep/"
log "    ${OUT}/asi_guided_30ep/"
log "  表格: ${OUT}/tables/"
log "  配图: ${OUT}/figures/"
log "  主日志: ${MASTER_LOG}"
log "============================================================"

# 打印 Table 2 预览
if [[ -f "${OUT}/tables/table2_main_results.csv" ]]; then
    log ""
    log "--- Table 2 Preview ---"
    column -t -s, "${OUT}/tables/table2_main_results.csv" 2>/dev/null | tee -a "$MASTER_LOG" || \
        cat "${OUT}/tables/table2_main_results.csv" | tee -a "$MASTER_LOG"
fi
