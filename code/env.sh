#!/usr/bin/env bash
# Source before running pipelines: source code/env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SAM2_LER_ROOT="$ROOT"
export PYTHONPATH="$ROOT/code/sam2_ler:$ROOT/code/pipelines:$ROOT/code/probes:$ROOT/code/analysis:${PYTHONPATH:-}"
export PYTHON="${PYTHON:-python3}"
