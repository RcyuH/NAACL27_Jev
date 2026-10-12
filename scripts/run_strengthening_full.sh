#!/usr/bin/env bash
# Run from the repository root; existing prediction files are required.
set -euo pipefail
CONFIG_PATH="${1:-configs/reproduce.yaml}"
OUTPUT_PATH="${2:-results/strengthening_v1}"
WORKER_COUNT="${3:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Prefer THIS checkout, even if an old editable installation points elsewhere.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m jevcal.cli --config "$CONFIG_PATH" strengthen \
  --settings configs/strengthen.yaml \
  --output-dir "$OUTPUT_PATH" \
  --workers "$WORKER_COUNT"
