#!/usr/bin/env bash
# Evaluate the best checkpoint on the validation split (20 % of data_train).

set -e
CONFIG=${1:-config.yaml}
SPLIT=${2:-val}          # val | test

echo "==> Evaluating split: ${SPLIT}  (config: ${CONFIG})"
python -m src.training.eval --config "${CONFIG}" --split "${SPLIT}"
