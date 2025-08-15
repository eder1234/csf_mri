#!/usr/bin/env bash
# Train the 2-D UNet on the training set, writing logs / checkpoints to ./outputs/

set -e
CONFIG=${1:-config.yaml}

echo "==> Training with config: ${CONFIG}"
python -m src.training.train --config "${CONFIG}"
