#!/usr/bin/env bash
# Run the Step 1 cells via the new clean entry points.
# Reproduces Step 1's evidential + softmax pair using YAML configs.
set -euo pipefail

CONFIGS=(
    "configs/exp_step1.yaml"
    "configs/exp_step1_softmax.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    echo "=== TRAIN $cfg ==="
    python scripts/train.py --config "$cfg"
    echo "=== EVAL  $cfg ==="
    python scripts/evaluate.py --config "$cfg"
done
