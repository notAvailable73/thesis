#!/usr/bin/env bash
# Thin wrapper over scripts/run_mvt_grid.py — implementation.txt Section 10.1
# names a `.sh` as the deliverable, but the actual driver is the Python
# script (Step 9 found subprocess output can silently vanish on hosted
# notebooks; see scripts/run_mvt_grid.py's module docstring). Forwards every
# argument unchanged, e.g.:
#   scripts/run_mvt_grid.sh --resume --only "dataset=cifar_fs,shots=5"
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_mvt_grid.py "$@"
