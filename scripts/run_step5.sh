#!/usr/bin/env bash
# Step 5 (Phase 3) runner: train + 600-episode eval for all 8 configs
# (LoRA / BitFit / Full-FT / Linear-Probe) x (evidential / softmax).
#
# Resumable: skips a config whose results/phase3_*_metrics.json already exists.
# W&B is disabled (base.yaml default). Add --use-tinyimagenet to the eval line
# for the strongest near-OOD comparison (needs the TinyImageNet zip cached;
# see instructions.txt gotcha (b) on Colab+Drive).
#
# Usage:  bash scripts/run_step5.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SUFFIX="phase3"
NUM_EPISODES=600
EXTRA_EVAL_FLAGS="--use-tinyimagenet"   # set to "" to skip TinyImageNet near-OOD

# config-basename : expected result-file head-descriptor (prototype-<interp>)
declare -A HEAD_DESC=(
  [exp_phase3_lora_evidential]="lora_prototype-evidential"
  [exp_phase3_lora_softmax]="lora_prototype-softmax"
  [exp_phase3_bitfit_evidential]="bitfit_prototype-evidential"
  [exp_phase3_bitfit_softmax]="bitfit_prototype-softmax"
  [exp_phase3_full_ft_evidential]="full_ft_prototype-evidential"
  [exp_phase3_full_ft_softmax]="full_ft_prototype-softmax"
  [exp_phase3_linear_probe_evidential]="linear_probe_prototype-evidential"
  [exp_phase3_linear_probe_softmax]="linear_probe_prototype-softmax"
)

for name in \
  exp_phase3_lora_evidential exp_phase3_lora_softmax \
  exp_phase3_bitfit_evidential exp_phase3_bitfit_softmax \
  exp_phase3_full_ft_evidential exp_phase3_full_ft_softmax \
  exp_phase3_linear_probe_evidential exp_phase3_linear_probe_softmax
do
  cfg="configs/${name}.yaml"
  out="results/${SUFFIX}_${HEAD_DESC[$name]}_metrics.json"
  if [[ -f "$out" ]]; then
    echo "== SKIP ${name} (found ${out}) =="
    continue
  fi
  echo "== TRAIN ${name} =="
  python scripts/train.py --config "$cfg" --wandb-mode disabled
  echo "== EVAL ${name} =="
  python scripts/evaluate.py --config "$cfg" \
    --num-episodes "$NUM_EPISODES" --wandb-mode disabled \
    --results-suffix "$SUFFIX" $EXTRA_EVAL_FLAGS
done

echo "== DONE: results/${SUFFIX}_*_metrics.json =="
ls -1 results/${SUFFIX}_*_metrics.json 2>/dev/null || true
