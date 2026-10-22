#!/usr/bin/env bash
# TMLR follow-up — download shard outputs into incoming/ (plan Task 10).
#
#   scripts/tmlr_pull.sh hf <hf-user>/bpeft-tmlr-runs
#       every shard zip any account uploaded (needs `huggingface-cli login` once)
#
#   scripts/tmlr_pull.sh kaggle <kaggle-user>/<notebook-slug> [<user>/<slug> ...]
#       the LATEST version's output of each notebook. Each account's API token
#       lives in ~/.kaggle/acct<N>/kaggle.json, matched by position (1st notebook
#       -> acct1, ...). Pull an account BEFORE its next session overwrites "latest".
#
# Then:  uv run --frozen python scripts/tmlr_collect.py --incoming incoming/
set -euo pipefail
mode="${1:?usage: tmlr_pull.sh hf <repo> | kaggle <user/slug>...}"; shift
mkdir -p incoming
case "$mode" in
  hf)
    repo="${1:?hf dataset repo, e.g. you/bpeft-tmlr-runs}"
    huggingface-cli download "$repo" --repo-type dataset --include "shards/*" --local-dir incoming/hf
    ;;
  kaggle)
    n=1
    for kernel in "$@"; do
      cfg="$HOME/.kaggle/acct$n"
      [ -f "$cfg/kaggle.json" ] || { echo "missing $cfg/kaggle.json" >&2; exit 1; }
      out="incoming/acct$n-$(echo "$kernel" | tr '/' '_')-$(date +%Y%m%d-%H%M%S)"
      mkdir -p "$out"
      KAGGLE_CONFIG_DIR="$cfg" kaggle kernels output "$kernel" -p "$out"
      echo "pulled $kernel -> $out"
      n=$((n + 1))
    done
    ;;
  *) echo "unknown mode $mode" >&2; exit 2 ;;
esac
