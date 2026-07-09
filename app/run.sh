#!/usr/bin/env bash
# Launch the Sentinel app. Run from the repo root (parent of app/).
#   ./app/run.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate the project venv if present.
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "Starting Sentinel on http://${SENTINEL_HOST:-127.0.0.1}:${SENTINEL_PORT:-8000}"
exec python -m app.backend.main
