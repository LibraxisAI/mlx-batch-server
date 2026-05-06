#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-10240}"
CORS="${CORS:-http://localhost:*,http://127.0.0.1:*,http://100.*:*,https://100.*:*}"

echo "Installing MLX Batch Server CLI from $ROOT"
uv sync --all-groups
uv tool install . --force --prerelease=allow
uv run python scripts/verify_operator_tools.py || true

if [[ -d .git ]]; then
  uv run pre-commit install
  uv run pre-commit install --hook-type pre-push
fi

cat <<EOF
Installed mlx-batch-server.

Start:
  mlx-batch-server --port ${PORT} --cors-allow-origins="${CORS}"

Operator panel:
  http://localhost:${PORT}/admin

Bundled operator tools:
  export PATH="${ROOT}/tools/bin/darwin-arm64:$PATH"
EOF
