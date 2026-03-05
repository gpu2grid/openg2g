#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

python "$SCRIPT_DIR/gen_api_docs.py"

case "${1:-}" in
  build)
    zensical build --clean --strict
    ;;
  serve)
    zensical serve --strict -a 0.0.0.0:7778
    ;;
  *)
    echo "Usage: uv run bash scripts/docs.sh {build|serve}" >&2
    exit 1
    ;;
esac
