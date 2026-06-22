#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Core env
: "${ROBOFLOW_API_KEY:?Set ROBOFLOW_API_KEY before running this legacy script}"
export ROBOFLOW_API_KEY
export ROBOFLOW_MODEL_ID="${ROBOFLOW_MODEL_ID:-parking-spaces-ezhxz/1}"
if [[ $# -ge 1 && -n "${1:-}" ]]; then
  export VIDEO_PATH="$1"
else
  : "${VIDEO_PATH:?Pass a video URL/path as the first argument or set VIDEO_PATH}"
  export VIDEO_PATH
fi
export SLOTS_SOURCE_IMAGE="${SLOTS_SOURCE_IMAGE:-CarParkFrameCropped.png}"

# Optional tuning
export INFER_EVERY_SEC="${INFER_EVERY_SEC:-2.0}"
export OCC_MIN_CONF="${OCC_MIN_CONF:-0.3}"
export SLOT_OCCUPIED_IOA="${SLOT_OCCUPIED_IOA:-0.2}"

echo "Using VIDEO_PATH: $VIDEO_PATH"

if lsof -iTCP:5001 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port 5001 is already in use. Stop the existing process and retry."
  exit 1
fi

./venv/bin/uvicorn backend_api:app --host 0.0.0.0 --port 5001 &
API_PID=$!

echo "Starting Cloudflare tunnel..."
if command -v cloudflared >/dev/null 2>&1; then
  cloudflared tunnel --url http://localhost:5001
else
  echo "cloudflared not found. Install with: brew install cloudflared"
  echo "API is running at http://localhost:5001"
  wait "$API_PID"
fi

echo "Shutting down API..."
kill "$API_PID" >/dev/null 2>&1 || true
