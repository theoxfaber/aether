#!/usr/bin/env bash
# Smoke test for aether-server. Requires a running server or starts one temporarily.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BINARY="${AETHER_SERVER_BIN:-$ROOT/target/release/aether-server}"
MODEL="${AETHER_MODEL_PATH:?Set AETHER_MODEL_PATH to a GGUF file}"
API_KEY="${AETHER_API_KEY:-smoke-test-key}"
HOST="${AETHER_HOST:-127.0.0.1}"
PORT="${AETHER_PORT:-18080}"
BASE="http://${HOST}:${PORT}"

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: model not found: $MODEL" >&2
  exit 1
fi

if [[ ! -x "$BINARY" ]]; then
  echo "Building aether-server..."
  cargo build --release --bin aether-server --manifest-path "$ROOT/Cargo.toml"
fi

STARTED=0
cleanup() {
  if [[ "$STARTED" -eq 1 ]] && [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if ! curl -sf "$BASE/health" >/dev/null 2>&1; then
  echo "Starting server on port $PORT..."
  AETHER_MODEL_PATH="$MODEL" \
  AETHER_API_KEY="$API_KEY" \
  AETHER_HOST="$HOST" \
  AETHER_PORT="$PORT" \
  AETHER_CPU_ONLY=1 \
  RUST_LOG=warn \
  "$BINARY" &
  SERVER_PID=$!
  STARTED=1
  for _ in $(seq 1 120); do
    if curl -sf "$BASE/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

echo "== Health =="
curl -sf "$BASE/health" | python3 -m json.tool

echo "== Ready =="
curl -sf "$BASE/ready" | python3 -m json.tool

echo "== Models =="
curl -sf -H "Authorization: Bearer $API_KEY" "$BASE/v1/models" | python3 -m json.tool

echo "== Chat completion =="
RESP=$(curl -sf -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Say hi in three words."}],"max_tokens":16,"temperature":0.1}' \
  "$BASE/v1/chat/completions")
echo "$RESP" | python3 -m json.tool

CONTENT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
if [[ -z "$CONTENT" ]]; then
  echo "ERROR: empty completion" >&2
  exit 1
fi

echo "== OK: got response: ${CONTENT:0:80} =="
