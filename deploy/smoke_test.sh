#!/usr/bin/env bash
set -euo pipefail

HOST="${AETHER_HOST:-127.0.0.1}"
PORT="${AETHER_PORT:-8080}"
BASE="http://${HOST}:${PORT}"
API_KEY="${AETHER_API_KEY:-test-key}"

pass=0
fail=0

check() {
    local label="$1" expected="$2" got="$3"
    if [[ "$got" == "$expected" ]]; then
        echo "  PASS  $label"
        ((pass++))
    else
        echo "  FAIL  $label (expected: $expected, got: $got)"
        ((fail++))
    fi
}

echo "=== Aether Smoke Test ==="
echo ""

# 1. Health
echo "--- Health ---"
status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/health")
check "/health" "200" "$status"

# 2. Ready
echo "--- Ready ---"
status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/ready")
check "/ready" "200" "$status"

# 3. Unauthenticated
echo "--- Auth required ---"
status=$(curl -s -w "%{http_code}" -o /dev/null "$BASE/v1/completions")
check "/v1/completions (no auth)" "401" "$status"

# 4. Bad auth
echo "--- Bad auth ---"
status=$(curl -s -w "%{http_code}" -o /dev/null \
    -H "Authorization: Bearer bad-key" \
    "$BASE/v1/completions")
check "/v1/completions (bad auth)" "401" "$status"

# 5. Missing model
echo "--- Missing model (completions) ---"
status=$(curl -s -w "%{http_code}" -o /dev/null \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"hello","max_tokens":1}' \
    "$BASE/v1/completions")
check "/v1/completions (no model loaded)" "503" "$status"

# 6. 404 for unknown route
echo "--- 404 ---"
status=$(curl -s -w "%{http_code}" -o /dev/null "$BASE/v1/nonexistent")
check "/v1/nonexistent" "404" "$status"

# 7. 405 for OPTIONS on /v1/*
echo "--- CORS OPTIONS ---"
status=$(curl -s -X OPTIONS -w "%{http_code}" -o /dev/null \
    -H "Origin: http://example.com" \
    -H "Access-Control-Request-Method: POST" \
    "$BASE/v1/completions")
check "OPTIONS /v1/completions (CORS)" "200" "$status"

echo ""
echo "=== Results: $pass passed, $fail failed ==="
exit $fail
