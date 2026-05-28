# Aether Private AI Server — Deployment Guide

Deploy an OpenAI-compatible LLM on your own hardware. Data never leaves your network.

## API Reference

### `POST /v1/completions`

OpenAI-compatible text completion. Uses the same request/response shape as
`/v1/completions` from the OpenAI REST API.

#### Request body (JSON)

| Field        | Type    | Required | Default | Description |
|-------------|---------|----------|---------|-------------|
| `model`     | string  | yes      | —       | Model identifier (must match the loaded GGUF) |
| `prompt`    | string  | yes      | —       | Input text to complete |
| `max_tokens`| integer | no       | 512     | Maximum tokens in the response |
| `temperature`| float  | no       | 0.7     | Sampling temperature (0 = greedy) |
| `top_p`     | float   | no       | 0.9     | Nucleus sampling probability mass |
| `stop`      | string or string[] | no | null | Stop sequences |
| `stream`    | boolean | no       | false   | Enable SSE streaming |
| `echo`      | boolean | no       | false   | Repeat the prompt in the response |
| `n`         | integer | no       | 1       | Number of completions to generate |

#### Response (JSON)

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1700000000,
  "model": "llama-3.2-3b",
  "choices": [
    {
      "text": "completion text here",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170
  }
}
```

### `GET /health`

Returns server health status.

```json
{
  "status": "ok",
  "version": "0.2.0",
  "uptime_secs": 3600
}
```

### `GET /ready`

Returns readiness (model loaded, accepting traffic).

```json
{ "ready": true }
```

Status code 503 when the model is still loading.

### `GET /metrics`

Prometheus-format metrics for monitoring and alerting:

```
# HELP aether_uptime_seconds Server uptime
# TYPE aether_uptime_seconds gauge
aether_uptime_seconds 1234.56
# HELP aether_requests_served_total Total requests served
# TYPE aether_requests_served_total counter
aether_requests_served_total 42
# HELP aether_requests_per_second Request rate
# TYPE aether_requests_per_second gauge
aether_requests_per_second 0.0340
# HELP aether_model_info Static model metadata
# TYPE aether_model_info gauge
aether_model_info{model="llama-3.2-3b",cpu_only="true"} 1
```

### Authentication

All `/v1/*` endpoints require a Bearer token set via `AETHER_API_KEY`:

```
Authorization: Bearer <your-api-key>
```

### Environment variables

| Variable           | Default       | Description |
|--------------------|---------------|-------------|
| `AETHER_MODEL_PATH` | `model.gguf` | Path to GGUF model file |
| `AETHER_API_KEY`   | —             | Required bearer token |
| `AETHER_HOST`      | `127.0.0.1`  | Bind address |
| `AETHER_PORT`      | `8080`        | Listen port |
| `AETHER_CPU_ONLY`  | `true`        | Disable GPU inference |
| `AETHER_MAX_TOKENS`| `512`         | Per-request token cap |
| `AETHER_RATE_LIMIT`| `60`          | Requests per minute (0 = off) |

## Requirements

- **OS**: macOS (Apple Silicon recommended) or Linux x86_64
- **RAM**: 8 GB minimum for 1–3B models; 16 GB+ for 7B Q4
- **Rust**: stable toolchain (2021 edition)
- **Model**: GGUF file (e.g. Llama 3.2 3B Q4_K_M)

## Quick start (development)

```bash
# Build release server
cargo build --release --bin aether-server

# Set your model path and API key
export AETHER_MODEL_PATH=/path/to/model.gguf
export AETHER_API_KEY=$(openssl rand -hex 24)
export AETHER_CPU_ONLY=1   # recommended for stable production (default)

# Run
./target/release/aether-server
```

Server listens on `http://127.0.0.1:8080` by default.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AETHER_MODEL_PATH` | `model.gguf` | Path to GGUF model |
| `AETHER_API_KEY` | *(none)* | Bearer token for `/v1/*` routes |
| `AETHER_HOST` | `127.0.0.1` | Bind address |
| `AETHER_PORT` | `8080` | Listen port |
| `AETHER_CPU_ONLY` | `true` | `1`/`true` = CPU inference (stable) |
| `AETHER_MAX_TOKENS` | `512` | Max tokens per request |
| `AETHER_RATE_LIMIT` | `30` | Global requests/min (0 = unlimited) |
| `AETHER_MAX_CONCURRENCY` | `1` | Max concurrent inference requests (increase for GPU) |
| `AETHER_EXPECTED_SHA256` | *(none)* | Verify model SHA-256 before loading (hex string) |
| `RUST_LOG` | `aether=info` | Log filter |

## Health checks

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
```

## Chat completion (OpenAI-compatible)

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $AETHER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model.gguf",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128
  }'
```

## Production checklist

- [ ] Set `AETHER_API_KEY` to a strong random value
- [ ] Bind to internal IP only, or put behind reverse proxy (nginx/Caddy) with TLS
- [ ] Keep `AETHER_CPU_ONLY=1` unless GPU path is validated for your model
- [ ] Document expected tok/s and RAM for your hardware + model
- [ ] Run smoke test: `deploy/smoke_test.sh`
- [ ] Install as service (systemd or launchd below)

## macOS (launchd)

1. Edit `deploy/com.aether.server.plist` — set `AETHER_MODEL_PATH` and `AETHER_API_KEY`
2. Copy binary to `/usr/local/bin/aether-server`
3. Install plist:

```bash
sudo cp deploy/com.aether.server.plist /Library/LaunchDaemons/
sudo launchctl load /Library/LaunchDaemons/com.aether.server.plist
```

Logs: `/var/log/aether-server.log`

## Linux (systemd)

1. Edit `deploy/aether-server.service` — set `Environment=` lines
2. Install:

```bash
sudo cp target/release/aether-server /usr/local/bin/
sudo cp deploy/aether-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now aether-server
```

## Supported scope (v1 deployment)

| Included | Not included |
|----------|--------------|
| Single model, single server process | Multi-tenant SaaS |
| Concurrent request handling (semaphore-limited) | High-throughput batching |
| GGUF Llama-family models + extensible arch registry | Every quant/arch combo |
| CPU-stable inference | GPU Metal (opt-in, beta) |
| Prometheus `/metrics` endpoint | Per-request authz / multi-user |
| SHA-256 model integrity verification | Automated model download |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Model file not found` | Check `AETHER_MODEL_PATH` |
| `401 Unauthorized` | Pass `Authorization: Bearer <AETHER_API_KEY>` |
| Slow first start | Model load is one-time; large models take 10–30s |
| OOM | Use smaller model or Q4 quant; enable streaming for huge models |

## Smoke test

```bash
AETHER_MODEL_PATH=/path/to/model.gguf \
AETHER_API_KEY=test-key \
./deploy/smoke_test.sh
```
