![CI](https://img.shields.io/github/actions/workflow/status/theoxfaber/aether/ci.yml?branch=master&label=CI&logo=github)
![Rust](https://img.shields.io/badge/Rust-1.85%2B-dea584?logo=rust)
![License](https://img.shields.io/badge/license-MIT-blue)
![GitHub last commit](https://img.shields.io/github/last-commit/theoxfaber/aether)

<h1 align="center">✦ Aether ✦</h1>

<p align="center">
  <em>A Rust-native heterogeneous compute runtime — DAG scheduler, autograd, WGSL fusion, and a production LLM inference engine.</em>
</p>

---

## Overview

Write compute operations once. Aether automatically schedules, fuses, and executes them across CPU and GPU (WGPU/Metal) without manual memory management.

```rust
let g = Graph::new();
let x = g.tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
let w = g.tensor(vec![2.0, 0.0, 0.0, 2.0], Shape::new(vec![2, 2]));
let y = x.matmul(w).softmax().sum_all();
let grads = y.backward()?;
let dx = grads[&x.id()].run(Device::Cpu)?;
```

---

## Architecture

```
┌───────────────────────────────────────────────────────┐
│  User API — Deferred compute DAG with chaining         │
├───────────────────────────────────────────────────────┤
│  Autograd — Reverse-mode AD (`.backward()`)            │
├───────────────────────────────────────────────────────┤
│  Compiler — Simplify → CSE → DCE → Fold → Layout      │
├───────────────────────────────────────────────────────┤
│  Fusion — Cost-model-driven MatMul+ReLU, chains       │
├───────────────────────────────────────────────────────┤
│  Memory — BufferRegistry, LRU eviction, arena planner │
├───────────────────────────────────────────────────────┤
│  Backends — CPU (ndarray/rayon) · GPU (WGPU/Metal)    │
├───────────────────────────────────────────────────────┤
│  LLM Inference — GGUF loader, quant matmul, server    │
└───────────────────────────────────────────────────────┘
```

---

## Compiler & Fusion

Six optimization passes run via `graph.compile()`:

| Pass | What it does |
|------|-------------|
| **Simplify** | Algebraic identities (x+0→x, x·1→x) — skips computed operands |
| **CSE** | Common subexpression elimination — deduplicates subgraphs |
| **DCE** | Dead code elimination — removes unreachable nodes |
| **Constant Fold** | Evaluates constant subgraphs at compile time |
| **Layout** | Pre-transposes constant weight matrices |

**Kernel fusion** detects fusible chains and checks a hardware cost model before emitting combined WGSL kernels — MatMul→ReLU, MatMul→Add, and elementwise chains. ~1.5× speedup on Apple M2 at 1024².

---

## Memory System

The memory layer tracks CPU/GPU/Both residency per tensor and manages the full lifecycle:

- **BufferRegistry** — LRU eviction under soft GPU memory limit, automatic host↔device transfer, pin support for in-flight tensors
- **Eviction** — Download to CPU when GPU budget is exceeded; `AtomicUsize` counter eliminates deadlock risk
- **Static Arena Planner** — Interval-coloring / first-fit-decreasing allocation with 256-byte alignment
- **Liveness Analysis** — Exact last-use computation for precise eviction timing
- **Prefetch Scheduler** — Overlaps host→device uploads with compute

---

## LLM Inference Engine

Loads GGUF v3 models and runs full autoregressive inference with LLaMA-family architectures (Mistral, Phi-3, Qwen2, Gemma2, DeepSeek2).

### Features

| Component | Details |
|-----------|---------|
| **GGUF Loader** | Full v3 metadata dispatch, mmap-backed zero-copy tensors, SHA-256 integrity check |
| **Quant MatMul** | NEON dotprod (Q4_K/Q6_K/Q8_0), AVX2 fallback, inline dequantization |
| **LlamaRunner** | RMSNorm → RoPE → GQA attention → SiLU-MLP — CPU, GPU, or hybrid |
| **KV Cache** | Static pre-allocated, O(1) per-token append, GPU sync |
| **Sampling** | Temperature, top-p nucleus, repetition penalty via `HashSet` O(1) |
| **Batched Prefill** | Zero heap allocation per decode step |
| **Streaming** | LRU layer cache for models exceeding RAM |
| **Telemetry** | Per-layer timing, throughput, memory |

### Quantization Formats

`F32` · `F16` · `Q4_0` · `Q4_1` · `Q5_0` · `Q5_1` · `Q8_0` · `Q8_1` · `Q2_K` · `Q3_K` · `Q4_K` · `Q5_K` · `Q6_K` · `Q8_K` · `I8` · `I16` · `I32`

### GPU Notes

- Quantized matmul on GPU dequantizes to f32 before dispatch (no native WGSL quantized shaders — contributions welcome)
- Dequant shaders handle F16 infinity/NaN correctly (returns ±inf or NaN instead of silent 0.0)
- Attention softmax uses workgroup-wide tree reduction (128 threads, not 1)
- Arena allocation compiled in but disabled at runtime pending WGPU binding separation

### Server

Axum-based HTTP server with OpenAI-compatible API:

```bash
cargo run --bin aether-server -- --model model.gguf       \
  --max-concurrency 4 --cpu-only --rate-limit 120          \
  --api-key "sk-..."                                       \
  --max-tokens 4096
```

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat (streaming SSE or JSON) |
| `POST /v1/completions` | Text completions |
| `GET /v1/models` | Loaded model info |
| `GET /health` · `GET /ready` | Liveness / readiness |
| `GET /metrics` | Prometheus format |

---

## CLI

```bash
# Run inference
cargo run -- run -m model.gguf -p "The meaning of life is" -n 200

# Benchmark against llama.cpp
cargo run -- bench -m model.gguf --compare-llama-cpp

# Interactive session
cargo run -- chat -m model.gguf
```

---

## Tests & Benchmarks

```bash
cargo test                    # CPU tests (default)
cargo test --features gpu-tests  # GPU tests (WGPU/Metal required)
cargo bench                   # Criterion benchmarks
```

- 70+ tests: op correctness, GGUF parsing, e2e synthetic model, LRU eviction
- Cosine similarity >0.99 between GPU and CPU decode paths
- Criterion benchmarks for quantized matmul (Q8_0, Q4_K, Q6_K, f32)

---

## Project Status

| Area | Status |
|------|--------|
| Computation Graph | ✓ DAG, ops, broadcasting, shape inference |
| Autograd | ✓ Reverse-mode, gradcheck-verified |
| Compiler | ✓ Simplify, CSE, DCE, constant folding, layout opt |
| GPU Backend | ✓ WGPU/Metal, 13 WGSL shaders, f32 pipelines |
| CUDA Backend | ✓ Stub (not implemented) |
| Memory Planner | ✓ Arena allocation (disabled), LRU eviction |
| GGUF Loader | ✓ v3 metadata, quantized tensors, SHA-256 |
| Quant Kernels | ✓ NEON dotprod (Q4_K/Q6_K/Q8_0), AVX2, scalar fallback |
| Inference | ✓ Prefill, decode, streaming, KV cache, GQA, RoPE |
| Server | ✓ HTTP/SSE, OpenAI API, Prometheus, rate limit, auth |
| Concurrency | ✓ `Arc<LlamaModel>` shared weights, semaphore admission |

---

<p align="center"><em>Built with Rust · ci: ubuntu-latest · clippy -D warnings · cargo fmt — zero warnings</em></p>
