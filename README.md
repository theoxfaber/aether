# Aether

[![CI](https://github.com/theoxfaber/aether/actions/workflows/ci.yml/badge.svg)](https://github.com/theoxfaber/aether/actions/workflows/ci.yml)

A Rust-native heterogeneous compute runtime with an optimizing scheduler, automatic differentiation, and LLM inference engine. Write compute operations once — Aether automatically schedules, fuses, and executes them across CPU and GPU (WGPU/Metal) without manual memory management.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  User API (Graph + GraphTensor)                         │
│  Deferred compute DAG with method chaining              │
├─────────────────────────────────────────────────────────┤
│  Autograd (reverse-mode AD via .backward())             │
├─────────────────────────────────────────────────────────┤
│  Graph Compiler                                         │
│  simplify → CSE → DCE → constant_fold → DCE → layout   │
├─────────────────────────────────────────────────────────┤
│  Scheduler & Fusion Pass                                │
│  Cost-model-driven MatMul+ReLU, MatMul+Add,             │
│  ElementwiseChain fusion into single WGSL kernels       │
├─────────────────────────────────────────────────────────┤
│  WGSL AST Codegen & PipelineCache                       │
├─────────────────────────────────────────────────────────┤
│  Memory Layer                                           │
│  BufferRegistry, LRU eviction, prefetch scheduler,      │
│  static arena planning, liveness analysis               │
├─────────────────────────────────────────────────────────┤
│  Hardware Backends                                      │
│  CPU (ndarray/parallel) | GPU (WGPU/Metal) | CUDA stub  │
├─────────────────────────────────────────────────────────┤
│  LLM Inference Engine                                   │
│  GGUF loader, quant matmul, tokenizer,                  │
│  KV cache, memory-aware layer scheduling, server        │
└─────────────────────────────────────────────────────────┘
```

## Supported Operations

### Matrix
- `matmul`, `batched_matmul`, `transpose`, `batched_transpose`, `reshape`

### Elementwise (all with NumPy-style broadcasting)
- `add`, `sub`, `mul`, `div`, `tanh`, `sigmoid`, `exp`, `sqrt`, `neg`

### Reductions
- `sum_all` (full reduction), `sum_dim` (axis reduction)

### Activations & Normalization
- `relu`, `softmax`, `layernorm`, `rmsnorm`

### Neural Network
- `conv2d` (NCHW, OIHW), `max_pool2d`, `avg_pool2d`
- `concat`, `slice`, `cast` (F32↔F16↔BF16)

### Attention
- Scaled dot-product `attention`, `causal_attention`, `multi_head_attention`, `flash_attention`

### Gradient Ops (autograd internals)
- `softmax_grad`, `step`, `slice_grad`, `layernorm_grad_{x,w,b}`, `conv2d_grad_{x,w,b}`, `max_pool2d_grad`, `avg_pool2d_grad`, `attention_grad_{q,k,v}`

## Autograd

Full reverse-mode automatic differentiation. Call `.backward()` on any scalar tensor to build the gradient graph:

```rust
let g = Graph::new();
let x = g.tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
let w = g.tensor(vec![2.0, 0.0, 0.0, 2.0], Shape::new(vec![2, 2]));
let y = x.matmul(w).softmax().sum_all();
let grads = y.backward()?;
let dx = grads[&x.id()].run(Device::Cpu)?;
```

Correctness verified via numerical gradient checking (`gradcheck`).

## Graph Compiler

Six optimization passes run via `graph.compile()` (or `graph.set_compile_on_run(true)`):

| Pass | Description |
|------|-------------|
| **Simplify** | Algebraic identities (x+0→x, x\*1→x, x\*0→0, etc.) — skips computed (non-constant) operands |
| **CSE** | Common subexpression elimination — deduplicates identical subgraphs |
| **DCE** | Dead code elimination — removes unreachable nodes (runs twice: after CSE and after constant fold) |
| **Constant Fold** | Evaluates constant subgraphs at compile time |
| **Layout Optimize** | Pre-transposes constant weight matrices for aligned memory access |

## Kernel Fusion

The `FusionPass` detects fusible operation chains and checks a hardware cost model before fusing:

- **MatMul → ReLU**: Fuses into a single tiled WGSL kernel
- **MatMul → Add (bias)**: Fuses into a single WGSL kernel
- **Elementwise chains** (Relu→Tanh→Sigmoid→...): Fuses into one kernel via AST codegen

The cost model evaluates memory bandwidth savings vs. compute overhead using real hardware characteristics (tested 1.46x speedup on Apple M2 at 1024×1024).

## Memory Management

### BufferRegistry
- Tracks CPU/GPU/Both residency per tensor
- LRU eviction under soft GPU memory limit
- Automatic host↔device upload/download
- **Arena allocation**: compiled in but **disabled at runtime** (pending WebGPU binding separation — do not rely on this for performance)

### Prefetch Scheduler
- Overlaps host→device uploads with compute
- Eager eviction of intermediate tensors

### Liveness Analysis
- Computes exact last-use per tensor for precise eviction

### Static Memory Planner
- Interval-coloring allocation for CPU and GPU arenas
- 256-byte alignment for GPU buffers
- Zero-copy Reshape aliasing

## Training

| Feature | Details |
|---------|---------|
| **SGD** | Momentum, weight decay |
| **AdamW** | Decoupled weight decay (Loshchilov & Hutter 2019) |
| **GradScaler** | Mixed-precision (fp16) gradient scaling |
| **gradcheck** | Numerical gradient verification |
| **Checkpointing** | Versioned JSON format, save/load weights + optimizer state |

## Neural Network Layers

| Layer | Description |
|-------|-------------|
| `Linear` | Dense layer with Kaiming init |
| `RMSNorm` | Root Mean Square Normalization |
| `LayerNorm` | Layer Normalization |
| `KVCache` | Autoregressive key-value cache |
| `RoPE` | Rotary Position Embedding |
| `LlamaDecoderLayer` | Full decoder: RMSNorm → RoPE → GQA → FlashAttn → SiLU-MLP |
| `TransformerBlock` | Pre-LN transformer: LayerNorm → MHA → MLP |

## LLM Inference Engine

### GGUF v3 Loader
- Reads GGUF files with full metadata and tensor parsing
- Supports architectures: Llama, Mistral, Phi3, Qwen2, Gemma2, DeepSeek2
- Extensible architecture registry (`ArchitectureLoader` trait + `register_loader()`)
- Optional SHA-256 integrity verification: set `AETHER_EXPECTED_SHA256=<hex>` to verify model checksum before loading

### Quantization Formats
`F32`, `F16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`, `I8`, `I16`, `I32`

### Quantized MatMul Kernels
- Inline dequantization (no full materialization of f32 weights)
- Q8_0 (32-element blocks, f16 scale)
- Q4_K (256-element super-blocks, per-sub-block scale+min)
- Q5_K (256-element super-blocks)
- Q6_K (256-element super-blocks, per-16-element int8 scale)
- Parallelized with rayon

### Dequantization
All major formats supported: F32, F16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, Q8_K, I8, I16, I32

### LlamaRunner
Full autoregressive inference with:

| Feature | Details |
|---------|---------|
| **Tokenization** | SentencePiece/BPE with byte-fallback `<0xNN>` decoding |
| **KV Cache** | Static pre-allocated cache, O(1) per-token append |
| **Prefill** | Full-sequence forward pass |
| **Decode** | Single-token steps with KV reuse |
| **Sampling** | Temperature, top-k, top-p, repetition penalty |
| **GPU Support** | Weights dequantized→transposed→uploaded to WGPU buffers |
| **Memory-Aware Layer Scheduling** | Greedy: first N layers on GPU, rest on CPU |
| **Streaming** | LRU layer cache for models exceeding RAM |
| **Batched Prefill** | Pre-allocated scratch buffers for multi-token prefill |
| **Telemetry** | Per-layer timing, throughput, memory usage |
| **llama.cpp Comparison** | `bench` subcommand runs and compares against llama-cli |

## GPU Limitations

**CPU fallback for unsupported GPU ops.** Attention, pooling, and several other complex operations currently download tensor data from GPU to CPU, compute on the host, then re-upload the result. This round-trip (GPU→CPU→GPU) can dominate inference time for transformer models, especially during prefill. Future work includes native WGSL implementations for these ops.

**No native quantized WGSL shaders exist.** All 13 WGSL shaders operate on f32 data. Quantized matmul on GPU dequantizes to f32 before dispatch — correct but slower than a native quantized kernel would be.

**Arena allocation is disabled.** Compiled into the binary but not active at runtime (pending WebGPU binding separation).

### Inference Server
- Axum-based HTTP server with OpenAI-compatible API
- Streaming SSE support via Server-Sent Events
- Chat completions, text completions, and model listing endpoints
- Token-level streaming with early stop on EOS
- Concurrent request handling via `AETHER_MAX_CONCURRENCY` (semaphore-based, default 1)
- **Observability**: `/health`, `/ready`, `/metrics` (Prometheus format) endpoints
- **Rate limiting**: configurable per-minute global rate limit (default 60)
- **Authentication**: optional API key via `AETHER_API_KEY` on `/v1/*` routes
- **Production deployment**: configure via environment variables (`AETHER_*`); `AETHER_CPU_ONLY=1` for stable CPU-only serving

### CLI

```bash
# Run inference
cargo run -- run -m model.gguf -p "The meaning of life is" -n 200

# Benchmark
cargo run -- bench -m model.gguf --compare-llama-cpp

# Start server
cargo run --bin aether-server -- --model model.gguf
```

## Getting Started

### Prerequisites
- Rust stable toolchain (2021 edition)
- macOS with Apple Silicon or WGPU-compatible GPU

### Run tests
```bash
cargo test
```

### Run examples
```bash
# Basic matmul + relu
cargo run --example benchmark

# LLM inference with GGUF model
cargo run --example generate
```

### Python Bindings
```bash
cargo rustc --features python -- -C link-arg=-undefined -C link-arg=dynamic_lookup
cp target/debug/libaether.dylib aether.so
python test_bindings.py
```

## Verification

- **Integration tests**: `cargo test` covers correctness of all ops on CPU and WGPU, memory limits, LRU eviction, prefetching, AST compilation, broadcasting, softmax, and autograd
- **Numerical gradcheck**: Finite-difference verification of all gradient formulas
- **Criterion benchmarks**: `cargo bench` for matmul, fusion, and end-to-end inference
- **llama.cpp comparison**: `bench` subcommand validates output quality and performance against reference
