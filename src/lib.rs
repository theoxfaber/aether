//! Aether — heterogeneous compute runtime with deferred DAG compilation,
//! automatic fusion scheduling, reverse-mode autograd, and an LLM inference
//! engine targeting CPU (scalar + SIMD) and GPU (WGPU/Metal).
//!
//! # Architecture overview
//!
//! | Layer | Module | Role |
//! |-------|--------|------|
//! | **User API** | [`graph`] | Deferred compute DAG (`Graph` + `GraphTensor`) with method chaining |
//! | **Autograd** | [`gradcheck`] | Reverse-mode AD — call `.backward()` on any scalar |
//! | **Compiler** | [`compiler`] | Simplify → CSE → DCE → constant-fold → DCE → layout-opt |
//! | **Scheduler** | [`scheduler`] | Cost-model fusion (MatMul+ReLU, MatMul+Add), memory-aware layer planning |
//! | **Codegen** | [`codegen`] | WGSL AST builder + pipeline cache for fused GPU kernels |
//! | **Memory** | [`memory`] | Buffer registry, LRU eviction, prefetch, arena planning |
//! | **Backends** | [`backend`] | CPU (ndarray+SIMD) / WGPU (Metal/Vulkan/DX12) / CUDA (stub) |
//! | **Inference** | [`inference`] | GGUF loader, quant matmul, KV cache, tokenizer, server |
//! | **Quantization** | [`quant`] | Q2_K … Q8_0 matmul kernels (CPU: scalar/NEON/AVX2, GPU: f32 fallback) |
//! | **Training** | [`optimizer`], [`mixed_precision`] | AdamW, SGD, GradScaler, checkpointing |
//! | **Python** | `python` (feature-gated) | PyO3 bindings for inference |
//!
//! # Quick start
//!
//! ```rust,ignore
//! use aether::{Graph, GraphTensor, Shape, Device};
//!
//! let g = Graph::new();
//! let x = g.tensor(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
//! let w = g.tensor(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));
//! let y = x.add(w);
//! let result = y.run(Device::Cpu).unwrap();
//! ```
//!
//! # Feature flags
//!
//! - `python` — PyO3 bindings
//! - `gpu-tests` — GPU integration tests (disabled by default)

#![deny(unsafe_code)]
#![cfg_attr(test, allow(dead_code))]
#![allow(
    clippy::too_many_arguments,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::needless_range_loop,
    clippy::manual_clamp,
)]

use thiserror::Error;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Error types returned by the Aether runtime.
#[derive(Error, Debug, Clone)]
pub enum Error {
    /// Dimension or shape mismatch.
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    /// Dimension rank mismatch.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    /// Errors occurring during runtime execution of operations.
    #[error("Execution error: {0}")]
    ExecutionError(String),
    /// Frame budget exceeded; decode was partial.
    /// The `usize` is the number of layers that completed before the budget was hit.
    #[error("Frame budget exceeded (completed {0} layers)")]
    BudgetExceeded(usize),
}

/// Supported hardware target devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// Host CPU execution backend.
    Cpu,
    /// Nvidia CUDA backend (uses CPU fallback if CUDA is unavailable).
    Cuda,
    /// GPU backend utilizing WGPU (Metal on macOS, Vulkan on Linux, DX12 on Windows).
    Wgpu,
    /// Automatically selects the best available GPU backend, falling back to CPU.
    Auto,
}

/// Hardware backend abstraction.
///
/// Every [`Op`](graph::Op) variant has a dedicated method on [`Backend`] with
/// a default fallback to [`Backend::execute`].  Override per-op for native
/// performance.
///
/// # Safety
///
/// The CPU backend uses `unsafe` to call BLAS FFI functions and NEON/AVX2
/// SIMD intrinsics that have no safe Rust equivalent.
pub mod backend;
/// Low-level BLAS bindings (accelerate / openblas).
///
/// # Safety
///
/// Calls `extern "C"` FFI functions (Apple Accelerate or platform BLAS).
/// These are inherently unsafe because they take raw pointers and have
/// no safe Rust wrapper.
#[allow(unsafe_code)]
pub mod blas;
/// C FFI bindings for Python/Node.js interop.
///
/// # Safety
///
/// Every function in this module is an FFI entry point.  It receives raw
/// pointers from C callers and must dereference them, allocate/deallocate
/// across the FFI boundary, and reconstruct Rust types from raw memory.
/// None of this can be done without `unsafe`.
#[allow(unsafe_code)]
pub mod c_api;
/// WGSL kernel code generation and pipeline caching.
pub mod codegen;
/// DAG compiler: simplification, CSE, DCE, constant folding, layout optimization.
pub mod compiler;
/// Numerical gradient checking (`gradcheck`) for autograd verification.
pub mod gradcheck;
/// Deferred-compute DAG — the primary user-facing API.
///
/// Create a [`Graph`](graph::Graph), add tensors with [`graph::Graph::tensor`],
/// chain operations, then call [`graph::GraphTensor::run`] to execute.
pub mod graph;
/// LLM inference engine: GGUF loader, quantized matmul, KV cache, tokenizer,
/// memory-aware layer scheduling, and OpenAI-compatible HTTP server.
pub mod inference;
/// GGUF / safetensors model weight loading and dequantization.
pub mod loader;
/// GPU memory registry, LRU eviction, prefetch scheduler, arena planning.
pub mod memory;
/// FP16 gradient scaling for mixed-precision training.
pub mod mixed_precision;
/// Neural network layer primitives (Linear, RMSNorm, LayerNorm, RoPE, etc.).
pub mod nn;
/// Optimizers: SGD (momentum, weight decay) and AdamW.
pub mod optimizer;
/// Parallel iteration utilities (scoped thread pools).
pub mod parallel;
/// Quantized matmul kernels for all GGUF formats.
///
/// CPU paths use scalar, NEON (aarch64), and AVX2 (x86_64) backends.
/// GPU quantized matmul currently dequantizes to f32 before dispatch.
pub mod quant;
/// Runtime execution dispatcher: plain ops, fused ops, and GPU command encoding.
pub mod runtime;
/// Execution scheduling: graph fusion pass, memory-aware layer scheduling,
/// async prefetch, and heterogeneous device dispatch.
pub mod scheduler;
/// Training checkpoint serialization (JSON-based, versioned).
pub mod serialization;
/// N-dimensional array with shape, strides, and contiguous storage.
pub mod tensor;
/// SentencePiece / BPE tokenizer (loaded from GGUF).
pub mod tokenizer;
/// `tracing`-based diagnostic instrumentation.
pub mod trace;

#[cfg(feature = "python")]
pub mod python;

pub use backend::{Backend, CpuBackend, CudaBackend, WgpuBackend};
pub use codegen::{ast::Expr, cache::PipelineCache, WgslKernelBuilder};
pub use compiler::GraphCompiler;
pub use gradcheck::gradcheck;
pub use graph::{Edge, Graph, GraphTensor, Node, Op};
pub use memory::prefetch::PrefetchScheduler;
pub use memory::registry::BufferRegistry;
pub use mixed_precision::GradScaler;
pub use optimizer::{AdamW, Sgd};
pub use scheduler::{
    AsyncPrefetchScheduler, DynamicScheduler, ExecutionPlan, FusedOp, FusionPass,
    MemoryAwareScheduler, ScheduledOp, ScheduledStep, Scheduler, SimpleScheduler,
};
pub use serialization::{
    load_checkpoint, load_weights, load_weights_into_graph, restore_weights_from_checkpoint,
    save_checkpoint, save_weights, SerializedAdamWState, SerializedTensor, TrainingCheckpoint,
    VersionedWeights,
};
pub use tensor::{AnyData, Dtype, GpuTensor, Shape, Tensor, TensorId};

#[cfg(feature = "python")]
#[pymodule]
fn aether_inference(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::aether_module(m)
}
