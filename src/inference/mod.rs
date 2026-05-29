/// LLM inference engine: model loading, quantized matmul, KV cache,
/// autoregressive decoding, and OpenAI-compatible HTTP server.
///
/// # Pipeline
///
/// 1. [`GGUFLoader`](crate::loader::gguf::GGUFLoader) parses a `.gguf` file.
/// 2. [`LlamaConfig::from_gguf`](model_loader::LlamaConfig::from_gguf) reads
///    hyper-parameters from metadata.
/// 3. [`LlamaModel::from_gguf`](model_loader::LlamaModel::from_gguf) loads
///    weights (lazy or eager).
/// 4. [`LlamaRunner`](runner::LlamaRunner) wraps everything: tokenizer,
///    prefill, decode step, sampling.
///
/// # GPU support
///
/// Quantized weights are dequantized to f32, transposed to row-major, and
/// uploaded to WGPU buffers.  Layer scheduling decides which layers run on
/// GPU vs CPU based on available VRAM.
///
/// # Architecture registry
///
/// [`ArchitectureLoader`] allows registering custom model architectures.
/// Default: Llama, Mistral, Phi3, Qwen2, Gemma2, DeepSeek2.
pub mod arch_registry;
pub mod kv_cache;
pub mod layer_cache;
pub mod model_loader;
pub mod runner;
pub mod telemetry;

pub use arch_registry::{load_model, register_loader, ArchitectureLoader};
pub use kv_cache::StaticKVCache;
pub use model_loader::{
    ArchConfig, LlamaConfig, LlamaLayerWeights, LlamaModel, ModelArchitecture, QuantWeight,
};
pub use runner::{InferenceContext, LlamaRunner, LoadOptions, RunnerGuard, RunnerPool};
pub use telemetry::{ExecutionTelemetry, LayerTelemetry, Stopwatch};
