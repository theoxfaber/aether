/// LlamaRunner: full autoregressive inference engine.
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use tracing::{debug, info, trace, warn};

use crate::inference::kv_cache::StaticKVCache;
use crate::inference::layer_cache::LayerCache;
use crate::inference::model_loader::{LlamaConfig, LlamaLayerWeights, LlamaModel, QuantWeight};
use crate::inference::telemetry::{ExecutionTelemetry, LayerTelemetry, Stopwatch};
use crate::loader::gguf::{sha256_hex, GGUFDtype, GGUFLoader, GGUFModel};
use crate::tokenizer::Tokenizer;
// matmul is called via layer.X.matmul() and model.lm_head.matmul()
use crate::backend::WgpuBackend;
use crate::loader::dequant::dequantize;
use crate::scheduler::memory_aware::{LayerAssignment, MemoryAwareScheduler, MemoryPlan};
use crate::Device;
use crate::Error;

/// A GPU weight buffer that stays in its native quantized format.
pub struct QuantGpuWeight {
    pub buffer: wgpu::Buffer,
    pub dtype: GGUFDtype,
}

pub struct LlamaLayerGpuWeights {
    q_proj: QuantGpuWeight,
    k_proj: QuantGpuWeight,
    v_proj: QuantGpuWeight,
    o_proj: QuantGpuWeight,
    gate_proj: QuantGpuWeight,
    up_proj: QuantGpuWeight,
    down_proj: QuantGpuWeight,
}

/// Pre-allocated scratch buffers to eliminate per-layer allocations in the hot path.
struct ScratchSpace {
    // Persistent state: token embeddings + residual stream [d_model]
    x: Vec<f32>,
    // Per-layer RMSNorm output (reused for both attn_norm and ffn_norm) [d_model]
    normed: Vec<f32>,
    // QKV projections
    q: Vec<f32>, // [d_model] (n_heads × head_dim)
    k: Vec<f32>, // [n_kv_heads × head_dim]
    v: Vec<f32>, // [n_kv_heads × head_dim]
    // Attention
    attn_out: Vec<f32>,  // [d_model]
    attn_proj: Vec<f32>, // [d_model]
    scores: Vec<f32>,    // [max_seq_len] — reused per head
    head_out: Vec<f32>,  // [head_dim] — single head output
    // MLP
    gate: Vec<f32>,    // [d_ff]
    up: Vec<f32>,      // [d_ff]
    mlp_out: Vec<f32>, // [d_model]

    // ── Batched prefill buffers ───────────────────────────────────────────
    // Sized for max_batch_tokens prompt tokens, allocated at load time.
    max_batch: usize,
    batch_x: Vec<f32>,          // [max_batch × d_model]
    batch_normed: Vec<f32>,     // [max_batch × d_model]
    batch_q: Vec<f32>,          // [max_batch × n_heads × head_dim]
    batch_k: Vec<f32>,          // [max_batch × n_kv_heads × head_dim]
    batch_v: Vec<f32>,          // [max_batch × n_kv_heads × head_dim]
    batch_attn: Vec<f32>,       // [max_batch × d_model]
    batch_ffn_hidden: Vec<f32>, // [max_batch × d_ff]
    batch_scores: Vec<f32>,     // [max_batch × max_batch]
    // These are per-layer and reused, but we pre-allocate them too
    batch_attn_proj: Vec<f32>, // [max_batch × d_model]
    batch_up: Vec<f32>,        // [max_batch × d_ff]
    batch_mlp_out: Vec<f32>,   // [max_batch × d_model]
}

/// Inference runtime for Llama-family models loaded from GGUF files.
///
/// Manages model weights, KV cache, tokenizer, GPU buffers, and layer
/// assignment. Supports CPU-only, GPU-accelerated (WGPU/Metal), and
/// hybrid (memory-aware) inference modes.
///
/// Heavyweight GPU resources (backend, weight buffers, RoPE GPU tables)
/// live in [`InferenceContext`] which is shared via `Arc` across all
/// runners in a pool. Each runner gets its own KV cache, scratch space,
/// and GPU KV cache buffers for true concurrent request processing.
pub struct LlamaRunner {
    /// Shared inference context (GPU backend, weight buffers, etc.)
    pub ctx: Arc<InferenceContext>,
    pub kv: StaticKVCache,
    pub tokenizer: Tokenizer,
    pub telemetry: ExecutionTelemetry,

    // Pre-computed RoPE tables: [max_seq × rope_dim/2] for sin and cos
    rope_sin: Vec<f32>,
    rope_cos: Vec<f32>,

    // GPU KV cache: per-layer buffers (one per concurrent slot)
    kv_cache_k_gpu: Vec<wgpu::Buffer>,
    kv_cache_v_gpu: Vec<wgpu::Buffer>,

    // Pre-allocated scratch buffers (no allocations in hot path)
    scratch: ScratchSpace,

    // Config shorthand
    cfg: LlamaConfig,

    // Memory-aware layer assignment (GPU vs CPU)
    pub layer_assignment: LayerAssignment,

    // Streaming / LRU cache fields (only when model doesn't fit in RAM)
    layer_cache: Option<LayerCache>,

    /// Whether CPU KV cache has been synced to GPU caches (after prefill).
    kv_synced_to_gpu: bool,

    // ── Cooperative frame budget decoding ────────────────────────────────
    /// Per-decode-step time budget in milliseconds. 0 = no budget.
    frame_budget_ms: f32,
    /// Saved state when a budgeted decode is interrupted mid-way.
    partial_decode: Option<PartialDecodeState>,
}

/// Saved state for resuming a partially-completed decode step.
struct PartialDecodeState {
    /// The token being decoded.
    token_id: u32,
    /// Sequence position of this token.
    pos: usize,
    /// Index of the last layer that completed successfully.
    /// The next layer to process = `completed_layer + 1`.
    completed_layer: usize,
    /// Saved residual stream after the last completed layer [d_model].
    x: Vec<f32>,
}

/// Shared inference context: heavyweight resources that can be shared
/// across multiple [`LlamaRunner`] instances in a pool.
///
/// Holds GPU backend, uploaded weight buffers, RoPE GPU tables, and the
/// mmap-backed GGUF handle. All fields are read-only after construction.
pub struct InferenceContext {
    pub model: Arc<LlamaModel>,
    pub wgpu_backend: Option<WgpuBackend>,
    pub gpu_weights: Vec<Option<LlamaLayerGpuWeights>>,
    pub rope_sin_gpu: Option<wgpu::Buffer>,
    pub rope_cos_gpu: Option<wgpu::Buffer>,
    pub gguf: Option<Arc<GGUFModel>>,
}

/// Options controlling model load behavior (inference deployments).
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadOptions {
    /// Run all transformer layers on CPU. Recommended for production until GPU decode
    /// is validated for your model/hardware. Set `AETHER_CPU_ONLY=0` to allow GPU.
    pub cpu_only: bool,
}

impl LlamaRunner {
    /// Load a GGUF model from disk and initialize the runner.
    pub fn from_gguf(path: &str) -> Result<Self, Error> {
        Self::from_gguf_with_options(path, LoadOptions::default())
    }

    /// Load with explicit options (e.g. CPU-only for stable serving).
    pub fn from_gguf_with_options(path: &str, options: LoadOptions) -> Result<Self, Error> {
        Self::from_gguf_internal(path, None, options)
    }

    /// Load a GGUF model with streaming / LRU caching forced.
    /// `max_hot` controls how many layers are kept in the LRU cache at once.
    /// Skips RAM detection — useful for testing the streaming code path.
    pub fn from_gguf_streaming(path: &str, max_hot: usize) -> Result<Self, Error> {
        Self::from_gguf_internal(path, Some(max_hot), LoadOptions::default())
    }

    fn from_gguf_internal(
        path: &str,
        streaming_max_hot: Option<usize>,
        options: LoadOptions,
    ) -> Result<Self, Error> {
        let (ctx, kv, tokenizer, telemetry, rope_sin, rope_cos, scratch,
             kv_cache_k_gpu, kv_cache_v_gpu, cfg, layer_assignment, layer_cache) =
            Self::load_context(path, streaming_max_hot, options)?;
        Ok(Self {
            ctx,
            kv,
            tokenizer,
            telemetry,
            rope_sin,
            rope_cos,
            kv_cache_k_gpu,
            kv_cache_v_gpu,
            scratch,
            cfg,
            layer_assignment,
            layer_cache,
            kv_synced_to_gpu: false,
            frame_budget_ms: 0.0,
            partial_decode: None,
        })
    }

    /// Create a new [`LlamaRunner`] that shares an existing [`InferenceContext`].
    ///
    /// Each runner gets its own KV cache, scratch space, and GPU KV cache
    /// buffers so multiple requests can be processed concurrently.
    pub fn new_with_context(ctx: Arc<InferenceContext>, tokenizer: &Tokenizer,
                            layer_assignment: &LayerAssignment) -> Self {
        let cfg = ctx.model.config.clone();
        let num_layers = cfg.num_layers;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let max_seq = cfg.max_seq_len.min(4096);

        let kv = StaticKVCache::new(num_layers, n_kv_heads, head_dim, max_seq);
        let scratch = ScratchSpace::new(&cfg, max_seq);

        let (rope_sin, rope_cos) = precompute_rope(max_seq, cfg.head_dim, cfg.rope_base);

        // Allocate GPU KV cache buffers if GPU is enabled
        let mut kv_cache_k_gpu = Vec::new();
        let mut kv_cache_v_gpu = Vec::new();
        if let Some(ref backend) = ctx.wgpu_backend {
            let cache_layer_elems = max_seq * n_kv_heads * head_dim;
            let zero_layer = vec![0.0f32; cache_layer_elems];
            for _ in 0..num_layers {
                let k = backend.create_buffer_with_data(
                    &zero_layer,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                );
                let v = backend.create_buffer_with_data(
                    &zero_layer,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                );
                kv_cache_k_gpu.push(k);
                kv_cache_v_gpu.push(v);
            }
        }

        // Copy GPU layer count from context's assignment if available
        let la = if cfg.num_layers > 0 && !ctx.gpu_weights.is_empty() {
            let mut l = layer_assignment.clone();
            for i in 0..cfg.num_layers.min(ctx.gpu_weights.len()) {
                if ctx.gpu_weights[i].is_some() {
                    l.layer_devices[i] = Device::Wgpu;
                }
            }
            l.gpu_layers = ctx.gpu_weights.iter().filter(|w| w.is_some()).count();
            l.cpu_layers = cfg.num_layers - l.gpu_layers;
            l
        } else {
            layer_assignment.clone()
        };

        let telemetry = ExecutionTelemetry::new(num_layers);

        Self {
            ctx,
            kv,
            tokenizer: tokenizer.clone(),
            telemetry,
            rope_sin,
            rope_cos,
            kv_cache_k_gpu,
            kv_cache_v_gpu,
            scratch,
            cfg,
            layer_assignment: la,
            layer_cache: None,
            kv_synced_to_gpu: false,
            frame_budget_ms: 0.0,
            partial_decode: None,
        }
    }

    /// Full model loading: returns the shared context and all per-runner state.
    #[allow(clippy::type_complexity)]
    fn load_context(
        path: &str,
        streaming_max_hot: Option<usize>,
        options: LoadOptions,
    ) -> Result<
        (Arc<InferenceContext>, StaticKVCache, Tokenizer, ExecutionTelemetry,
         Vec<f32>, Vec<f32>, ScratchSpace,
         Vec<wgpu::Buffer>, Vec<wgpu::Buffer>,
         LlamaConfig, LayerAssignment, Option<LayerCache>),
        Error
    > {
        let load_start = Instant::now();
        info!("Loading {}...", path);

        let gguf = GGUFLoader::load(path)?;

        // Integrity verification (optional, env AETHER_EXPECTED_SHA256=<hex>)
        if let Ok(expected) = std::env::var("AETHER_EXPECTED_SHA256") {
            let actual = sha256_hex(path)?;
            if !actual.eq_ignore_ascii_case(&expected) {
                return Err(Error::ExecutionError(format!(
                    "SHA-256 mismatch: expected {expected}, got {actual}"
                )));
            }
            info!("Model integrity verified (SHA-256 match)");
        } else {
            let hex = sha256_hex(path)?;
            debug!("Model SHA-256: {hex}");
        }

        let tokenizer = Tokenizer::from_gguf(&gguf)?;

        let config = LlamaConfig::from_gguf(&gguf)?;
        let cfg = config.clone();

        let max_seq = cfg.max_seq_len.min(4096);
        let kv = StaticKVCache::new(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, max_seq);

        let (rope_sin, rope_cos) = precompute_rope(max_seq, cfg.head_dim, cfg.rope_base);

        let load_time = load_start.elapsed();

        let (model_bytes, _quant_bpe) = estimate_model_bytes_from_gguf(&gguf);
        let kv_bytes = kv.size_bytes();

        // Decide streaming vs in-memory
        let streaming =
            streaming_max_hot.is_some() || MemoryAwareScheduler::needs_streaming(model_bytes);
        let memory_plan = if streaming {
            let max_hot = streaming_max_hot
                .unwrap_or_else(|| {
                    (MemoryAwareScheduler::detect_available_ram() as usize
                        / (model_bytes / cfg.num_layers.max(1)).max(1))
                    .max(2)
                    .min(cfg.num_layers)
                })
                .min(cfg.num_layers);
            MemoryPlan::Streaming {
                total_ram: MemoryAwareScheduler::detect_total_ram(),
                model_bytes,
                max_hot_layers: max_hot,
                per_layer_bytes: model_bytes / cfg.num_layers.max(1),
            }
        } else {
            MemoryPlan::InMemory {
                total_ram: MemoryAwareScheduler::detect_total_ram(),
                model_bytes,
            }
        };

        info!(
            "Loaded in {:.2}s | Model: {:.0} MB | KV cache: {:.1} MB",
            load_time.as_secs_f64(),
            model_bytes as f64 / 1e6,
            kv_bytes as f64 / 1e6,
        );

        let model = LlamaModel::from_gguf(&gguf, streaming)?;

        let mut telemetry = ExecutionTelemetry::new(cfg.num_layers);
        telemetry.record_load(load_time);
        telemetry.record_memory(model_bytes + kv_bytes);

        let scratch = ScratchSpace::new(&cfg, max_seq);

        let activation_scratch = scratch.total_heap_bytes();
        let mut layer_assignment = MemoryAwareScheduler::new().assign_from_bytes(
            model_bytes,
            cfg.num_layers,
            kv_bytes,
            activation_scratch,
            &config,
        );

        if options.cpu_only {
            info!("LoadOptions.cpu_only: forcing all layers to CPU (stable inference mode)");
            for dev in &mut layer_assignment.layer_devices {
                *dev = Device::Cpu;
            }
            layer_assignment.gpu_layers = 0;
            layer_assignment.cpu_layers = cfg.num_layers;
            layer_assignment.gpu_budget = 0;
        }

        let (gguf_arc, layer_cache) = if streaming {
            let gguf_arc = Arc::new(gguf);
            let max_hot = match &memory_plan {
                MemoryPlan::Streaming { max_hot_layers, .. } => *max_hot_layers,
                _ => cfg.num_layers,
            };
            let mut cache = LayerCache::with_f32_cache(cfg.clone(), max_hot, false);
            cache.preload(&gguf_arc);
            info!("Streaming mode: LRU cache holds {} hot layers", max_hot,);
            (Some(gguf_arc), Some(cache))
        } else {
            (None, None)
        };

        let mut wgpu_backend = None;
        let mut kv_cache_k_gpu = Vec::new();
        let mut kv_cache_v_gpu = Vec::new();
        let mut rope_sin_gpu = None;
        let mut rope_cos_gpu = None;
        if layer_assignment.is_gpu_enabled() {
            let cache_layer_elems = max_seq * cfg.num_kv_heads * cfg.head_dim;
            let zero_layer = vec![0.0f32; cache_layer_elems];
            let gpu_init = WgpuBackend::try_init_with(|backend| {
                let mut k_bufs = Vec::with_capacity(cfg.num_layers);
                let mut v_bufs = Vec::with_capacity(cfg.num_layers);
                for _ in 0..cfg.num_layers {
                    let k = backend.create_buffer_with_data(
                        &zero_layer,
                        wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    );
                    let v = backend.create_buffer_with_data(
                        &zero_layer,
                        wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    );
                    k_bufs.push(k);
                    v_bufs.push(v);
                }
                let rope_sin_gpu =
                    backend.create_buffer_with_data(&rope_sin, wgpu::BufferUsages::STORAGE);
                let rope_cos_gpu =
                    backend.create_buffer_with_data(&rope_cos, wgpu::BufferUsages::STORAGE);
                Ok((backend.clone(), k_bufs, v_bufs, rope_sin_gpu, rope_cos_gpu))
            });
            match gpu_init {
                Ok((backend, k_bufs, v_bufs, rsin, rcos)) => {
                    kv_cache_k_gpu = k_bufs;
                    kv_cache_v_gpu = v_bufs;
                    rope_sin_gpu = Some(rsin);
                    rope_cos_gpu = Some(rcos);
                    wgpu_backend = Some(backend);
                }
                Err(e) => {
                    warn!("GPU init failed: {e}. Falling back to CPU.");
                    for dev in &mut layer_assignment.layer_devices {
                        *dev = Device::Cpu;
                    }
                    layer_assignment.gpu_layers = 0;
                    layer_assignment.cpu_layers = cfg.num_layers;
                }
            }
        }

        info!(
            "Memory plan: {} GPU layers + {} CPU layers ({} MB model, {} MB KV)",
            layer_assignment.gpu_layers,
            layer_assignment.cpu_layers,
            model_bytes / 1_000_000,
            kv_bytes / 1_000_000,
        );

        layer_assignment.memory_plan = memory_plan;
        MemoryAwareScheduler::print_assignment(&layer_assignment);

        info!(
            "Uploading {} GPU layers to Metal...",
            layer_assignment.gpu_layers
        );
        let mut gpu_weights = Vec::with_capacity(cfg.num_layers);
        if !streaming {
            if let Some(ref backend) = wgpu_backend {
                for (i, layer) in model.layers.iter().enumerate() {
                    if layer_assignment.device_for_layer(i) == Device::Wgpu {
                        let upload =
                            |qw: &QuantWeight| -> QuantGpuWeight {
                                match qw.dtype {
                                    GGUFDtype::Q8_0 | GGUFDtype::Q4_K => {
                                        let mut padded = qw.data.to_vec();
                                        while !padded.len().is_multiple_of(4) {
                                            padded.push(0);
                                        }
                                        let buffer = backend.create_quant_buffer(
                                            &padded,
                                            wgpu::BufferUsages::STORAGE,
                                        );
                                        QuantGpuWeight {
                                            buffer,
                                            dtype: qw.dtype,
                                        }
                                    }
                                    _ => {
                                        let buf = if let Ok(buf) = backend.dequantize_on_gpu(
                                            &qw.data,
                                            qw.dtype,
                                            &qw.shape,
                                        ) {
                                            buf
                                        } else {
                                            let dequant = dequantize(&qw.data, qw.dtype, &qw.shape);
                                            backend.create_buffer_with_data(
                                                &dequant,
                                                wgpu::BufferUsages::STORAGE,
                                            )
                                        };
                                        QuantGpuWeight {
                                            buffer: buf,
                                            dtype: GGUFDtype::F32,
                                        }
                                    }
                                }
                            };

                        gpu_weights.push(Some(LlamaLayerGpuWeights {
                            q_proj: upload(&layer.q_proj),
                            k_proj: upload(&layer.k_proj),
                            v_proj: upload(&layer.v_proj),
                            o_proj: upload(&layer.o_proj),
                            gate_proj: upload(&layer.gate_proj),
                            up_proj: upload(&layer.up_proj),
                            down_proj: upload(&layer.down_proj),
                        }));
                    } else {
                        gpu_weights.push(None);
                    }
                }
            }
        } else {
            for _ in 0..cfg.num_layers {
                gpu_weights.push(None);
            }
        }

        let ctx = Arc::new(InferenceContext {
            model: Arc::new(model),
            wgpu_backend,
            gpu_weights,
            rope_sin_gpu,
            rope_cos_gpu,
            gguf: gguf_arc,
        });

        Ok((ctx, kv, tokenizer, telemetry, rope_sin, rope_cos, scratch,
            kv_cache_k_gpu, kv_cache_v_gpu, cfg, layer_assignment, layer_cache))
    }

    /// Generate text from a prompt using autoregressive decoding.
    ///
    /// Runs prefill on the full prompt, then loops decode + sample until
    /// EOS, max_tokens, or the KV cache fills up. Greedy sampling by
    /// default; use temperature/top_p/repetition_penalty for controlled
    /// generation.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        repetition_penalty: f32,
    ) -> Result<String, Error> {
        let mut tokens = Vec::new();
        self.generate_callback(
            prompt,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
            |_| {},
            |t| tokens.push(t.to_string()),
        )?;
        Ok(tokens.concat())
    }

    /// Generate text, calling `on_token` for each new token as it is produced.
    /// Useful for streaming UIs and pyo3 bindings.
    pub fn generate_callback(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        repetition_penalty: f32,
        mut on_first_token: impl FnMut(&str),
        mut on_token: impl FnMut(&str),
    ) -> Result<(), Error> {
        self.kv.reset();
        self.kv_synced_to_gpu = false;

        // Tokenize
        let mut token_ids = self.tokenizer.encode(prompt, true);
        let prompt_len = token_ids.len();
        info!(
            "Prompt: {} tokens: {:?}",
            prompt_len,
            &token_ids[..token_ids.len().min(20)]
        );

        // Prefill (all prompt tokens in one forward pass)
        let prefill_start = Instant::now();
        let mut last_logits = self.prefill(&token_ids)?;
        let prefill_time = prefill_start.elapsed();
        self.telemetry.record_prefill(prefill_time, prompt_len);

        // Sample first token
        debug!(
            "PREFILL logits: top-5: {:?}",
            &top_k_indices(&last_logits, 5)
        );
        let prev_set: HashSet<u32> = token_ids.iter().copied().collect();
        let mut next_token = sample(
            &last_logits,
            temperature,
            top_p,
            &prev_set,
            repetition_penalty,
        );
        token_ids.push(next_token);

        debug!(
            "First token: {} decoded={:?}",
            next_token,
            self.tokenizer.decode_one(next_token)
        );
        let first = self.tokenizer.decode_one(next_token);
        on_first_token(&first);
        on_token(&first);

        // Decode loop
        for step in 0..max_new_tokens.saturating_sub(1) {
            if next_token == self.tokenizer.eos_id {
                break;
            }
            if self.kv.seq_len >= self.kv.max_seq {
                warn!("Context window full at {} tokens", self.kv.seq_len);
                break;
            }

            let pos = prompt_len + step;

            let mut step_telemetry = vec![LayerTelemetry::default(); self.cfg.num_layers];
            last_logits = self.decode_step(next_token, pos, &mut step_telemetry)?;

            debug!(
                "Decode step {}: token={} decoded={:?} top-5: {:?}",
                step,
                next_token,
                self.tokenizer.decode_one(next_token),
                &top_k_indices(&last_logits, 5)
            );
            let prev_set: HashSet<u32> = token_ids.iter().copied().collect();
            next_token = sample(
                &last_logits,
                temperature,
                top_p,
                &prev_set,
                repetition_penalty,
            );
            token_ids.push(next_token);

            let tok_str = self.tokenizer.decode_one(next_token);
            on_token(&tok_str);
        }

        Ok(())
    }

    /// Run all prompt tokens through the model to warm up the KV cache.
    /// Returns logits for the last token position.
    pub fn prefill(&mut self, token_ids: &[u32]) -> Result<Vec<f32>, Error> {
        let n = token_ids.len();
        if n == 0 {
            return Err(Error::ExecutionError("Empty prompt".into()));
        }
        let mut layer_tel = vec![LayerTelemetry::default(); self.cfg.num_layers];
        let logits = self.forward_batch(token_ids, &mut layer_tel)?;
        self.kv.seq_len = n;
        self.sync_kv_cache_to_gpu()?;
        Ok(logits)
    }

    /// Copy CPU KV cache → GPU KV cache for all GPU layers.
    /// Required because `forward_batch` (CPU prefill) writes to CPU KV cache
    /// only, while GPU decode reads from separate GPU KV cache buffers.
    fn sync_kv_cache_to_gpu(&self) -> Result<(), Error> {
        let backend = match &self.ctx.wgpu_backend {
            Some(b) => b,
            None => return Ok(()),
        };
        let kv_heads = self.cfg.num_kv_heads;
        let head_dim = self.cfg.head_dim;
        let head_stride = self.kv.head_stride();
        let layer_stride = self.kv.layer_stride();
        let storage = self.kv.storage();
        let seq_len = self.kv.seq_len;
        if seq_len == 0 {
            return Ok(());
        }
        let pos_stride = kv_heads * head_dim;
        for layer_idx in 0..self.cfg.num_layers {
            if self.layer_assignment.device_for_layer(layer_idx) != Device::Wgpu {
                continue;
            }
            let kv_k = &self.kv_cache_k_gpu[layer_idx];
            let kv_v = &self.kv_cache_v_gpu[layer_idx];
            let layer_base = layer_idx * layer_stride;
            let v_base = layer_base + kv_heads * head_stride;
            let mut gpu_k = vec![0.0f32; seq_len * pos_stride];
            let mut gpu_v = vec![0.0f32; seq_len * pos_stride];
            for p in 0..seq_len {
                for h in 0..kv_heads {
                    let cpu_k = &storage[layer_base + h * head_stride + p * head_dim..][..head_dim];
                    let cpu_v = &storage[v_base + h * head_stride + p * head_dim..][..head_dim];
                    let gpu_offset = p * pos_stride + h * head_dim;
                    gpu_k[gpu_offset..gpu_offset + head_dim].copy_from_slice(cpu_k);
                    gpu_v[gpu_offset..gpu_offset + head_dim].copy_from_slice(cpu_v);
                }
            }
            backend
                .queue()
                .write_buffer(kv_k, 0, bytemuck::cast_slice(&gpu_k));
            backend
                .queue()
                .write_buffer(kv_v, 0, bytemuck::cast_slice(&gpu_v));
        }
        Ok(())
    }

    /// Number of transformer layers in the loaded model.
    pub fn num_layers(&self) -> usize {
        self.cfg.num_layers
    }

    /// Set the per-decode-step frame budget in milliseconds.
    /// 0 means no budget (run to completion).
    pub fn set_frame_budget(&mut self, max_ms: f32) {
        self.frame_budget_ms = max_ms;
    }

    /// Get the current frame budget in milliseconds.
    pub fn frame_budget(&self) -> f32 {
        self.frame_budget_ms
    }

    /// Clear any in-progress partial decode state.
    /// Called automatically when a different token/pos is requested.
    pub fn clear_partial_decode(&mut self) {
        self.partial_decode = None;
    }

    /// Decode one new token with optional frame budget.
    /// Non-budgeted: runs to completion, clears any partial state.
    pub fn decode_step(
        &mut self,
        token_id: u32,
        pos: usize,
        layer_tel: &mut [LayerTelemetry],
    ) -> Result<Vec<f32>, Error> {
        // Non-budgeted decode: clear any stale partial state
        self.partial_decode = None;
        self.forward_one(token_id, pos, layer_tel)
    }

    /// Decode one new token respecting the frame budget.
    ///
    /// * Returns `Ok(logits)` on successful completion.
    /// * Returns `Err(BudgetExceeded(n))` if the budget was exceeded after
    ///   completing `n` layers. The partial state is saved internally; call
    ///   `decode_step_budgeted` again with the same `token_id` and `pos`
    ///   to resume.
    /// * Returns `Err(...)` on other errors.
    pub fn decode_step_budgeted(
        &mut self,
        token_id: u32,
        pos: usize,
        layer_tel: &mut [LayerTelemetry],
    ) -> Result<Vec<f32>, Error> {
        if self.frame_budget_ms <= 0.0 {
            // No budget → run to completion like decode_step
            self.partial_decode = None;
            return self.forward_one(token_id, pos, layer_tel);
        }

        // If there's a stale partial state for a different token/pos, clear it
        if let Some(ref p) = self.partial_decode {
            if p.token_id != token_id || p.pos != pos {
                self.partial_decode = None;
            }
        }

        self.forward_one_budgeted(token_id, pos, layer_tel)
    }

    /// Public hook for logit dumping / debugging.
    pub fn forward_one_hook(
        &mut self,
        token_id: u32,
        pos: usize,
        layer_tel: &mut [LayerTelemetry],
    ) -> Result<Vec<f32>, Error> {
        self.forward_one(token_id, pos, layer_tel)
    }

    /// Batched forward pass for all prompt tokens at once.
    /// Processes `token_ids` in parallel with causal attention mask.
    /// Returns logits [vocab_size] for the LAST token only.
    /// KV cache is filled with all token positions.
    fn forward_batch(
        &mut self,
        token_ids: &[u32],
        layer_tel: &mut [LayerTelemetry],
    ) -> Result<Vec<f32>, Error> {
        let n = token_ids.len();
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let n_heads = cfg.num_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let d_ff = cfg.d_ff;
        let s = &mut self.scratch;

        // Ensure batch buffers are large enough (fall back to alloc if not)
        if n > s.max_batch {
            return self.forward_batch_fallback(token_ids, layer_tel);
        }

        // Use pre-allocated scratch buffers (zero allocations in hot path)
        let batch_x = &mut s.batch_x[..n * d];
        let batch_normed = &mut s.batch_normed[..n * d];
        let batch_q = &mut s.batch_q[..n * n_heads * head_dim];
        let batch_k = &mut s.batch_k[..n * n_kv_heads * head_dim];
        let batch_v = &mut s.batch_v[..n * n_kv_heads * head_dim];
        let batch_attn = &mut s.batch_attn[..n * d];
        let batch_ffn_hidden = &mut s.batch_ffn_hidden[..n * d_ff];
        let scores = &mut s.batch_scores[..n * n];

        // 1. Embed all tokens
        let model = &self.ctx.model;
        let rope_sin = &self.rope_sin;
        let rope_cos = &self.rope_cos;
        let kv = &mut self.kv;
        let layer_cache = &mut self.layer_cache;
        let gguf = &self.ctx.gguf;
        for (i, &tok) in token_ids.iter().enumerate() {
            let emb = &model.token_embeddings[(tok as usize) * d..][..d];
            batch_x[i * d..(i + 1) * d].copy_from_slice(emb);
        }

        // 2. Transformer layers — CPU-only path
        for layer_idx in 0..self.cfg.num_layers {
            let layer = resolve_layer(layer_cache, gguf, model, layer_idx)?;
            let lt = &mut layer_tel[layer_idx];

            // 2a. Batched RMSNorm
            for i in 0..n {
                let x = &batch_x[i * d..(i + 1) * d];
                let out = &mut batch_normed[i * d..(i + 1) * d];
                rmsnorm_inplace(x, &layer.attn_norm, cfg.rms_norm_eps, out);
            }

            let attn_sw = Stopwatch::start();

            // 2b. Batched QKV projection (M=n instead of M=1)
            layer.q_proj.matmul(batch_normed, n, batch_q);
            layer.k_proj.matmul(batch_normed, n, batch_k);
            layer.v_proj.matmul(batch_normed, n, batch_v);

            // 2c. Batched RoPE (each token at its position)
            for i in 0..n {
                let q_i = &mut batch_q[i * n_heads * head_dim..(i + 1) * n_heads * head_dim];
                let k_i = &mut batch_k[i * n_kv_heads * head_dim..(i + 1) * n_kv_heads * head_dim];
                apply_rope_inplace(q_i, n_heads, head_dim, i, rope_sin, rope_cos);
                apply_rope_inplace(k_i, n_kv_heads, head_dim, i, rope_sin, rope_cos);
            }

            // 2d. Write K,V to cache (all positions)
            let kv_stride = kv.layer_stride();
            let head_stride = kv.head_stride();
            let kv_storage = kv.storage_mut();
            let layer_kv_base = layer_idx * kv_stride;
            for i in 0..n {
                for h in 0..n_kv_heads {
                    let k_src = &batch_k[(i * n_kv_heads + h) * head_dim..][..head_dim];
                    let k_dst = layer_kv_base + h * head_stride + i * head_dim;
                    kv_storage[k_dst..k_dst + head_dim].copy_from_slice(k_src);

                    let v_src = &batch_v[(i * n_kv_heads + h) * head_dim..][..head_dim];
                    let v_dst =
                        layer_kv_base + n_kv_heads * head_stride + h * head_stride + i * head_dim;
                    kv_storage[v_dst..v_dst + head_dim].copy_from_slice(v_src);
                }
            }

            // 2e. Batched causal attention
            attention_batch(
                batch_q,
                batch_k,
                batch_v,
                n,
                n_heads,
                n_kv_heads,
                head_dim,
                batch_attn,
                scores,
                cfg.arch.sliding_window,
            );

            // 2f. Output projection
            let batch_attn_proj = &mut s.batch_attn_proj[..n * d];
            layer.o_proj.matmul(batch_attn, n, batch_attn_proj);

            lt.attn_us += attn_sw.elapsed_us();

            // 2g. Residual
            for i in 0..n {
                let x = &mut batch_x[i * d..(i + 1) * d];
                let attn = &batch_attn_proj[i * d..(i + 1) * d];
                for j in 0..d {
                    x[j] += attn[j];
                }
            }

            let mlp_sw = Stopwatch::start();

            // 2h. Post-attention RMSNorm
            for i in 0..n {
                let x = &batch_x[i * d..(i + 1) * d];
                let out = &mut batch_normed[i * d..(i + 1) * d];
                rmsnorm_inplace(x, &layer.ffn_norm, cfg.rms_norm_eps, out);
            }

            // 2i. SiLU MLP (batched)
            layer.gate_proj.matmul(batch_normed, n, batch_ffn_hidden);
            let batch_up = &mut s.batch_up[..n * d_ff];
            layer.up_proj.matmul(batch_normed, n, batch_up);

            // SiLU: gate = gate * sigmoid(gate) * up
            for i in 0..n * d_ff {
                batch_ffn_hidden[i] =
                    batch_ffn_hidden[i] * sigmoid(batch_ffn_hidden[i]) * batch_up[i];
            }

            let batch_mlp_out = &mut s.batch_mlp_out[..n * d];
            layer.down_proj.matmul(batch_ffn_hidden, n, batch_mlp_out);

            lt.mlp_us += mlp_sw.elapsed_us();

            // 2j. Residual
            for i in 0..n {
                let x = &mut batch_x[i * d..(i + 1) * d];
                let mlp = &batch_mlp_out[i * d..(i + 1) * d];
                for j in 0..d {
                    x[j] += mlp[j];
                }
            }
        }

        // 3. Final RMSNorm on last token
        let last_x = &batch_x[(n - 1) * d..n * d];
        let final_normed = &mut batch_normed[..d];
        rmsnorm_inplace(last_x, &model.norm, cfg.rms_norm_eps, final_normed);

        // 4. LM head: logits = last_x_norm × lm_head^T
        let mut logits = vec![0.0f32; cfg.vocab_size];
        model.lm_head.matmul(final_normed, 1, &mut logits);

        Ok(logits)
    }

    /// Fallback for prompts larger than pre-allocated batch buffers.
    fn forward_batch_fallback(
        &mut self,
        token_ids: &[u32],
        layer_tel: &mut [LayerTelemetry],
    ) -> Result<Vec<f32>, Error> {
        let n = token_ids.len();
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let n_heads = cfg.num_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let d_ff = cfg.d_ff;

        let mut batch_x = vec![0.0f32; n * d];
        let mut batch_normed = vec![0.0f32; n * d];
        let mut batch_q = vec![0.0f32; n * n_heads * head_dim];
        let mut batch_k = vec![0.0f32; n * n_kv_heads * head_dim];
        let mut batch_v = vec![0.0f32; n * n_kv_heads * head_dim];
        let mut batch_attn = vec![0.0f32; n * d];
        let mut batch_ffn_hidden = vec![0.0f32; n * d_ff];
        let mut scores = vec![0.0f32; n * n];

        let model = &self.ctx.model;
        let rope_sin = &self.rope_sin;
        let rope_cos = &self.rope_cos;
        let kv = &mut self.kv;
        let layer_cache = &mut self.layer_cache;
        let gguf = &self.ctx.gguf;

        for (i, &tok) in token_ids.iter().enumerate() {
            let emb = &model.token_embeddings[(tok as usize) * d..][..d];
            batch_x[i * d..(i + 1) * d].copy_from_slice(emb);
        }

        // 2. Transformer layers — CPU-only path
        for layer_idx in 0..self.cfg.num_layers {
            let layer = resolve_layer(layer_cache, gguf, model, layer_idx)?;
            let lt = &mut layer_tel[layer_idx];

            // 2a. Batched RMSNorm
            for i in 0..n {
                let x = &batch_x[i * d..(i + 1) * d];
                let out = &mut batch_normed[i * d..(i + 1) * d];
                rmsnorm_inplace(x, &layer.attn_norm, cfg.rms_norm_eps, out);
            }

            let attn_sw = Stopwatch::start();

            // 2b. Batched QKV projection
            layer.q_proj.matmul(&batch_normed, n, &mut batch_q);
            layer.k_proj.matmul(&batch_normed, n, &mut batch_k);
            layer.v_proj.matmul(&batch_normed, n, &mut batch_v);

            // 2c. Batched RoPE (each token at its position)
            for i in 0..n {
                let q_i = &mut batch_q[i * n_heads * head_dim..(i + 1) * n_heads * head_dim];
                let k_i = &mut batch_k[i * n_kv_heads * head_dim..(i + 1) * n_kv_heads * head_dim];
                apply_rope_inplace(q_i, n_heads, head_dim, i, rope_sin, rope_cos);
                apply_rope_inplace(k_i, n_kv_heads, head_dim, i, rope_sin, rope_cos);
            }

            // 2d. Write K,V to cache (all positions)
            let kv_stride = kv.layer_stride();
            let head_stride = kv.head_stride();
            let kv_storage = kv.storage_mut();
            let layer_kv_base = layer_idx * kv_stride;
            for i in 0..n {
                for h in 0..n_kv_heads {
                    let k_src = &batch_k[(i * n_kv_heads + h) * head_dim..][..head_dim];
                    let k_dst = layer_kv_base + h * head_stride + i * head_dim;
                    kv_storage[k_dst..k_dst + head_dim].copy_from_slice(k_src);

                    let v_src = &batch_v[(i * n_kv_heads + h) * head_dim..][..head_dim];
                    let v_dst =
                        layer_kv_base + n_kv_heads * head_stride + h * head_stride + i * head_dim;
                    kv_storage[v_dst..v_dst + head_dim].copy_from_slice(v_src);
                }
            }

            // 2e. Batched causal attention
            attention_batch(
                &batch_q,
                &batch_k,
                &batch_v,
                n,
                n_heads,
                n_kv_heads,
                head_dim,
                &mut batch_attn,
                &mut scores,
                cfg.arch.sliding_window,
            );

            let mut batch_attn_proj = vec![0.0f32; n * d];
            layer.o_proj.matmul(&batch_attn, n, &mut batch_attn_proj);
            lt.attn_us += attn_sw.elapsed_us();

            // 2f. Residual
            for i in 0..n {
                let x = &mut batch_x[i * d..(i + 1) * d];
                let attn = &batch_attn_proj[i * d..(i + 1) * d];
                for j in 0..d {
                    x[j] += attn[j];
                }
            }

            let mlp_sw = Stopwatch::start();

            // 2g. Post-attention RMSNorm
            for i in 0..n {
                let x = &batch_x[i * d..(i + 1) * d];
                let out = &mut batch_normed[i * d..(i + 1) * d];
                rmsnorm_inplace(x, &layer.ffn_norm, cfg.rms_norm_eps, out);
            }

            // 2h. SiLU MLP (batched)
            layer
                .gate_proj
                .matmul(&batch_normed, n, &mut batch_ffn_hidden);
            let mut batch_up = vec![0.0f32; n * d_ff];
            layer.up_proj.matmul(&batch_normed, n, &mut batch_up);

            // SiLU
            for i in 0..n * d_ff {
                batch_ffn_hidden[i] =
                    batch_ffn_hidden[i] * sigmoid(batch_ffn_hidden[i]) * batch_up[i];
            }

            let mut batch_mlp_out = vec![0.0f32; n * d];
            layer
                .down_proj
                .matmul(&batch_ffn_hidden, n, &mut batch_mlp_out);
            lt.mlp_us += mlp_sw.elapsed_us();

            // 2i. Residual
            for i in 0..n {
                let x = &mut batch_x[i * d..(i + 1) * d];
                let mlp = &batch_mlp_out[i * d..(i + 1) * d];
                for j in 0..d {
                    x[j] += mlp[j];
                }
            }
        }

        let last_x = &batch_x[(n - 1) * d..n * d];
        let mut final_normed = vec![0.0f32; d];
        rmsnorm_inplace(last_x, &model.norm, cfg.rms_norm_eps, &mut final_normed);

        let mut logits = vec![0.0f32; cfg.vocab_size];
        model.lm_head.matmul(&final_normed, 1, &mut logits);

        Ok(logits)
    }

    /// Full forward pass for a single token at position `pos`.
    /// Returns logits [vocab_size].
    /// Non-budgeted: clears any partial state, runs to completion.
    fn forward_one(
        &mut self,
        token_id: u32,
        pos: usize,
        layer_tel: &mut [LayerTelemetry],
    ) -> Result<Vec<f32>, Error> {
        self.partial_decode = None;
        self.forward_one_internal(token_id, pos, layer_tel, false)
    }

    /// Budgeted forward pass that supports partial decode and resume.
    /// Returns `Err(BudgetExceeded(n))` if the frame budget was exceeded.
    fn forward_one_budgeted(
        &mut self,
        token_id: u32,
        pos: usize,
        layer_tel: &mut [LayerTelemetry],
    ) -> Result<Vec<f32>, Error> {
        self.forward_one_internal(token_id, pos, layer_tel, true)
    }

    /// Internal forward pass — shared by budgeted and non-budgeted paths.
    fn forward_one_internal(
        &mut self,
        token_id: u32,
        pos: usize,
        layer_tel: &mut [LayerTelemetry],
        budgeted: bool,
    ) -> Result<Vec<f32>, Error> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let n_heads = cfg.num_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        // Sync CPU KV cache to GPU caches on first decode after prefill.
        if !self.kv_synced_to_gpu && self.ctx.wgpu_backend.is_some() {
            self.sync_kv_cache_to_gpu()?;
            self.kv_synced_to_gpu = true;
        }

        let s = &mut self.scratch;
        let model = &self.ctx.model;
        let rope_sin = &self.rope_sin;
        let rope_cos = &self.rope_cos;
        let kv = &mut self.kv;
        let layer_cache = &mut self.layer_cache;
        let gguf = &self.ctx.gguf;

        // Determine start layer and restore partial state if resuming
        let start_layer = if let Some(ref partial) = self.partial_decode {
            if partial.token_id == token_id && partial.pos == pos {
                let completed = partial.completed_layer;
                s.x.copy_from_slice(&partial.x);
                completed + 1
            } else {
                self.partial_decode = None;
                let emb_start = (token_id as usize) * d;
                s.x.copy_from_slice(&model.token_embeddings[emb_start..emb_start + d]);
                0
            }
        } else {
            let emb_start = (token_id as usize) * d;
            s.x.copy_from_slice(&model.token_embeddings[emb_start..emb_start + d]);
            0
        };

        // Pre-compute budget deadline
        let deadline = if budgeted && self.frame_budget_ms > 0.0 {
            Some(
                std::time::Instant::now()
                    + std::time::Duration::from_secs_f32(self.frame_budget_ms / 1000.0),
            )
        } else {
            None
        };

        /// Dispatch fused quant matmul or fallback to f32 pipeline based on
        /// the weight's dtype.
        fn quant_matmul(
            backend: &WgpuBackend,
            encoder: &mut wgpu::CommandEncoder,
            a_buf: &wgpu::Buffer,
            weight: &QuantGpuWeight,
            m: u32,
            n: u32,
            k: u32,
        ) -> Result<wgpu::Buffer, crate::Error> {
            match weight.dtype {
                GGUFDtype::Q8_0 => {
                    backend.execute_matmul_q8_0_buffers_with_encoder(
                        encoder, a_buf, &weight.buffer, m, n, k,
                    )
                }
                GGUFDtype::Q4_K => {
                    backend.execute_matmul_q4_k_buffers_with_encoder(
                        encoder, a_buf, &weight.buffer, m, n, k,
                    )
                }
                _ => {
                    backend.execute_matmul_buffers_with_encoder(
                        encoder, a_buf, &weight.buffer, m, n, k,
                    )
                }
            }
        }

        // 2. Transformer layers
        for layer_idx in start_layer..self.cfg.num_layers {
            // Check budget before processing this layer
            if let Some(deadline) = deadline {
                if std::time::Instant::now() >= deadline {
                    let prev = layer_idx.saturating_sub(1);
                    self.partial_decode = Some(PartialDecodeState {
                        token_id,
                        pos,
                        completed_layer: prev,
                        x: s.x.clone(),
                    });
                    return Err(Error::BudgetExceeded(prev));
                }
            }

            let lt = &mut layer_tel[layer_idx];

            if self.layer_assignment.device_for_layer(layer_idx) == Device::Wgpu {
                let backend = self
                    .ctx
                    .wgpu_backend
                    .as_ref()
                    .ok_or_else(|| Error::ExecutionError("GPU buffer not initialized".into()))?;
                let gw = self.ctx.gpu_weights[layer_idx]
                    .as_ref()
                    .ok_or_else(|| Error::ExecutionError("GPU buffer not initialized".into()))?;
                let rsin = self
                    .ctx
                    .rope_sin_gpu
                    .as_ref()
                    .ok_or_else(|| Error::ExecutionError("GPU buffer not initialized".into()))?;
                let rcos = self
                    .ctx
                    .rope_cos_gpu
                    .as_ref()
                    .ok_or_else(|| Error::ExecutionError("GPU buffer not initialized".into()))?;
                let kv_k = &self.kv_cache_k_gpu[layer_idx];
                let kv_v = &self.kv_cache_v_gpu[layer_idx];
                let head_dim_u = cfg.head_dim as u32;
                let n_kv_u = cfg.num_kv_heads as u32;
                let n_h_u = cfg.num_heads as u32;
                let d_u = cfg.d_model as u32;
                let d_ff_u = cfg.d_ff as u32;
                let kv_d_u = n_kv_u * head_dim_u;
                let max_seq_u = cfg.max_seq_len as u32;

                // CPU RMSNorm (matches CPU decode path exactly for GPU precision parity)
                let attn_norm = &model.layers[layer_idx].attn_norm;
                rmsnorm_inplace(&s.x, attn_norm, cfg.rms_norm_eps, &mut s.normed);

                if layer_idx == 0 {
                    let sx_sum: f32 = s.x.iter().map(|x| x.abs()).sum();
                    let sn_sum: f32 = s.normed.iter().map(|x| x.abs()).sum();
                    trace!(
                        "[diag L0 CPU] s.x abs_sum={:.4} s.normed abs_sum={:.4}",
                        sx_sum,
                        sn_sum
                    );
                    trace!(
                        "[diag L0 CPU] s.x first3={:.6?} s.normed first3={:.6?}",
                        &s.x[..3.min(s.x.len())],
                        &s.normed[..3.min(s.normed.len())]
                    );

                    // Sanity check: create a tiny [3] f32 buffer and read it back
                    let sanity = backend.create_buffer_with_data(
                        &[1.0f32, 2.0, 3.0],
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    );
                    let sanity_read: Vec<f32> = backend.read_buffer(&sanity, 12)?;
                    trace!(
                        "[diag L0 sanity] create_buffer_with_data([1,2,3]) readback={:.6?}",
                        &sanity_read
                    );
                }

                // Upload:
                //   normed_buf    → QKV matmuls (CPU-normalized)
                //   residual_buf  → skip connections (original s.x, unnormalized)
                let normed_buf = backend.create_buffer_with_data(
                    &s.normed,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                );
                let residual_buf = backend.create_buffer_with_data(
                    &s.x,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                );

                let mut enc = backend.create_encoder("qkv_matmul");
                let q = quant_matmul(
                    backend,
                    &mut enc,
                    &normed_buf,
                    &gw.q_proj,
                    1,
                    d_u,
                    d_u,
                )?;
                let k = quant_matmul(
                    backend,
                    &mut enc,
                    &normed_buf,
                    &gw.k_proj,
                    1,
                    kv_d_u,
                    d_u,
                )?;
                let v = quant_matmul(
                    backend,
                    &mut enc,
                    &normed_buf,
                    &gw.v_proj,
                    1,
                    kv_d_u,
                    d_u,
                )?;
                backend.submit_encoder(enc);

                backend.execute_rope_buffers(&q, 1, n_h_u, head_dim_u, pos as u32, rsin, rcos)?;
                backend.execute_rope_buffers(&k, 1, n_kv_u, head_dim_u, pos as u32, rsin, rcos)?;

                let seq_len_u = (kv.seq_len + 1) as u32;
                let kv_cache_bytes = (kv_d_u * 4) as u64;
                let mut enc2 = backend.create_encoder("kv_append");
                backend.copy_buffers_with_encoder(
                    &mut enc2,
                    &k,
                    kv_k,
                    0,
                    (pos as u64) * kv_cache_bytes,
                    kv_cache_bytes,
                );
                backend.copy_buffers_with_encoder(
                    &mut enc2,
                    &v,
                    kv_v,
                    0,
                    (pos as u64) * kv_cache_bytes,
                    kv_cache_bytes,
                );
                backend.submit_encoder(enc2);

                let attn_out = backend.create_device_buffer(
                    (d_u as u64) * 4,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    "attn_out",
                );
                let mut enc_attn = backend.create_encoder("attn");
                backend.execute_attention_buffers_with_encoder(
                    &mut enc_attn,
                    &q,
                    kv_k,
                    kv_v,
                    &attn_out,
                    1,
                    n_h_u,
                    n_kv_u,
                    head_dim_u,
                    seq_len_u,
                    max_seq_u,
                    pos as u32,
                )?;
                backend.submit_encoder(enc_attn);

                if layer_idx == 0 {
                    // Diagnostic: read back input and weights to find where zeros originate
                    let normed_cpu: Vec<f32> = backend.read_buffer(&normed_buf, cfg.d_model * 4)?;
                    let normed_sum: f32 = normed_cpu.iter().map(|x| x.abs()).sum();
                    let normed_first3 = if normed_cpu.len() >= 3 {
                        [normed_cpu[0], normed_cpu[1], normed_cpu[2]]
                    } else {
                        [0.0; 3]
                    };
                    trace!(
                        "[diag L0] normed_buf abs_sum={:.4} first3={:.6?}",
                        normed_sum,
                        normed_first3
                    );
                    trace!(
                        "[diag L0] normed_buf min={:.6} max={:.6}",
                        normed_cpu.iter().cloned().fold(f32::INFINITY, f32::min),
                        normed_cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                    );
                    trace!(
                        "[diag L0] d_u={} n_h_u={} head_dim_u={} kv_d_u={}",
                        d_u,
                        n_h_u,
                        head_dim_u,
                        kv_d_u
                    );

                    let q_cpu: Vec<f32> =
                        backend.read_buffer(&q, (n_h_u * head_dim_u) as usize * 4)?;
                    let k_cpu: Vec<f32> =
                        backend.read_buffer(&k, (n_kv_u * head_dim_u) as usize * 4)?;
                    let v_cpu: Vec<f32> =
                        backend.read_buffer(&v, (n_kv_u * head_dim_u) as usize * 4)?;
                    let qnan: usize = q_cpu.iter().filter(|x| x.is_nan()).count();
                    let knan: usize = k_cpu.iter().filter(|x| x.is_nan()).count();
                    let vnan: usize = v_cpu.iter().filter(|x| x.is_nan()).count();
                    trace!(
                        "[diag L0] Q nan={} sum={:.4}",
                        qnan,
                        q_cpu.iter().map(|x| x.abs()).sum::<f32>()
                    );
                    trace!(
                        "[diag L0] K nan={} sum={:.4}",
                        knan,
                        k_cpu.iter().map(|x| x.abs()).sum::<f32>()
                    );
                    trace!(
                        "[diag L0] V nan={} sum={:.4}",
                        vnan,
                        v_cpu.iter().map(|x| x.abs()).sum::<f32>()
                    );

                    let ar: Vec<f32> = backend.read_buffer(&attn_out, cfg.d_model * 4)?;
                    let asum: f32 = ar.iter().map(|v| v.abs()).sum();
                    let anan: usize = ar.iter().filter(|v| v.is_nan()).count();
                    trace!(
                        "[diag L0] attn_out abs_sum={:.4} nan={} first3={:.6?}",
                        asum,
                        anan,
                        &ar[..3]
                    );
                }

                let mut enc3 = backend.create_encoder("output_proj");
                let attn_proj = quant_matmul(
                    backend, &mut enc3, &attn_out, &gw.o_proj, 1, d_u, d_u,
                )?;
                let x_new = backend.execute_add_buffers_with_encoder(
                    &mut enc3,
                    &residual_buf,
                    &attn_proj,
                    cfg.d_model,
                )?;
                backend.submit_encoder(enc3);

                // Read x_new back to CPU for precise RMSNorm
                let x_new_cpu: Vec<f32> = backend.read_buffer(&x_new, cfg.d_model * 4)?;
                let layer_ffn_norm = &model.layers[layer_idx].ffn_norm;
                rmsnorm_inplace(&x_new_cpu, layer_ffn_norm, cfg.rms_norm_eps, &mut s.normed);

                // Upload CPU-normalized FFN input
                let normed_ff_buf = backend.create_buffer_with_data(
                    &s.normed,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                );

                let mut enc4 = backend.create_encoder("ffn_matmul");
                let gate = quant_matmul(
                    backend,
                    &mut enc4,
                    &normed_ff_buf,
                    &gw.gate_proj,
                    1,
                    d_ff_u,
                    d_u,
                )?;
                let up = quant_matmul(
                    backend,
                    &mut enc4,
                    &normed_ff_buf,
                    &gw.up_proj,
                    1,
                    d_ff_u,
                    d_u,
                )?;
                backend.submit_encoder(enc4);

                let silu_out = backend.execute_silu_mul_buffers(&gate, &up, 1, d_ff_u)?;

                let mut enc5 = backend.create_encoder("ffn_down");
                let mlp_out = quant_matmul(
                    backend,
                    &mut enc5,
                    &silu_out,
                    &gw.down_proj,
                    1,
                    d_u,
                    d_ff_u,
                )?;
                let x_final = backend.execute_add_buffers_with_encoder(
                    &mut enc5,
                    &x_new,
                    &mlp_out,
                    cfg.d_model,
                )?;
                backend.submit_encoder(enc5);

                let final_cpu = backend.read_buffer(&x_final, cfg.d_model * 4)?;
                if layer_idx == 0 {
                    let xnew_cpu: Vec<f32> = backend.read_buffer(&x_new, cfg.d_model * 4)?;
                    let silu_cpu: Vec<f32> = backend.read_buffer(&silu_out, d_ff_u as usize * 4)?;
                    trace!(
                        "[diag L0] x_new (residual+attn) nan={} sum={:.4}",
                        xnew_cpu.iter().filter(|x| x.is_nan()).count(),
                        xnew_cpu.iter().map(|x| x.abs()).sum::<f32>()
                    );
                    trace!(
                        "[diag L0] silu_out nan={} sum={:.4}",
                        silu_cpu.iter().filter(|x| x.is_nan()).count(),
                        silu_cpu.iter().map(|x| x.abs()).sum::<f32>()
                    );
                    let fsum: f32 = final_cpu.iter().map(|v| v.abs()).sum();
                    let fnan: usize = final_cpu.iter().filter(|v| v.is_nan()).count();
                    trace!(
                        "[diag L0] x_final abs_sum={:.4} nan={} first3={:.6?}",
                        fsum,
                        fnan,
                        &final_cpu[..3]
                    );
                }
                s.x.copy_from_slice(&final_cpu);
            } else {
                let layer = resolve_layer(layer_cache, gguf, model, layer_idx)?;

                rmsnorm_inplace(&s.x, &layer.attn_norm, cfg.rms_norm_eps, &mut s.normed);

                let attn_sw = Stopwatch::start();

                layer.q_proj.matmul(&s.normed, 1, &mut s.q);
                layer.k_proj.matmul(&s.normed, 1, &mut s.k);
                layer.v_proj.matmul(&s.normed, 1, &mut s.v);

                apply_rope_inplace(&mut s.q, n_heads, head_dim, pos, rope_sin, rope_cos);
                apply_rope_inplace(&mut s.k, n_kv_heads, head_dim, pos, rope_sin, rope_cos);

                kv.append(layer_idx, &s.k, &s.v);

                let seq = kv.seq_len + 1;
                let kv_stride = kv.layer_stride();
                let head_stride = kv.head_stride();
                let kv_base = layer_idx * kv_stride;
                let kv_storage = kv.storage();

                let kv_groups = n_heads / n_kv_heads;
                s.attn_out.fill(0.0);

                for h in 0..n_heads {
                    let kv_h = h / kv_groups;
                    let q_h = &s.q[h * head_dim..(h + 1) * head_dim];
                    let k_src = &kv_storage
                        [kv_base + kv_h * head_stride..kv_base + (kv_h + 1) * head_stride];
                    let v_src = &kv_storage[kv_base + n_kv_heads * head_stride + kv_h * head_stride
                        ..kv_base + n_kv_heads * head_stride + (kv_h + 1) * head_stride];

                    let scores = &mut s.scores[..seq];
                    attn_direct(
                        q_h,
                        k_src,
                        v_src,
                        seq,
                        head_dim,
                        scores,
                        &mut s.head_out,
                        cfg.arch.sliding_window,
                    );
                    let dst = h * head_dim;
                    s.attn_out[dst..dst + head_dim].copy_from_slice(&s.head_out);
                }

                layer.o_proj.matmul(&s.attn_out, 1, &mut s.attn_proj);

                lt.attn_us += attn_sw.elapsed_us();

                for i in 0..d {
                    s.x[i] += s.attn_proj[i];
                }

                let mlp_sw = Stopwatch::start();

                rmsnorm_inplace(&s.x, &layer.ffn_norm, cfg.rms_norm_eps, &mut s.normed);

                let d_ff = cfg.d_ff;
                layer.gate_proj.matmul(&s.normed, 1, &mut s.gate);
                layer.up_proj.matmul(&s.normed, 1, &mut s.up);

                for i in 0..d_ff {
                    s.gate[i] = s.gate[i] * sigmoid(s.gate[i]) * s.up[i];
                }

                layer.down_proj.matmul(&s.gate, 1, &mut s.mlp_out);

                lt.mlp_us += mlp_sw.elapsed_us();

                for i in 0..d {
                    s.x[i] += s.mlp_out[i];
                }
            }
        }

        // All layers completed — clear partial state
        self.partial_decode = None;

        // Advance KV cache position
        kv.advance();

        // 3. Final RMSNorm → normed
        rmsnorm_inplace(&s.x, &model.norm, cfg.rms_norm_eps, &mut s.normed);

        // 4. LM head: logits = x_norm × lm_head^T   [vocab_size]
        let mut logits = vec![0.0f32; cfg.vocab_size];
        model.lm_head.matmul(&s.normed, 1, &mut logits);

        Ok(logits)
    }
}

/// Resolve a layer's weights: from LRU cache (streaming) or eager `model.layers`.
fn resolve_layer<'a>(
    layer_cache: &'a mut Option<LayerCache>,
    gguf: &'a Option<Arc<GGUFModel>>,
    model: &'a LlamaModel,
    idx: usize,
) -> Result<&'a LlamaLayerWeights, Error> {
    if let Some(ref mut cache) = layer_cache {
        let gguf_ref = gguf
            .as_ref()
            .ok_or_else(|| Error::ExecutionError("LayerCache requires GGUF model".into()))?;
        cache.get(gguf_ref, idx)
    } else {
        Ok(&model.layers[idx])
    }
}

impl ScratchSpace {
    /// Total heap-allocated bytes across all vectors.
    fn total_heap_bytes(&self) -> usize {
        self.x.capacity() * 4
            + self.normed.capacity() * 4
            + self.q.capacity() * 4
            + self.k.capacity() * 4
            + self.v.capacity() * 4
            + self.attn_out.capacity() * 4
            + self.attn_proj.capacity() * 4
            + self.scores.capacity() * 4
            + self.head_out.capacity() * 4
            + self.gate.capacity() * 4
            + self.up.capacity() * 4
            + self.mlp_out.capacity() * 4
            + self.batch_x.capacity() * 4
            + self.batch_normed.capacity() * 4
            + self.batch_q.capacity() * 4
            + self.batch_k.capacity() * 4
            + self.batch_v.capacity() * 4
            + self.batch_attn.capacity() * 4
            + self.batch_ffn_hidden.capacity() * 4
            + self.batch_scores.capacity() * 4
            + self.batch_attn_proj.capacity() * 4
            + self.batch_up.capacity() * 4
            + self.batch_mlp_out.capacity() * 4
    }

    fn new(cfg: &LlamaConfig, max_seq: usize) -> Self {
        let kv_d = cfg.num_kv_heads * cfg.head_dim;
        let n_heads = cfg.num_heads;
        let n_kv = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let d_ff = cfg.d_ff;
        let d = cfg.d_model;
        // Batch buffers sized for up to 512 simultaneous prompt tokens
        let mb = cfg.max_seq_len.min(512).max(128);
        ScratchSpace {
            x: vec![0.0; d],
            normed: vec![0.0; d],
            q: vec![0.0; d],
            k: vec![0.0; kv_d],
            v: vec![0.0; kv_d],
            attn_out: vec![0.0; d],
            attn_proj: vec![0.0; d],
            scores: vec![0.0; max_seq],
            head_out: vec![0.0; head_dim],
            gate: vec![0.0; d_ff],
            up: vec![0.0; d_ff],
            mlp_out: vec![0.0; d],

            max_batch: mb,
            batch_x: vec![0.0; mb * d],
            batch_normed: vec![0.0; mb * d],
            batch_q: vec![0.0; mb * n_heads * head_dim],
            batch_k: vec![0.0; mb * n_kv * head_dim],
            batch_v: vec![0.0; mb * n_kv * head_dim],
            batch_attn: vec![0.0; mb * d],
            batch_ffn_hidden: vec![0.0; mb * d_ff],
            batch_scores: vec![0.0; mb * mb],
            batch_attn_proj: vec![0.0; mb * d],
            batch_up: vec![0.0; mb * d_ff],
            batch_mlp_out: vec![0.0; mb * d],
        }
    }
}

// ─── Math primitives ──────────────────────────────────────────────────────────

/// RMSNorm: out[i] = x[i] / rms(x) * weight[i]
fn rmsnorm_inplace(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len() as f32;
    let sum_sq = x.iter().map(|&v| v * v).sum::<f32>();
    let rms = (sum_sq / n + eps).sqrt();
    for (i, (&xi, &wi)) in x.iter().zip(weight.iter()).enumerate() {
        out[i] = xi / rms * wi;
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Scaled dot-product attention with direct KV cache indexing.
/// q: [head_dim], k_storage: [max_seq × head_dim], v_storage: [max_seq × head_dim]
/// Writes output into `out` (len=head_dim) and uses `scores` (len=seq) as scratch.
/// Only the first `seq` positions in storage are valid and accessed.
fn attn_direct(
    q: &[f32],
    k_storage: &[f32],
    v_storage: &[f32],
    seq: usize,
    head_dim: usize,
    scores: &mut [f32],
    out: &mut [f32],
    sliding_window: Option<usize>,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let window_start = match sliding_window {
        Some(w) if seq > w => seq - w,
        _ => 0,
    };

    for t in 0..window_start {
        scores[t] = f32::NEG_INFINITY;
    }

    for t in window_start..seq {
        let k_t = &k_storage[t * head_dim..(t + 1) * head_dim];
        let dot: f32 = q.iter().zip(k_t.iter()).map(|(&a, &b)| a * b).sum();
        scores[t] = dot * scale;
    }

    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut().take(seq) {
        *s = (*s - max_score).exp();
        sum += *s;
    }
    for s in scores.iter_mut().take(seq) {
        *s /= sum;
    }

    out.fill(0.0);
    for t in 0..seq {
        let v_t = &v_storage[t * head_dim..(t + 1) * head_dim];
        let w = scores[t];
        for i in 0..head_dim {
            out[i] += w * v_t[i];
        }
    }
}

/// Batched causal self-attention for prefill.
/// q: [n_tokens, n_heads * head_dim]
/// k: [n_tokens, n_kv_heads * head_dim]
/// v: [n_tokens, n_kv_heads * head_dim]
/// out: [n_tokens, n_heads * head_dim]
/// scores: [n_tokens * n_tokens] scratch
fn attention_batch(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_tokens: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
    scores: &mut [f32],
    sliding_window: Option<usize>,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_groups = n_heads / n_kv_heads;
    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv_heads * head_dim;
    let window = sliding_window.unwrap_or(usize::MAX);

    for h in 0..n_heads {
        let kv_h = h / kv_groups;

        // Q·K^T with causal + sliding window mask
        for i in 0..n_tokens {
            let q_row = &q[i * q_stride + h * head_dim..][..head_dim];
            let score_row = &mut scores[i * n_tokens..(i + 1) * n_tokens];
            let win_start = if i >= window { i - window + 1 } else { 0 };
            for j in 0..win_start {
                score_row[j] = f32::NEG_INFINITY;
            }
            for j in win_start..=i {
                let k_row = &k[j * kv_stride + kv_h * head_dim..][..head_dim];
                let dot: f32 = q_row.iter().zip(k_row.iter()).map(|(&a, &b)| a * b).sum();
                score_row[j] = dot * scale;
            }
            for j in (i + 1)..n_tokens {
                score_row[j] = f32::NEG_INFINITY;
            }
        }

        // Softmax per row (over non-masked positions)
        for i in 0..n_tokens {
            let score_row = &mut scores[i * n_tokens..(i + 1) * n_tokens];
            let max_val = score_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in score_row.iter_mut() {
                if *s > f32::NEG_INFINITY / 2.0 {
                    *s = (*s - max_val).exp();
                    sum += *s;
                }
            }
            for s in score_row.iter_mut() {
                if *s > 0.0 {
                    *s /= sum;
                }
            }
        }

        // Weighted sum of V
        for i in 0..n_tokens {
            let score_row = &scores[i * n_tokens..(i + 1) * n_tokens];
            let out_row = &mut out[i * q_stride + h * head_dim..][..head_dim];
            out_row.fill(0.0);
            for j in 0..=i {
                let w = score_row[j];
                if w <= 0.0 {
                    continue;
                }
                let v_row = &v[j * kv_stride + kv_h * head_dim..][..head_dim];
                for d in 0..head_dim {
                    out_row[d] += w * v_row[d];
                }
            }
        }
    }
}

/// Apply RoPE in-place to a [n_heads × head_dim] vector at position `pos`.
fn apply_rope_inplace(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    rope_sin: &[f32],
    rope_cos: &[f32],
) {
    let half = head_dim / 2;
    let max_seq = rope_sin.len() / half; // rope_sin is [max_seq × half]
    let clamped_pos = pos.min(max_seq.saturating_sub(1));
    let sin_row = &rope_sin[clamped_pos * half..(clamped_pos + 1) * half];
    let cos_row = &rope_cos[clamped_pos * half..(clamped_pos + 1) * half];

    for h in 0..n_heads {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        for i in 0..half {
            let x0 = head[i];
            let x1 = head[i + half];
            head[i] = x0 * cos_row[i] - x1 * sin_row[i];
            head[i + half] = x0 * sin_row[i] + x1 * cos_row[i];
        }
    }
}

/// Pre-compute RoPE sin/cos table: [max_seq × head_dim/2]
fn precompute_rope(max_seq: usize, head_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut sin_table = vec![0.0f32; max_seq * half];
    let mut cos_table = vec![0.0f32; max_seq * half];
    for pos in 0..max_seq {
        for i in 0..half {
            let theta = pos as f32 * base.powf(-2.0 * i as f32 / head_dim as f32);
            sin_table[pos * half + i] = theta.sin();
            cos_table[pos * half + i] = theta.cos();
        }
    }
    (sin_table, cos_table)
}

// ─── Sampling ─────────────────────────────────────────────────────────────────

/// Temperature + top-p (nucleus) sampling with repetition penalty.
///
/// temperature=0.0 → greedy (argmax)
/// repetition_penalty=1.0 → no penalty; >1.0 → penalize seen tokens
pub fn sample(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    prev_tokens: &HashSet<u32>,
    repetition_penalty: f32,
) -> u32 {
    // Apply repetition penalty + temperature scaling
    let inv_temp = if temperature > 1e-6 {
        1.0 / temperature
    } else {
        0.0
    };
    let mut probs = Vec::with_capacity(logits.len());
    let max_l = if temperature > 1e-6 && top_p > 1e-6 {
        let mut mx = f32::NEG_INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            let mut val = l * inv_temp;
            // repetition penalty
            if repetition_penalty > 1.0 + 1e-6 && prev_tokens.contains(&(i as u32)) {
                if val > 0.0 {
                    val /= repetition_penalty;
                } else {
                    val *= repetition_penalty;
                }
            }
            if val > mx {
                mx = val;
            }
            probs.push(val);
        }
        mx
    } else {
        // Greedy
        return logits
            .iter()
            .enumerate()
            // Model logits are finite f32 values (no NaN/inf) from the
            // quantized_matmul / LM head path, so partial_cmp never returns None.
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("logits are always finite"))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    };

    // Softmax
    let mut sum = 0.0f32;
    for p in probs.iter_mut() {
        *p = (*p - max_l).exp();
        sum += *p;
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // Top-p nucleus
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    // After softmax all probabilities are finite f32 ∈ (0,1] — no NaN/inf,
    // and the nucleus is non-empty because top_p > 0, so partial_cmp never returns None.
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).expect("probs are always finite"));

    let mut cumulative = 0.0f32;
    let mut nucleus: Vec<(usize, f32)> = Vec::new();
    for (idx, p) in &indexed {
        nucleus.push((*idx, *p));
        cumulative += p;
        if cumulative >= top_p {
            break;
        }
    }

    let nuc_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();
    let mut cdf = 0.0f32;
    let r = fast_rand_f32();
    for (idx, p) in &nucleus {
        cdf += p / nuc_sum;
        if r <= cdf {
            return *idx as u32;
        }
    }

    nucleus.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

/// Debug helper: return the top-k token indices and logit values.
fn top_k_indices(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut indexed: Vec<(usize, &f32)> = logits.iter().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed
        .iter()
        .take(k)
        .map(|(i, v)| (*i as u32, **v))
        .collect()
}

/// Fast xorshift PRNG for sampling (no external rand crate dependency).
fn fast_rand_f32() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x123456789ABCDEF);
    let mut x = STATE.load(Ordering::Relaxed);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    STATE.store(x, Ordering::Relaxed);
    (x as f32) / (u64::MAX as f32)
}

/// Estimate model byte size from raw GGUF tensor data (before layer loading).
/// Uses exact `tensor.data.len()` from the mmap-backed SharedBytes.
/// Returns (total_bytes, quantized_bytes_per_element).
pub fn estimate_model_bytes_from_gguf(gguf: &GGUFModel) -> (usize, f64) {
    let mut total = 0usize;

    for tensor in gguf.tensors.values() {
        total += tensor.data.len();
    }

    // Compute bytes per element for the first quantized weight
    let quant_bpe = gguf
        .tensors
        .get("blk.0.attn_q.weight")
        .map(|t| {
            let bs = t.dtype.block_size() as f64;
            let bbs = t.dtype.block_byte_size() as f64;
            bbs / bs
        })
        .unwrap_or(4.0);

    (total, quant_bpe)
}

pub fn estimate_model_bytes(model: &LlamaModel) -> usize {
    let emb = model.token_embeddings.len() * 4;
    let norm = model.norm.len() * 4;
    let lm = model.lm_head.data.len();
    let layers: usize = model
        .layers
        .iter()
        .map(|l| {
            let qw_size = |qw: &QuantWeight| -> usize { qw.data.len() };
            qw_size(&l.q_proj)
                + qw_size(&l.k_proj)
                + qw_size(&l.v_proj)
                + qw_size(&l.o_proj)
                + qw_size(&l.gate_proj)
                + qw_size(&l.up_proj)
                + qw_size(&l.down_proj)
                + (l.attn_norm.len() + l.ffn_norm.len()) * 4
        })
        .sum();
    emb + norm + lm + layers
}

// ─── RunnerPool ─────────────────────────────────────────────────────────────────

/// A pool of pre-allocated [`LlamaRunner`] instances sharing a single
/// [`InferenceContext`]. Enables true concurrent request processing
/// (N simultaneous inferences) instead of serializing on a single mutex.
///
/// # Usage
///
/// ```ignore
/// let pool = RunnerPool::new(ctx, &tokenizer, &layer_assignment, 4);
/// let mut runner = pool.acquire().await;
/// runner.kv.reset();
/// let logits = runner.prefill(&tokens)?;
/// // ... drop guard returns runner to pool automatically
/// ```
pub struct RunnerPool {
    runners: std::sync::Mutex<Vec<LlamaRunner>>,
    pub semaphore: tokio::sync::Semaphore,
}

impl RunnerPool {
    /// Create a pool with `count` runners sharing the given context.
    pub fn new(
        ctx: Arc<InferenceContext>,
        tokenizer: &Tokenizer,
        layer_assignment: &LayerAssignment,
        count: usize,
    ) -> Self {
        let count = count.max(1);
        let mut runners = Vec::with_capacity(count);
        for _ in 0..count {
            runners.push(LlamaRunner::new_with_context(
                Arc::clone(&ctx),
                tokenizer,
                layer_assignment,
            ));
        }
        RunnerPool {
            runners: std::sync::Mutex::new(runners),
            semaphore: tokio::sync::Semaphore::new(count),
        }
    }

    /// Acquire a runner from the pool. Blocks until one is available.
    pub async fn acquire(&self) -> RunnerGuard<'_> {
        let permit = self.semaphore.acquire().await.unwrap_or_else(|_| {
            self.semaphore.add_permits(1);
            self.semaphore.try_acquire().unwrap()
        });
        let runner = self.runners.lock().unwrap().pop().unwrap();
        RunnerGuard {
            runner: Some(runner),
            pool: self,
            permit,
        }
    }

}

/// RAII guard returned by [`RunnerPool::acquire`].
/// Auto-returns the runner to the pool when dropped.
pub struct RunnerGuard<'a> {
    runner: Option<LlamaRunner>,
    pool: &'a RunnerPool,
    #[allow(dead_code)]
    permit: tokio::sync::SemaphorePermit<'a>,
}

impl<'a> std::ops::Deref for RunnerGuard<'a> {
    type Target = LlamaRunner;
    fn deref(&self) -> &LlamaRunner {
        self.runner.as_ref().unwrap()
    }
}

impl<'a> std::ops::DerefMut for RunnerGuard<'a> {
    fn deref_mut(&mut self) -> &mut LlamaRunner {
        self.runner.as_mut().unwrap()
    }
}

impl<'a> Drop for RunnerGuard<'a> {
    fn drop(&mut self) {
        if let Some(mut runner) = self.runner.take() {
            runner.kv.reset();
            runner.kv_synced_to_gpu = false;
            runner.partial_decode = None;
            self.pool.runners.lock().unwrap().push(runner);
            // permit drops automatically when the guard dies,
            // which adds 1 back to the semaphore
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum();
        let nb: f32 = b.iter().map(|x| x * x).sum();
        (dot as f64) / ((na as f64).sqrt() * (nb as f64).sqrt())
    }

    #[test]
    fn test_buffer_upload_roundtrip() {
        let backend = match crate::backend::WgpuBackend::get_or_init() {
            Ok(b) => b,
            Err(_) => return,
        };
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let buf = backend.create_device_buffer(
            (data.len() * 4) as u64,
            wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            "test_buffer",
        );
        backend
            .queue()
            .write_buffer(&buf, 0, bytemuck::cast_slice(&data));
        let flush_enc = backend.create_encoder("flush");
        backend.submit_encoder(flush_enc);
        let readback = backend.read_buffer(&buf, data.len() * 4).unwrap();
        assert_eq!(
            readback, data,
            "Buffer upload round-trip failed, got {:?}",
            readback
        );
    }

    #[test]
    #[ignore = "requires tinyllama-q4.gguf in project root"]
    fn test_streaming_matches_in_memory() {
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tinyllama-q4.gguf");

        // Load in-memory (normal)
        let mut runner_fast = LlamaRunner::from_gguf(model_path).unwrap();
        let prompt = "The future of AI is";
        let tokens = runner_fast.tokenizer.encode(prompt, true);
        let logits_fast = runner_fast.prefill(&tokens).unwrap();
        info!(
            "[test] In-memory prefill: first 5 logits = {:?}",
            &logits_fast[..5]
        );

        // Load streaming (max_hot=2, which forces aggressive eviction)
        let mut runner_stream = LlamaRunner::from_gguf_streaming(model_path, 2).unwrap();
        let logits_stream = runner_stream.prefill(&tokens).unwrap();
        info!(
            "[test] Streaming prefill: first 5 logits = {:?}",
            &logits_stream[..5]
        );

        let sim = cosine_sim(&logits_fast, &logits_stream);
        info!(
            "[test] Cosine similarity (in-memory vs streaming) = {:.8}",
            sim
        );

        assert!(
            sim > 0.95,
            "Cosine similarity {} is below 0.95 — streaming introduces numerical drift",
            sim,
        );
    }

    /// GPU vs CPU decode parity.
    /// Enable: `AETHER_RUN_GPU_TESTS=1 cargo test --features gpu-tests`
    #[test]
    #[cfg_attr(not(feature = "gpu-tests"), ignore)]
    fn test_gpu_decode_matches_cpu_reference() {
        if std::env::var("AETHER_RUN_GPU_TESTS").ok().as_deref() != Some("1") {
            info!("[test] Skipping GPU decode test (set AETHER_RUN_GPU_TESTS=1 to enable)");
            return;
        }
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tinyllama-q4.gguf");
        let prompt = "The future of AI is";

        // GPU-enabled runner (default: ~70% RAM for GPU layers)
        let mut runner_gpu = LlamaRunner::from_gguf(model_path).unwrap();

        let tokens = runner_gpu.tokenizer.encode(prompt, true);

        // CPU-only reference runner (force all layers to CPU)
        let mut runner_cpu = LlamaRunner::from_gguf(model_path).unwrap();
        for dev in &mut runner_cpu.layer_assignment.layer_devices {
            *dev = Device::Cpu;
        }

        // Prefill — CPU-only path for both runners
        let logits_prefill_gpu = runner_gpu.prefill(&tokens).unwrap();
        let logits_prefill_cpu = runner_cpu.prefill(&tokens).unwrap();

        let sim_prefill = cosine_sim(&logits_prefill_gpu, &logits_prefill_cpu);
        info!(
            "[test] Prefill similarity (GPU runner vs CPU runner, both CPU prefill) = {:.8}",
            sim_prefill
        );
        assert!(
            sim_prefill > 0.9999,
            "Prefill mismatch between GPU and CPU runner: similarity={}",
            sim_prefill,
        );

        let num_gpu = runner_gpu
            .layer_assignment
            .layer_devices
            .iter()
            .filter(|&&d| d == Device::Wgpu)
            .count();
        if num_gpu == 0 {
            info!("[test] No GPU layers assigned — GPU decode path not tested");
            return;
        }

        // Greedy decode of first token (temperature=0 for reproducibility)
        let prev_set: HashSet<u32> = tokens.iter().copied().collect();
        let first_token = sample(&logits_prefill_gpu, 0.0, 0.9, &prev_set, 1.0);
        let pos = tokens.len();

        // GPU-runner decode step — GPU layers use GPU path
        let mut tel_gpu = vec![LayerTelemetry::default(); runner_gpu.cfg.num_layers];
        let logits_decode_gpu = runner_gpu
            .decode_step(first_token, pos, &mut tel_gpu)
            .unwrap();

        // CPU-runner decode step — all layers use CPU path
        let mut tel_cpu = vec![LayerTelemetry::default(); runner_cpu.cfg.num_layers];
        let logits_decode_cpu = runner_cpu
            .decode_step(first_token, pos, &mut tel_cpu)
            .unwrap();

        let gpu_nan: usize = logits_decode_gpu.iter().map(|&x| x.is_nan() as usize).sum();
        let gpu_finite: usize = logits_decode_gpu
            .iter()
            .map(|&x| x.is_finite() as usize)
            .sum();
        info!(
            "[test] GPU decode logits: NaN={}, finite={}, first5={:?}",
            gpu_nan,
            gpu_finite,
            &logits_decode_gpu[..5.min(logits_decode_gpu.len())]
        );

        if gpu_finite == 0 {
            panic!("All GPU decode logits are non-finite (NaN/Inf) — GPU path broken");
        }

        let sim_decode = cosine_sim(&logits_decode_gpu, &logits_decode_cpu);
        info!(
            "[test] GPU ({}/{}) vs CPU decode cosine similarity = {:.4}",
            num_gpu, runner_gpu.cfg.num_layers, sim_decode,
        );

        // Numerical differences between GPU (Metal) and CPU (LLVM) f32 arithmetic
        // can reduce cosine similarity. We check the result is non-catastrophic.
        assert!(
            sim_decode > 0.99,
            "GPU decode diverged from CPU (Metal f32 vs LLVM f32): similarity={:.4}",
            sim_decode,
        );
    }
}
