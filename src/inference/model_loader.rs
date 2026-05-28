use crate::loader::dequant::dequantize;
use crate::loader::gguf::{GGUFDtype, GGUFModel, GGUFValue, SharedBytes};
use crate::Error;
/// GGUF → LlamaModel weight loader.
///
/// Maps GGUF tensor names (blk.N.* format used by llama.cpp) to typed weight
/// slots. Reads model hyper-parameters from GGUF metadata.
///
/// Weight storage: keeps raw quantized bytes + dtype rather than dequantizing
/// upfront. Dequantization happens inline during forward pass via the fused
/// quantized matmul kernels in `crate::quant`.
use std::collections::HashMap;
use tracing::{info, warn};

/// Model architecture detected from GGUF `general.architecture`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelArchitecture {
    Llama,
    Mistral,
    Phi3,
    Qwen2,
    Gemma2,
    DeepSeek2,
    Unknown,
}

impl ModelArchitecture {
    pub fn detect(meta: &HashMap<String, GGUFValue>) -> Self {
        match meta.get("general.architecture") {
            Some(GGUFValue::String(s)) => match s.as_str() {
                "llama" => Self::Llama,
                "mistral" => Self::Mistral,
                "phi3" => Self::Phi3,
                "qwen2" => Self::Qwen2,
                "gemma2" => Self::Gemma2,
                "deepseek2" => Self::DeepSeek2,
                _ => {
                    warn!("Unknown architecture '{}', treating as Llama", s);
                    Self::Llama
                }
            },
            _ => {
                warn!("No general.architecture field, assuming Llama");
                Self::Llama
            }
        }
    }

    pub fn key_prefix(&self) -> &'static str {
        match self {
            Self::Llama => "llama.",
            Self::Mistral => "mistral.",
            Self::Phi3 => "phi3.",
            Self::Qwen2 => "qwen2.",
            Self::Gemma2 => "gemma2.",
            Self::DeepSeek2 => "deepseek2.",
            Self::Unknown => "",
        }
    }
}

/// Architecture-specific metadata keys resolved from GGUF.
#[derive(Debug, Clone)]
pub struct ArchConfig {
    pub architecture: ModelArchitecture,
    pub sliding_window: Option<usize>,
}

/// Configuration extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub d_model: usize,      // embedding_length
    pub num_layers: usize,   // block_count
    pub num_heads: usize,    // attention.head_count
    pub num_kv_heads: usize, // attention.head_count_kv (GQA)
    pub head_dim: usize,     // d_model / num_heads
    pub d_ff: usize,         // feed_forward_length
    pub max_seq_len: usize,  // context_length
    pub rope_base: f32,      // rope.freq_base
    pub rms_norm_eps: f32,   // attention.layer_norm_rms_epsilon
    pub rope_dim: usize,     // rope.dimension_count (often head_dim)
    pub arch: ArchConfig,    // architecture-specific config
}

/// One quantized weight tensor: raw bytes + dtype + shape.
#[derive(Clone)]
pub struct QuantWeight {
    pub data: SharedBytes,
    pub dtype: GGUFDtype,
    pub shape: Vec<usize>, // [out_features, in_features] (GGUF convention)
    pub f32_data: Option<Vec<f32>>, // pre-dequantized f32 (hot cache)
}

impl QuantWeight {
    /// Materialize to f32. For hot-path inference, prefer the fused matmul kernels.
    pub fn to_f32(&self) -> Vec<f32> {
        dequantize(&self.data, self.dtype, &self.shape)
    }

    /// out_features (rows in GGUF weight layout)
    pub fn out_features(&self) -> usize {
        self.shape[0]
    }

    /// in_features (columns)
    pub fn in_features(&self) -> usize {
        self.shape[1]
    }

    /// Bytes needed for GPU storage after dequantization to f32.
    /// The GPU stores all weights as f32 (4 bytes per element).
    pub fn gpu_storage_bytes(&self) -> usize {
        let total_elements: usize = self.shape.iter().product();
        total_elements * 4
    }
}

impl QuantWeight {
    /// Run y = x @ W^T, using the optional f32 cache when available.
    pub fn matmul(&self, a: &[f32], m: usize, c: &mut [f32]) {
        crate::quant::quantized_matmul_impl(
            a,
            m,
            &self.data,
            &self.shape,
            self.dtype,
            c,
            self.f32_data.as_deref(),
        )
    }
}

/// A single Llama decoder layer's weights.
pub struct LlamaLayerWeights {
    pub attn_norm: Vec<f32>,    // RMSNorm weight [d_model] — always F32/F16
    pub q_proj: QuantWeight,    // [d_model, d_model]
    pub k_proj: QuantWeight,    // [d_model_kv, d_model]
    pub v_proj: QuantWeight,    // [d_model_kv, d_model]
    pub o_proj: QuantWeight,    // [d_model, d_model]
    pub ffn_norm: Vec<f32>,     // RMSNorm weight [d_model]
    pub gate_proj: QuantWeight, // [d_ff, d_model]
    pub up_proj: QuantWeight,   // [d_ff, d_model]
    pub down_proj: QuantWeight, // [d_model, d_ff]
}

/// Full LlamaModel weights loaded from GGUF.
pub struct LlamaModel {
    pub config: LlamaConfig,
    pub token_embeddings: Vec<f32>, // [vocab_size × d_model] — dequantized
    pub layers: Vec<LlamaLayerWeights>,
    pub norm: Vec<f32>,       // Final RMSNorm [d_model]
    pub lm_head: QuantWeight, // [vocab_size × d_model] — quantized
}

impl LlamaModel {
    /// Build `LlamaModel` from parsed GGUF data.
    ///
    /// If `lazy_layers = true`, decoder layers are not loaded eagerly.
    /// This saves memory when using streaming (layers are loaded on-demand
    /// by `LayerCache`). The `layers` field will be empty.
    pub fn from_gguf(model: &GGUFModel, lazy_layers: bool) -> Result<Self, Error> {
        let cfg = LlamaConfig::from_gguf(model)?;
        info!(
            "Model config: d_model={} layers={} heads={} kv_heads={} d_ff={} vocab={}",
            cfg.d_model, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads, cfg.d_ff, cfg.vocab_size
        );

        let tensors = &model.tensors;

        // ── Token embeddings ──────────────────────────────────────────────────
        let token_embeddings_raw = load_f32_tensor(tensors, "token_embd.weight", &cfg)?;
        // GGUF stores token_embd.weight as [d_model, vocab_size] row-major
        // (data[d * vocab_size + v] = dimension d, token v). We need
        // [vocab_size, d_model] for the contiguous embedding lookup. Transpose:
        let d = cfg.d_model;
        let vs = cfg.vocab_size;
        let mut token_embeddings = vec![0.0f32; token_embeddings_raw.len()];
        for d_idx in 0..d {
            let src_off = d_idx * vs;
            for v_idx in 0..vs {
                token_embeddings[v_idx * d + d_idx] = token_embeddings_raw[src_off + v_idx];
            }
        }

        // ── Decoder layers ────────────────────────────────────────────────────
        let layers = if lazy_layers {
            info!("Using lazy layer loading (streaming mode)");
            Vec::new()
        } else {
            let mut layers = Vec::with_capacity(cfg.num_layers);
            for i in 0..cfg.num_layers {
                let prefix = format!("blk.{}.", i);

                let attn_norm =
                    load_f32_norm(tensors, &format!("{}attn_norm.weight", prefix), cfg.d_model)?;
                let q_proj = load_quant(tensors, &format!("{}attn_q.weight", prefix))?;
                let k_proj = load_quant(tensors, &format!("{}attn_k.weight", prefix))?;
                let v_proj = load_quant(tensors, &format!("{}attn_v.weight", prefix))?;
                let o_proj = load_quant(tensors, &format!("{}attn_output.weight", prefix))?;
                let ffn_norm =
                    load_f32_norm(tensors, &format!("{}ffn_norm.weight", prefix), cfg.d_model)?;
                let gate_proj = load_quant(tensors, &format!("{}ffn_gate.weight", prefix))?;
                let up_proj = load_quant(tensors, &format!("{}ffn_up.weight", prefix))?;
                let down_proj = load_quant(tensors, &format!("{}ffn_down.weight", prefix))?;

                layers.push(LlamaLayerWeights {
                    attn_norm,
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    ffn_norm,
                    gate_proj,
                    up_proj,
                    down_proj,
                });

                if (i + 1) % 4 == 0 || i == cfg.num_layers - 1 {
                    info!("Loaded layer {}/{}", i + 1, cfg.num_layers);
                }
            }
            layers
        };

        // ── Final norm + LM head ──────────────────────────────────────────────
        let norm = load_f32_norm(tensors, "output_norm.weight", cfg.d_model)?;
        let lm_head = load_quant(tensors, "output.weight").unwrap_or_else(|_| QuantWeight {
            data: SharedBytes::new_owned(vec![]),
            dtype: GGUFDtype::F32,
            shape: vec![cfg.vocab_size, cfg.d_model],
            f32_data: None,
        });

        Ok(LlamaModel {
            config: cfg,
            token_embeddings,
            layers,
            norm,
            lm_head,
        })
    }
}

impl LlamaConfig {
    pub fn from_gguf(model: &GGUFModel) -> Result<Self, Error> {
        let meta = &model.metadata;
        let arch = ModelArchitecture::detect(meta);

        info!("Architecture: {:?}", arch);

        // Resolve key with arch-specific prefix + fallback
        let key = |k: &str| -> String {
            let arch_key = format!("{}{}", arch.key_prefix(), k);
            if meta.contains_key(&arch_key) {
                return arch_key;
            }
            let llama_key = format!("llama.{}", k);
            if meta.contains_key(&llama_key) {
                return llama_key;
            }
            k.to_string()
        };

        let d_model = gguf_usize(meta, &key("embedding_length"))
            .ok_or_else(|| Error::ExecutionError("Missing embedding_length".into()))?;
        let num_layers = gguf_usize(meta, &key("block_count"))
            .ok_or_else(|| Error::ExecutionError("Missing block_count".into()))?;
        let num_heads = gguf_usize(meta, &key("attention.head_count"))
            .ok_or_else(|| Error::ExecutionError("Missing attention.head_count".into()))?;
        let num_kv_heads = gguf_usize(meta, &key("attention.head_count_kv")).unwrap_or(num_heads);
        let d_ff = gguf_usize(meta, &key("feed_forward_length"))
            .ok_or_else(|| Error::ExecutionError("Missing feed_forward_length".into()))?;
        let vocab_size = match model.tensors.get("token_embd.weight") {
            Some(t) => t.shape[1],
            None => return Err(Error::ExecutionError("Cannot determine vocab size".into())),
        };
        let max_seq_len = gguf_usize(meta, &key("context_length")).unwrap_or(2048);
        let rope_base = gguf_f32(meta, &key("rope.freq_base")).unwrap_or(10000.0);
        let rms_norm_eps = gguf_f32(meta, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);
        let head_dim = d_model / num_heads;
        let rope_dim = gguf_usize(meta, &key("rope.dimension_count")).unwrap_or(head_dim);

        let sliding_window = gguf_usize(meta, &key("attention.sliding_window_length"));

        Ok(LlamaConfig {
            vocab_size,
            d_model,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            d_ff,
            max_seq_len,
            rope_base,
            rms_norm_eps,
            rope_dim,
            arch: ArchConfig {
                architecture: arch,
                sliding_window,
            },
        })
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

pub(crate) type TensorMap = HashMap<String, crate::loader::gguf::GGUFTensor>;

pub(crate) fn load_f32_norm(tensors: &TensorMap, name: &str, expected_len: usize) -> Result<Vec<f32>, Error> {
    let t = tensors
        .get(name)
        .ok_or_else(|| Error::ExecutionError(format!("Missing norm tensor: {}", name)))?;
    let n: usize = t.shape.iter().product();
    if n != expected_len {
        return Err(Error::ExecutionError(format!(
            "Norm tensor {} has {} elements, expected {}",
            name, n, expected_len
        )));
    }
    Ok(dequantize(&t.data, t.dtype, &t.shape))
}

fn load_quant(tensors: &TensorMap, name: &str) -> Result<QuantWeight, Error> {
    let t = tensors
        .get(name)
        .ok_or_else(|| Error::ExecutionError(format!("Missing tensor: {}", name)))?;
    // GGUF stores 2D tensor shapes as [in_features, out_features].
    // We want [out_features, in_features] so that one output neuron's
    // weights are contiguous in memory (for efficient SIMD matmul).
    // Transpose the raw quantized bytes to match this layout.
    let shape = if t.shape.len() == 2 {
        vec![t.shape[1], t.shape[0]]
    } else {
        t.shape.clone()
    };

    let data = if t.shape.len() == 2 && is_quantized_type_supported(t.dtype) {
        let k = t.shape[0];
        let n = t.shape[1];
        transpose_quantized_bytes(&t.data, t.dtype, k, n, &shape)?
    } else {
        t.data.clone()
    };

    Ok(QuantWeight {
        data,
        dtype: t.dtype,
        shape,
        f32_data: None,
    })
}

/// Transpose quantized weight bytes from GGUF [K, N] layout to [N, K] layout
/// (output-major, so weights for one output neuron are contiguous).
fn is_quantized_type_supported(dtype: GGUFDtype) -> bool {
    matches!(
        dtype,
        GGUFDtype::Q8_0 | GGUFDtype::Q4_K | GGUFDtype::Q5_K | GGUFDtype::Q6_K
    )
}

fn transpose_quantized_bytes(
    data: &SharedBytes,
    dtype: GGUFDtype,
    k: usize,
    n: usize,
    target_shape: &[usize],
) -> Result<SharedBytes, Error> {
    let f32_data = dequantize(data, dtype, &[k, n]);
    let mut transposed = vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            transposed[j * k + i] = f32_data[i * n + j];
        }
    }
    let requantized = crate::quant::requantize(&transposed, dtype, target_shape)
        .map_err(|e| Error::ExecutionError(format!("requantize: {}", e)))?;
    Ok(SharedBytes::new_owned(requantized))
}

pub(crate) fn load_f32_tensor(tensors: &TensorMap, name: &str, cfg: &LlamaConfig) -> Result<Vec<f32>, Error> {
    let t = tensors
        .get(name)
        .ok_or_else(|| Error::ExecutionError(format!("Missing tensor: {}", name)))?;
    let expected = cfg.vocab_size * cfg.d_model;
    let actual: usize = t.shape.iter().product();
    // GGUF token embedding shape is [vocab_size, d_model]
    if actual != expected {
        warn!(
            "token_embd.weight has {} elements, expected {} (vocab={} d_model={})",
            actual, expected, cfg.vocab_size, cfg.d_model
        );
    }
    Ok(dequantize(&t.data, t.dtype, &t.shape))
}

fn gguf_usize(meta: &HashMap<String, GGUFValue>, key: &str) -> Option<usize> {
    match meta.get(key) {
        Some(GGUFValue::Uint32(v)) => Some(*v as usize),
        Some(GGUFValue::Uint64(v)) => Some(*v as usize),
        Some(GGUFValue::Int32(v)) => Some(*v as usize),
        Some(GGUFValue::Int64(v)) => Some(*v as usize),
        _ => None,
    }
}

fn gguf_f32(meta: &HashMap<String, GGUFValue>, key: &str) -> Option<f32> {
    match meta.get(key) {
        Some(GGUFValue::Float32(v)) => Some(*v),
        Some(GGUFValue::Float64(v)) => Some(*v as f32),
        _ => None,
    }
}
