use crate::inference::model_loader::{LlamaConfig, LlamaLayerWeights, QuantWeight};
use crate::loader::gguf::{GGUFDtype, GGUFModel, SharedBytes};
use crate::Error;
use std::sync::Arc;
use tracing::warn;

/// LRU layer cache backed by mmap.
///
/// Holds a hot set of `LlamaLayerWeights` in memory. Cold layers are dropped
/// (freeing their norm vectors and SharedBytes references). The OS manages
/// the actual kernel-level page cache — dropping a layer's `SharedBytes` ref
/// does NOT munmap the file, but reduces the process's virtual memory working
/// set. The kernel's page reclaim handles actual memory pressure.
pub struct LayerCache {
    config: LlamaConfig,
    /// Hot layer weights: None = cold, Some = resident.
    weights: Vec<Option<LlamaLayerWeights>>,
    /// LRU order: indices into `weights`, [0] = least recently used.
    lru: Vec<usize>,
    /// Maximum number of hot layers.
    max_hot: usize,
    /// Pre-dequantize asymmetric weights to f32 on load (reduces per-token dequant cost).
    enable_f32_cache: bool,
}

impl LayerCache {
    /// Build a new `LayerCache`.
    ///
    /// `max_hot_layers = 0` means no limit (load all eagerly — equivalent
    /// to the non-streaming path).
    pub fn new(config: LlamaConfig, max_hot_layers: usize) -> Self {
        Self::with_f32_cache(config, max_hot_layers, false)
    }

    /// Create a `LayerCache` with optional pre-dequantized f32 weight caching.
    /// When `enable_f32_cache` is true, asymmetric weights are dequantized to f32
    /// on load, eliminating per-token dequant overhead at the cost of ~3× more
    /// memory per cached asymmetric weight.
    pub fn with_f32_cache(
        config: LlamaConfig,
        max_hot_layers: usize,
        enable_f32_cache: bool,
    ) -> Self {
        let num_layers = config.num_layers;
        let max_hot = if max_hot_layers == 0 || max_hot_layers >= num_layers {
            num_layers // load all layers eagerly
        } else {
            max_hot_layers
        };

        let weights: Vec<Option<LlamaLayerWeights>> = (0..num_layers).map(|_| None).collect();
        let lru: Vec<usize> = Vec::with_capacity(max_hot);

        LayerCache {
            config,
            weights,
            lru,
            max_hot,
            enable_f32_cache,
        }
    }

    /// Preload the first batch of layers (up to `max_hot`).
    pub fn preload(&mut self, gguf: &Arc<GGUFModel>) {
        let to_load = self.max_hot.min(self.config.num_layers);
        for i in 0..to_load {
            if self.weights[i].is_none() {
                match Self::load_layer(gguf, i, &self.config, self.enable_f32_cache) {
                    Ok(layer) => {
                        self.weights[i] = Some(layer);
                        self.lru.push(i);
                    }
                    Err(e) => warn!("Failed to preload layer {}: {}", i, e),
                }
            }
        }
    }

    /// Get a layer's weights, loading from mmap if cold and evicting if hot.
    /// Panics if `layer_idx` is out of range (caller must validate < num_layers).
    pub fn get(
        &mut self,
        gguf: &Arc<GGUFModel>,
        layer_idx: usize,
    ) -> Result<&LlamaLayerWeights, Error> {
        if layer_idx >= self.config.num_layers {
            return Err(Error::ExecutionError(format!(
                "Layer index {} out of range (max {})",
                layer_idx, self.config.num_layers,
            )));
        }

        if self.weights[layer_idx].is_some() {
            self.touch(layer_idx);
            return Ok(self.weights[layer_idx]
                .as_ref()
                .expect("Layer weights should be Some after is_some() check"));
        }

        // Evict LRU if at capacity
        let hot_count = self.lru.len();
        if hot_count >= self.max_hot {
            self.evict_lru();
        }

        // Load layer from GGUF
        let layer = Self::load_layer(gguf, layer_idx, &self.config, self.enable_f32_cache)?;
        self.weights[layer_idx] = Some(layer);
        self.lru.push(layer_idx);

        Ok(self.weights[layer_idx]
            .as_ref()
            .expect("Layer weights should be Some after load"))
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Load a single layer's weights from the GGUF model by looking up tensor names.
    fn load_layer(
        gguf: &Arc<GGUFModel>,
        layer_idx: usize,
        cfg: &LlamaConfig,
        enable_f32_cache: bool,
    ) -> Result<LlamaLayerWeights, Error> {
        let prefix = format!("blk.{}.", layer_idx);
        let d = cfg.d_model;

        let attn_norm = load_norm_from_gguf(gguf, &format!("{}attn_norm.weight", prefix), d);
        let q_proj =
            load_quant_from_gguf(gguf, &format!("{}attn_q.weight", prefix), enable_f32_cache)
                .ok_or_else(|| {
                    Error::ExecutionError(format!("Missing tensor: {}attn_q.weight", prefix))
                })?;
        let k_proj =
            load_quant_from_gguf(gguf, &format!("{}attn_k.weight", prefix), enable_f32_cache)
                .ok_or_else(|| {
                    Error::ExecutionError(format!("Missing tensor: {}attn_k.weight", prefix))
                })?;
        let v_proj =
            load_quant_from_gguf(gguf, &format!("{}attn_v.weight", prefix), enable_f32_cache)
                .ok_or_else(|| {
                    Error::ExecutionError(format!("Missing tensor: {}attn_v.weight", prefix))
                })?;
        let o_proj = load_quant_from_gguf(
            gguf,
            &format!("{}attn_output.weight", prefix),
            enable_f32_cache,
        )
        .ok_or_else(|| {
            Error::ExecutionError(format!("Missing tensor: {}attn_output.weight", prefix))
        })?;
        let ffn_norm = load_norm_from_gguf(gguf, &format!("{}ffn_norm.weight", prefix), d);
        let gate_proj = load_quant_from_gguf(
            gguf,
            &format!("{}ffn_gate.weight", prefix),
            enable_f32_cache,
        )
        .ok_or_else(|| {
            Error::ExecutionError(format!("Missing tensor: {}ffn_gate.weight", prefix))
        })?;
        let up_proj =
            load_quant_from_gguf(gguf, &format!("{}ffn_up.weight", prefix), enable_f32_cache)
                .ok_or_else(|| {
                    Error::ExecutionError(format!("Missing tensor: {}ffn_up.weight", prefix))
                })?;
        let down_proj = load_quant_from_gguf(
            gguf,
            &format!("{}ffn_down.weight", prefix),
            enable_f32_cache,
        )
        .ok_or_else(|| {
            Error::ExecutionError(format!("Missing tensor: {}ffn_down.weight", prefix))
        })?;

        Ok(LlamaLayerWeights {
            attn_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            ffn_norm,
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Mark a layer as recently used.
    fn touch(&mut self, layer_idx: usize) {
        if let Some(pos) = self.lru.iter().position(|&i| i == layer_idx) {
            self.lru.remove(pos);
        }
        self.lru.push(layer_idx);
    }

    /// Evict the least recently used layer.
    fn evict_lru(&mut self) {
        if self.lru.is_empty() {
            return;
        }
        let lru_idx = self.lru.remove(0);
        let _ = self.weights[lru_idx].take();
    }
}

// ── Helper functions ─────────────────────────────────────────────────────────

fn load_norm_from_gguf(gguf: &Arc<GGUFModel>, name: &str, _d_model: usize) -> Vec<f32> {
    let tensor = match gguf.tensors.get(name) {
        Some(t) => t,
        None => {
            warn!("Missing norm tensor: {}", name);
            return vec![];
        }
    };
    let data = &tensor.data;
    let bytes: &[u8] = data;
    match tensor.dtype {
        GGUFDtype::F32 => {
            if (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()) {
                let floats = bytemuck::cast_slice(bytes);
                floats.to_vec()
            } else {
                let mut floats = vec![0.0f32; bytes.len() / 4];
                for i in 0..floats.len() {
                    floats[i] = f32::from_ne_bytes([
                        bytes[i * 4],
                        bytes[i * 4 + 1],
                        bytes[i * 4 + 2],
                        bytes[i * 4 + 3],
                    ]);
                }
                floats
            }
        }
        GGUFDtype::F16 => {
            if (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<half::f16>()) {
                let halves = bytemuck::cast_slice::<_, half::f16>(bytes);
                halves.iter().map(|h| h.to_f32()).collect()
            } else {
                let mut floats = Vec::with_capacity(bytes.len() / 2);
                for i in 0..bytes.len() / 2 {
                    let bits = u16::from_ne_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    floats.push(half::f16::from_bits(bits).to_f32());
                }
                floats
            }
        }
        _ => {
            warn!("Unsupported norm dtype {:?} for {}", tensor.dtype, name);
            vec![]
        }
    }
}

fn load_quant_from_gguf(
    gguf: &Arc<GGUFModel>,
    name: &str,
    enable_f32_cache: bool,
) -> Option<QuantWeight> {
    let tensor = match gguf.tensors.get(name) {
        Some(t) => t,
        None => {
            warn!("Missing quant tensor: {}", name);
            return None;
        }
    };
    let shape = if tensor.shape.len() == 2 {
        vec![tensor.shape[1], tensor.shape[0]]
    } else {
        tensor.shape.clone()
    };

    // Pre-dequantize asymmetric weights for hot cache when enabled
    let f32_data = if enable_f32_cache && shape[0] != shape[1] {
        // Dequantize using GGUF [in, out] order = gguf tensor raw shape
        Some(crate::loader::dequant::dequantize(
            &tensor.data,
            tensor.dtype,
            &[tensor.shape[0], tensor.shape[1]],
        ))
    } else {
        None
    };

    let is_supported = matches!(
        tensor.dtype,
        GGUFDtype::Q8_0 | GGUFDtype::Q4_K | GGUFDtype::Q5_K | GGUFDtype::Q6_K
    );

    let data = if tensor.shape.len() == 2 && is_supported {
        let k = tensor.shape[0];
        let n = tensor.shape[1];
        let f32_data = crate::loader::dequant::dequantize(&tensor.data, tensor.dtype, &[k, n]);
        let mut transposed = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                transposed[j * k + i] = f32_data[i * n + j];
            }
        }
        let requantized = match crate::quant::requantize(&transposed, tensor.dtype, &shape) {
            Ok(rq) => rq,
            Err(_) => return None,
        };
        SharedBytes::new_owned(requantized)
    } else {
        tensor.data.clone()
    };

    Some(QuantWeight {
        data,
        dtype: tensor.dtype,
        shape,
        f32_data,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{ArchConfig, ModelArchitecture};
    use crate::loader::gguf::GGUFLoader;
    use std::io::Write;
    use std::sync::Arc;

    fn write_u32(w: &mut impl Write, v: u32) {
        w.write_all(&v.to_le_bytes()).unwrap();
    }
    fn write_u64(w: &mut impl Write, v: u64) {
        w.write_all(&v.to_le_bytes()).unwrap();
    }
    fn write_f32_le(w: &mut impl Write, v: f32) {
        w.write_all(&v.to_le_bytes()).unwrap();
    }
    fn write_string(w: &mut impl Write, s: &str) {
        write_u64(w, s.len() as u64);
        w.write_all(s.as_bytes()).unwrap();
    }
    fn pad_to(w: &mut Vec<u8>, align: usize) {
        while !w.len().is_multiple_of(align) {
            w.push(0);
        }
    }

    /// Build a minimal GGUF with `num_layers` layers, each having the required
    /// tensors: attn_norm.weight (F32, [d_model]), attn_q.weight (F32, [d_model, d_model]),
    /// attn_k.weight, attn_v.weight, attn_output.weight, ffn_norm.weight,
    /// ffn_gate.weight, ffn_up.weight, ffn_down.weight.
    /// We use F32 for everything (simpler, still exercises the cache machinery).
    fn build_mock_model(num_layers: usize, d_model: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        let magic: u32 = 0x46554747;
        write_u32(&mut buf, magic);
        write_u32(&mut buf, 3); // version
        write_u64(&mut buf, 0); // tensor_count (patched later)
        write_u64(&mut buf, 2); // metadata_kv_count

        // Metadata: block_count
        write_string(&mut buf, "llama.block_count");
        write_u32(&mut buf, 4); // uint32 KV
        write_u32(&mut buf, num_layers as u32);

        // Metadata: d_model
        write_string(&mut buf, "llama.embedding_length");
        write_u32(&mut buf, 4);
        write_u32(&mut buf, d_model as u32);

        // Per-layer tensor names
        let layer_tensor_names: Vec<String> = (0..num_layers)
            .flat_map(|i| {
                let p = format!("blk.{}.", i);
                vec![
                    format!("{}attn_norm.weight", p),
                    format!("{}attn_q.weight", p),
                    format!("{}attn_k.weight", p),
                    format!("{}attn_v.weight", p),
                    format!("{}attn_output.weight", p),
                    format!("{}ffn_norm.weight", p),
                    format!("{}ffn_gate.weight", p),
                    format!("{}ffn_up.weight", p),
                    format!("{}ffn_down.weight", p),
                ]
            })
            .collect();

        // Per-weight-tensor size in bytes (F32, [d_model, d_model])
        let weight_bytes = (d_model * d_model * 4) as u64;
        // Per-norm-tensor size in bytes (F32, [d_model])
        let norm_bytes = (d_model * 4) as u64;

        // Compute tensor data offsets
        let mut offsets: Vec<u64> = Vec::new();
        let mut cur_offset: u64 = 0;
        for name in &layer_tensor_names {
            offsets.push(cur_offset);
            if name.contains("norm") {
                cur_offset += norm_bytes;
            } else {
                cur_offset += weight_bytes;
            }
        }

        // Tensor info section
        let tensor_count = layer_tensor_names.len() as u64;
        // Write tensor infos
        for (name, &offset) in layer_tensor_names.iter().zip(offsets.iter()) {
            write_string(&mut buf, name); // tensor name
                                          // n_dims
            if name.contains("norm") {
                write_u32(&mut buf, 1); // 1D
            } else {
                write_u32(&mut buf, 2); // 2D
            }
            // dims
            if name.contains("norm") {
                write_u64(&mut buf, d_model as u64);
            } else {
                write_u64(&mut buf, d_model as u64);
                write_u64(&mut buf, d_model as u64);
            }
            write_u32(&mut buf, 0); // dtype = F32
            write_u64(&mut buf, offset);
        }

        // Patch tensor_count at offset 8 (magic=4 + version=4)
        let tensor_count_bytes = tensor_count.to_le_bytes();
        buf[8..16].copy_from_slice(&tensor_count_bytes);

        // Align to 32 bytes
        pad_to(&mut buf, 32);

        // Write tensor data
        for name in layer_tensor_names.iter() {
            let n_elems = if name.contains("norm") {
                d_model
            } else {
                d_model * d_model
            };
            for _ in 0..n_elems {
                write_f32_le(&mut buf, 0.001); // dummy data
            }
        }

        buf
    }

    /// Build a LlamaConfig for the mock model.
    fn mock_config(num_layers: usize, d_model: usize) -> LlamaConfig {
        let head_dim = d_model / 2;
        LlamaConfig {
            d_model,
            num_layers,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim,
            d_ff: d_model * 4,
            vocab_size: 128,
            max_seq_len: 128,
            rms_norm_eps: 1e-6,
            rope_base: 10000.0,
            rope_dim: head_dim,
            arch: ArchConfig {
                architecture: ModelArchitecture::Llama,
                sliding_window: None,
            },
        }
    }

    /// Load a synthetic GGUF and return (Arc<GGUFModel>, LlamaConfig).
    /// Each test must call with a unique `label` to avoid temp file races.
    fn load_synthetic(
        label: &str,
        num_layers: usize,
        d_model: usize,
    ) -> (Arc<GGUFModel>, LlamaConfig) {
        let buf = build_mock_model(num_layers, d_model);
        let tmp = std::env::temp_dir().join(format!("aether_test_layer_cache_{}.gguf", label));
        // Clean up previous run's leftovers
        let _ = std::fs::remove_file(&tmp);
        std::fs::write(&tmp, &buf).unwrap();
        let model = GGUFLoader::load(tmp.to_str().unwrap()).unwrap();
        std::fs::remove_file(&tmp).ok();
        let cfg = mock_config(num_layers, d_model);
        (Arc::new(model), cfg)
    }

    #[test]
    fn test_cache_all_layers_fit() {
        let (gguf, cfg) = load_synthetic("all_fit", 4, 32);
        let mut cache = LayerCache::new(cfg.clone(), 10);
        cache.preload(&gguf);
        for i in 0..4 {
            let layer = cache.get(&gguf, i).unwrap();
            assert_eq!(layer.attn_norm.len(), cfg.d_model);
            assert_eq!(layer.q_proj.shape[0], cfg.d_model);
            assert_eq!(layer.q_proj.shape[1], cfg.d_model);
        }
    }

    #[test]
    fn test_cache_zero_max_hot_means_all() {
        let (_gguf, cfg) = load_synthetic("zero_max", 4, 32);
        let cache = LayerCache::new(cfg.clone(), 0);
        assert_eq!(cache.max_hot, cfg.num_layers);
    }

    #[test]
    fn test_cache_preload_respects_max_hot() {
        let (gguf, cfg) = load_synthetic("preload_hot", 6, 32);
        let mut cache = LayerCache::new(cfg.clone(), 2);
        cache.preload(&gguf);
        assert!(cache.weights[0].is_some());
        assert!(cache.weights[1].is_some());
        for i in 2..6 {
            assert!(cache.weights[i].is_none());
        }
    }

    #[test]
    fn test_cache_lru_eviction_order() {
        let (gguf, cfg) = load_synthetic("evict_order", 6, 32);
        let mut cache = LayerCache::new(cfg.clone(), 2);

        cache.get(&gguf, 0).unwrap();
        cache.get(&gguf, 1).unwrap();
        assert_eq!(cache.lru, vec![0, 1]);

        cache.get(&gguf, 0).unwrap();
        assert_eq!(cache.lru, vec![1, 0]);

        cache.get(&gguf, 2).unwrap();
        assert_eq!(cache.lru, vec![0, 2]);
        assert!(cache.weights[1].is_none());
        assert!(cache.weights[2].is_some());
    }

    #[test]
    fn test_cache_reload_after_eviction() {
        let (gguf, cfg) = load_synthetic("reload", 6, 32);
        let mut cache = LayerCache::new(cfg.clone(), 2);

        cache.get(&gguf, 0).unwrap();
        cache.get(&gguf, 1).unwrap();
        cache.get(&gguf, 2).unwrap();
        assert!(cache.weights[0].is_none());
        assert!(cache.weights[2].is_some());

        cache.get(&gguf, 0).unwrap();
        assert!(cache.weights[1].is_none());
        assert!(cache.weights[0].is_some());
        assert_eq!(cache.lru, vec![2, 0]);
    }

    #[test]
    fn test_cache_thrashing_sequential_access() {
        let (gguf, cfg) = load_synthetic("sequential", 8, 32);
        let mut cache = LayerCache::new(cfg.clone(), 2);

        for i in 0..8 {
            cache.get(&gguf, i).unwrap();
        }

        assert_eq!(cache.lru, vec![6, 7]);

        for i in 0..8 {
            let layer = cache.get(&gguf, i).unwrap();
            assert_eq!(layer.attn_norm.len(), cfg.d_model);
        }
    }

    #[test]
    fn test_cache_reverse_access() {
        let (gguf, cfg) = load_synthetic("reverse", 5, 32);
        let mut cache = LayerCache::new(cfg.clone(), 3);

        for i in (0..5).rev() {
            cache.get(&gguf, i).unwrap();
        }

        assert_eq!(cache.lru, vec![2, 1, 0]);
        assert!(cache.weights[4].is_none());
        assert!(cache.weights[3].is_none());
        assert!(cache.weights[2].is_some());
        assert!(cache.weights[1].is_some());
        assert!(cache.weights[0].is_some());
    }

    #[test]
    fn test_cache_random_access_no_crash() {
        let (gguf, cfg) = load_synthetic("random", 10, 16);
        let mut cache = LayerCache::new(cfg.clone(), 3);

        let pattern = [0, 5, 3, 7, 1, 9, 4, 2, 8, 6, 0, 5, 3];
        for &i in &pattern {
            cache.get(&gguf, i).unwrap();
        }
        assert!(cache.lru.len() <= 3);
        let hot_count = cache.weights.iter().filter(|w| w.is_some()).count();
        assert!(hot_count <= 3);
    }

    #[test]
    fn test_cache_single_layer() {
        let (gguf, cfg) = load_synthetic("single", 1, 32);
        let mut cache = LayerCache::new(cfg.clone(), 1);

        let layer = cache.get(&gguf, 0).unwrap();
        assert_eq!(layer.attn_norm.len(), 32);

        let layer = cache.get(&gguf, 0).unwrap();
        assert_eq!(layer.attn_norm.len(), 32);
    }

    #[test]
    fn test_cache_max_hot_one() {
        let (gguf, cfg) = load_synthetic("maxhot1", 5, 32);
        let mut cache = LayerCache::new(cfg.clone(), 1);

        cache.get(&gguf, 0).unwrap();
        assert!(cache.weights[0].is_some());

        cache.get(&gguf, 1).unwrap();
        assert!(cache.weights[0].is_none());
        assert!(cache.weights[1].is_some());

        cache.get(&gguf, 2).unwrap();
        assert!(cache.weights[1].is_none());
        assert!(cache.weights[2].is_some());
    }
}
