use aether::loader::dequant::dequantize;
use aether::loader::gguf::GGUFLoader;
use aether::tokenizer::Tokenizer;
use aether::{Device, Graph, GraphTensor, Shape, Tensor};
use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;

const D_MODEL: usize = 2048;
const N_HEADS: usize = 32;
const N_KV_HEADS: usize = 4;
const HEAD_DIM: usize = D_MODEL / N_HEADS;
const D_FF: usize = 5632;
const N_LAYERS: usize = 22;
const VOCAB_SIZE: usize = 32000;
const ROPE_BASE: f32 = 10000.0;

type Dequantized = HashMap<String, Vec<f32>>;

fn load_and_dequant(model: &aether::loader::gguf::GGUFModel) -> Dequantized {
    let mut w = HashMap::new();
    for (name, tensor) in &model.tensors {
        let mut deq = dequantize(&tensor.data, tensor.dtype, &tensor.shape);
        let nan_cnt = deq.iter().filter(|x| x.is_nan()).count();
        if nan_cnt > 0 {
            println!("Weight {} contains {} NaNs", name, nan_cnt);
        }
        // Transpose all 2D tensors except token_embd.weight to match Aether's row-major expectation [in_features, out_features]
        if tensor.shape.len() == 2 && name != "token_embd.weight" {
            let cols = tensor.shape[0];
            let rows = tensor.shape[1];
            deq = transpose(&deq, rows, cols);
        }
        w.insert(name.clone(), deq);
    }
    w
}

fn rope_cos_sin(seq_len: usize, head_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos_emb = vec![0.0f32; seq_len * head_dim];
    let mut sin_emb = vec![0.0f32; seq_len * head_dim];
    for pos in 0..seq_len {
        for i in 0..half {
            let theta = (pos as f32) * base.powf(-2.0 * (i as f32) / (head_dim as f32));
            cos_emb[pos * head_dim + i] = theta.cos();
            cos_emb[pos * head_dim + i + half] = theta.cos();
            sin_emb[pos * head_dim + i] = -(theta.sin());
            sin_emb[pos * head_dim + i + half] = theta.sin();
        }
    }
    (cos_emb, sin_emb)
}

fn apply_rope_2d(x: &GraphTensor, cos: &GraphTensor, sin: &GraphTensor) -> GraphTensor {
    let shape = x.shape();
    let dims = shape.dims();
    let d = dims[1];
    let half = d / 2;
    let first = x.slice(1, 0, half);
    let second = x.slice(1, half, d);
    let rotated = GraphTensor::concat(&[second, first], 1);
    x.mul(cos.clone()).add(rotated.mul(sin.clone()))
}

fn repeat_axis0(t: &GraphTensor, n: usize) -> GraphTensor {
    let mut parts = Vec::with_capacity(n);
    for _ in 0..n {
        parts.push(t.clone());
    }
    GraphTensor::concat(&parts, 0)
}

fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = v[i * cols + j];
        }
    }
    out
}

fn linear_2d(x: &GraphTensor, w: GraphTensor, out_features: usize) -> GraphTensor {
    let shape = x.shape();
    if shape.ndim() == 3 {
        let b = shape.dims()[0];
        let s = shape.dims()[1];
        let d = shape.dims()[2];
        let x_flat = x.reshape(Shape::new(vec![b * s, d]));
        let out_flat = x_flat.matmul(w);
        out_flat.reshape(Shape::new(vec![b, s, out_features]))
    } else {
        x.matmul(w)
    }
}

// ── KV Cache ──────────────────────────────────────────────────────────────

struct KVCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    seq_len: usize,
}

impl KVCache {
    fn new(n_layers: usize) -> Self {
        Self {
            k: (0..n_layers).map(|_| Vec::new()).collect(),
            v: (0..n_layers).map(|_| Vec::new()).collect(),
            seq_len: 0,
        }
    }

    fn push(&mut self, layer: usize, k_new: &[f32], v_new: &[f32]) {
        self.k[layer].extend_from_slice(k_new);
        self.v[layer].extend_from_slice(v_new);
    }

    fn layer_data(&self, layer: usize) -> (&[f32], &[f32]) {
        (&self.k[layer], &self.v[layer])
    }
}

/// Pre-fill: build full sequence graph and return (logits_last_token, kv_cache_with_post_rope_data)
fn prefill_and_cache(tokens: &[u32], w: &Dequantized, emb_data: &[f32]) -> (Tensor, KVCache) {
    let seq_len = tokens.len();
    let graph = Graph::new();

    let emb_weight = graph.tensor(emb_data.to_vec(), Shape::new(vec![VOCAB_SIZE, D_MODEL]));

    let (cos_emb_host, sin_emb_host) = rope_cos_sin(seq_len, HEAD_DIM, ROPE_BASE);
    let cos_2d = graph.tensor(cos_emb_host, Shape::new(vec![seq_len, HEAD_DIM]));
    let sin_2d = graph.tensor(sin_emb_host, Shape::new(vec![seq_len, HEAD_DIM]));
    let cos_q = repeat_axis0(&cos_2d, N_HEADS);
    let sin_q = repeat_axis0(&sin_2d, N_HEADS);
    let cos_k = repeat_axis0(&cos_2d, N_KV_HEADS);
    let sin_k = repeat_axis0(&sin_2d, N_KV_HEADS);

    let mut emb_parts = Vec::new();
    for &tok in tokens {
        emb_parts.push(emb_weight.slice(0, tok as usize, (tok + 1) as usize));
    }
    let mut h = GraphTensor::concat(&emb_parts, 0).reshape(Shape::new(vec![1, seq_len, D_MODEL]));

    let mut cache = KVCache::new(N_LAYERS);

    // Build full graph, extract K/V at each layer as separate output nodes
    let mut kv_tensors: Vec<(GraphTensor, GraphTensor)> = Vec::new();

    for layer in 0..N_LAYERS {
        let norm_w = graph.tensor(
            w[&format!("blk.{}.attn_norm.weight", layer)].clone(),
            Shape::new(vec![D_MODEL]),
        );
        let h_norm = h.rmsnorm(norm_w, 1e-5);

        let wq = graph.tensor(
            w[&format!("blk.{}.attn_q.weight", layer)].clone(),
            Shape::new(vec![D_MODEL, D_MODEL]),
        );
        let wk = graph.tensor(
            w[&format!("blk.{}.attn_k.weight", layer)].clone(),
            Shape::new(vec![D_MODEL, N_KV_HEADS * HEAD_DIM]),
        );
        let wv = graph.tensor(
            w[&format!("blk.{}.attn_v.weight", layer)].clone(),
            Shape::new(vec![D_MODEL, N_KV_HEADS * HEAD_DIM]),
        );

        // QKV projections
        let q = linear_2d(&h_norm, wq, D_MODEL); // [1, S, 2048]
        let k = linear_2d(&h_norm, wk, N_KV_HEADS * HEAD_DIM); // [1, S, 256]
        let v = linear_2d(&h_norm, wv, N_KV_HEADS * HEAD_DIM); // [1, S, 256]

        // Flatten for per-head RoPE
        let q_flat = q.reshape(Shape::new(vec![seq_len * N_HEADS, HEAD_DIM]));
        let k_flat = k.reshape(Shape::new(vec![seq_len * N_KV_HEADS, HEAD_DIM]));

        // Apply RoPE
        let q_rope = apply_rope_2d(&q_flat, &cos_q, &sin_q);
        let k_rope = apply_rope_2d(&k_flat, &cos_k, &sin_k);

        // Save (k, v) as additional graph outputs for cache extraction
        // We store k_rope-shaped and v-shaped (before GQA repeat)
        let k_cache_shape = k_rope.reshape(Shape::new(vec![1, seq_len, N_KV_HEADS, HEAD_DIM]));
        let v_cache_shape = v.reshape(Shape::new(vec![1, seq_len, N_KV_HEADS, HEAD_DIM]));
        kv_tensors.push((k_cache_shape.clone(), v_cache_shape.clone()));

        // Reshape back to head structure
        let q_shaped = q_rope.reshape(Shape::new(vec![1, seq_len, N_HEADS, HEAD_DIM]));
        let k_shaped = k_rope.reshape(Shape::new(vec![1, seq_len, N_KV_HEADS, HEAD_DIM]));
        let v_shaped = v.reshape(Shape::new(vec![1, seq_len, N_KV_HEADS, HEAD_DIM]));

        // GQA repeat
        let rep = N_HEADS / N_KV_HEADS;
        let mut k_repeated = Vec::with_capacity(N_KV_HEADS * rep);
        let mut v_repeated = Vec::with_capacity(N_KV_HEADS * rep);
        for h in 0..N_KV_HEADS {
            let k_slice = k_shaped.slice(2, h, h + 1);
            let v_slice = v_shaped.slice(2, h, h + 1);
            for _ in 0..rep {
                k_repeated.push(k_slice.clone());
                v_repeated.push(v_slice.clone());
            }
        }
        let k_exp = GraphTensor::concat(&k_repeated, 2);
        let v_exp = GraphTensor::concat(&v_repeated, 2);

        // Flatten for flash_attention
        let q_attn = q_shaped.reshape(Shape::new(vec![N_HEADS, seq_len, HEAD_DIM]));
        let k_attn = k_exp.reshape(Shape::new(vec![N_HEADS, seq_len, HEAD_DIM]));
        let v_attn = v_exp.reshape(Shape::new(vec![N_HEADS, seq_len, HEAD_DIM]));

        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let attn_out = q_attn.flash_attention(k_attn, v_attn, scale, true);

        let mut attn_heads = Vec::with_capacity(N_HEADS);
        for h_idx in 0..N_HEADS {
            let head_out = attn_out
                .slice(0, h_idx, h_idx + 1)
                .reshape(Shape::new(vec![seq_len, HEAD_DIM]));
            attn_heads.push(head_out);
        }
        let attn_merged =
            GraphTensor::concat(&attn_heads, 1).reshape(Shape::new(vec![1, seq_len, D_MODEL]));

        let wo = graph.tensor(
            w[&format!("blk.{}.attn_output.weight", layer)].clone(),
            Shape::new(vec![D_MODEL, D_MODEL]),
        );
        let attn_proj = linear_2d(&attn_merged, wo, D_MODEL);

        h = h.add(attn_proj);

        // FFN
        let ffn_norm_w = graph.tensor(
            w[&format!("blk.{}.ffn_norm.weight", layer)].clone(),
            Shape::new(vec![D_MODEL]),
        );
        let h_ffn = h.rmsnorm(ffn_norm_w, 1e-5);

        let w_gate = graph.tensor(
            w[&format!("blk.{}.ffn_gate.weight", layer)].clone(),
            Shape::new(vec![D_MODEL, D_FF]),
        );
        let w_up = graph.tensor(
            w[&format!("blk.{}.ffn_up.weight", layer)].clone(),
            Shape::new(vec![D_MODEL, D_FF]),
        );
        let w_down = graph.tensor(
            w[&format!("blk.{}.ffn_down.weight", layer)].clone(),
            Shape::new(vec![D_FF, D_MODEL]),
        );

        let gate = linear_2d(&h_ffn, w_gate, D_FF);
        let up = linear_2d(&h_ffn, w_up, D_FF);
        let silu = gate.mul(gate.sigmoid());
        let ffn_out = linear_2d(&silu.mul(up), w_down, D_MODEL);

        h = h.add(ffn_out);
    }

    // Final norm + output projection
    let out_norm_w = graph.tensor(w["output_norm.weight"].clone(), Shape::new(vec![D_MODEL]));
    let h_final = h.rmsnorm(out_norm_w, 1e-5);

    let out_w = graph.tensor(
        w["output.weight"].clone(),
        Shape::new(vec![D_MODEL, VOCAB_SIZE]),
    );
    let logits = linear_2d(&h_final, out_w, VOCAB_SIZE);
    let last = logits
        .slice(1, seq_len - 1, seq_len)
        .reshape(Shape::new(vec![VOCAB_SIZE]));

    // Mega tensor construction: concatenate logits and all KV cache tensors
    let mut concat_parts = Vec::new();
    concat_parts.push(last.clone());

    let kv_len = seq_len * N_KV_HEADS * HEAD_DIM;
    for (k_tensor, v_tensor) in &kv_tensors {
        concat_parts.push(k_tensor.reshape(Shape::new(vec![kv_len])));
        concat_parts.push(v_tensor.reshape(Shape::new(vec![kv_len])));
    }

    let mega_tensor = GraphTensor::concat(&concat_parts, 0);
    let mega_result = mega_tensor.run(Device::Cpu).unwrap();
    let mega_data = mega_result.data();

    // Slicing mega_data to reconstruct outputs
    let mut offset = 0;
    let last_data = mega_data[offset..offset + VOCAB_SIZE].to_vec();
    offset += VOCAB_SIZE;

    for layer in 0..N_LAYERS {
        let k_data = mega_data[offset..offset + kv_len].to_vec();
        offset += kv_len;
        let v_data = mega_data[offset..offset + kv_len].to_vec();
        offset += kv_len;

        cache.k[layer] = k_data;
        cache.v[layer] = v_data;
    }
    cache.seq_len = seq_len;

    let logit_result = Tensor::new(last_data, Shape::new(vec![VOCAB_SIZE]));
    let logit_data = logit_result.data();
    println!("logits len: {}", logit_data.len());
    let first5: Vec<f32> = logit_data.iter().take(5).copied().collect();
    println!("logits[0..5]: {:?}", first5);
    let max_val = logit_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let max_idx = logit_data.iter().position(|&x| x == max_val).unwrap_or(0);
    println!("max: {} at index {}", max_val, max_idx);
    let min_val = logit_data.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("min: {}", min_val);
    let nan_count = logit_data.iter().filter(|x| x.is_nan()).count();
    println!("NaN count: {}", nan_count);
    let zero_count = logit_data.iter().filter(|&x| *x == 0.0).count();
    println!("zero count: {}", zero_count);
    let val0 = logit_data[0];
    let const_count = logit_data
        .iter()
        .filter(|&x| {
            if val0.is_nan() {
                x.is_nan()
            } else {
                *x == val0
            }
        })
        .count();
    println!("constant count: {}", const_count);

    (logit_result, cache)
}

/// Decode: single-token forward pass using KV cache
fn decode_with_cache(
    token: u32,
    cache: &KVCache,
    w: &Dequantized,
    emb_data: &[f32],
) -> (Tensor, KVCache) {
    let graph = Graph::new();
    let new_pos = cache.seq_len; // position of this new token

    let emb_weight = graph.tensor(emb_data.to_vec(), Shape::new(vec![VOCAB_SIZE, D_MODEL]));

    // RoPE tables for the single new position
    let (cos_emb_host, sin_emb_host) = rope_cos_sin(new_pos + 1, HEAD_DIM, ROPE_BASE);
    // Extract just the last position's cos/sin for the new token
    let cos_new = graph.tensor(
        cos_emb_host[(new_pos * HEAD_DIM)..].to_vec(),
        Shape::new(vec![1, HEAD_DIM]),
    );
    let sin_new = graph.tensor(
        sin_emb_host[(new_pos * HEAD_DIM)..].to_vec(),
        Shape::new(vec![1, HEAD_DIM]),
    );
    // Expanded for Q: [N_HEADS, 1, HEAD_DIM] after repeat_axis0 + reshape
    let cos_q_single = repeat_axis0(&cos_new, N_HEADS);
    let sin_q_single = repeat_axis0(&sin_new, N_HEADS);
    // Expanded for K: [N_KV_HEADS, 1, HEAD_DIM]
    let cos_k_single = repeat_axis0(&cos_new, N_KV_HEADS);
    let sin_k_single = repeat_axis0(&sin_new, N_KV_HEADS);

    // Embed single token
    let x_emb = emb_weight.slice(0, token as usize, (token + 1) as usize); // [1, D_MODEL]
    let mut h = x_emb.reshape(Shape::new(vec![1, 1, D_MODEL])); // [1, 1, D_MODEL]

    let mut new_cache = KVCache::new(N_LAYERS);
    new_cache.seq_len = cache.seq_len + 1;
    // Copy all existing cached data into new cache
    for layer in 0..N_LAYERS {
        let (ck, cv) = cache.layer_data(layer);
        new_cache.k[layer] = ck.to_vec();
        new_cache.v[layer] = cv.to_vec();
    }

    let mut new_kv_tensors = Vec::with_capacity(N_LAYERS);

    for layer in 0..N_LAYERS {
        // RMSNorm
        let norm_w = graph.tensor(
            w[&format!("blk.{}.attn_norm.weight", layer)].clone(),
            Shape::new(vec![D_MODEL]),
        );
        let h_norm = h.rmsnorm(norm_w, 1e-5);

        // QKV projections for single position
        let wq = graph.tensor(
            w[&format!("blk.{}.attn_q.weight", layer)].clone(),
            Shape::new(vec![D_MODEL, D_MODEL]),
        );
        let wk = graph.tensor(
            w[&format!("blk.{}.attn_k.weight", layer)].clone(),
            Shape::new(vec![D_MODEL, N_KV_HEADS * HEAD_DIM]),
        );
        let wv = graph.tensor(
            w[&format!("blk.{}.attn_v.weight", layer)].clone(),
            Shape::new(vec![D_MODEL, N_KV_HEADS * HEAD_DIM]),
        );

        let q = linear_2d(&h_norm, wq, D_MODEL); // [1, 1, 2048]
        let k_new = linear_2d(&h_norm, wk, N_KV_HEADS * HEAD_DIM); // [1, 1, 256]
        let v_new = linear_2d(&h_norm, wv, N_KV_HEADS * HEAD_DIM); // [1, 1, 256]

        // Apply RoPE to single-position K (flatten to [N_KV_HEADS, HEAD_DIM])
        let k_new_flat = k_new.reshape(Shape::new(vec![N_KV_HEADS, HEAD_DIM]));
        let q_new_flat = q.reshape(Shape::new(vec![N_HEADS, HEAD_DIM]));

        let q_rope = apply_rope_2d(&q_new_flat, &cos_q_single, &sin_q_single);
        let k_new_rope = apply_rope_2d(&k_new_flat, &cos_k_single, &sin_k_single);

        // Reshape to head structure for concat
        let q_shaped = q_rope.reshape(Shape::new(vec![1, 1, N_HEADS, HEAD_DIM]));
        let k_new_shaped = k_new_rope.reshape(Shape::new(vec![1, 1, N_KV_HEADS, HEAD_DIM]));
        let v_new_shaped = v_new.reshape(Shape::new(vec![1, 1, N_KV_HEADS, HEAD_DIM]));

        // Save for mega-tensor concat later
        new_kv_tensors.push((k_new_shaped.clone(), v_new_shaped.clone()));

        // Load cached K, V as graph tensors
        let (cached_k_data, cached_v_data) = cache.layer_data(layer);
        let cached_seq = cache.seq_len;

        let cached_k = graph.tensor(
            cached_k_data.to_vec(),
            Shape::new(vec![1, cached_seq, N_KV_HEADS, HEAD_DIM]),
        );
        let cached_v = graph.tensor(
            cached_v_data.to_vec(),
            Shape::new(vec![1, cached_seq, N_KV_HEADS, HEAD_DIM]),
        );

        // Concat cached + new along seq dim
        let k_full = GraphTensor::concat(&[cached_k, k_new_shaped], 1);
        let v_full = GraphTensor::concat(&[cached_v, v_new_shaped], 1);

        // GQA repeat
        let rep = N_HEADS / N_KV_HEADS;
        let mut k_repeated = Vec::with_capacity(N_KV_HEADS * rep);
        let mut v_repeated = Vec::with_capacity(N_KV_HEADS * rep);
        for h in 0..N_KV_HEADS {
            let k_slice = k_full.slice(2, h, h + 1);
            let v_slice = v_full.slice(2, h, h + 1);
            for _ in 0..rep {
                k_repeated.push(k_slice.clone());
                v_repeated.push(v_slice.clone());
            }
        }
        let k_exp = GraphTensor::concat(&k_repeated, 2);
        let v_exp = GraphTensor::concat(&v_repeated, 2);

        let full_seq = cached_seq + 1;
        let q_attn = q_shaped.reshape(Shape::new(vec![N_HEADS, 1, HEAD_DIM]));
        let k_attn = k_exp.reshape(Shape::new(vec![N_HEADS, full_seq, HEAD_DIM]));
        let v_attn = v_exp.reshape(Shape::new(vec![N_HEADS, full_seq, HEAD_DIM]));

        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let attn_out = q_attn.flash_attention(k_attn, v_attn, scale, true);

        let mut attn_heads = Vec::with_capacity(N_HEADS);
        for h_idx in 0..N_HEADS {
            let head_out = attn_out
                .slice(0, h_idx, h_idx + 1)
                .reshape(Shape::new(vec![1, HEAD_DIM]));
            attn_heads.push(head_out);
        }
        let attn_merged =
            GraphTensor::concat(&attn_heads, 1).reshape(Shape::new(vec![1, 1, D_MODEL]));

        let wo = graph.tensor(
            w[&format!("blk.{}.attn_output.weight", layer)].clone(),
            Shape::new(vec![D_MODEL, D_MODEL]),
        );
        h = h.add(linear_2d(&attn_merged, wo, D_MODEL));

        // FFN
        let ffn_norm_w = graph.tensor(
            w[&format!("blk.{}.ffn_norm.weight", layer)].clone(),
            Shape::new(vec![D_MODEL]),
        );
        let h_ffn = h.rmsnorm(ffn_norm_w, 1e-5);

        let w_gate = graph.tensor(
            w[&format!("blk.{}.ffn_gate.weight", layer)].clone(),
            Shape::new(vec![D_MODEL, D_FF]),
        );
        let w_up = graph.tensor(
            w[&format!("blk.{}.ffn_up.weight", layer)].clone(),
            Shape::new(vec![D_MODEL, D_FF]),
        );
        let w_down = graph.tensor(
            w[&format!("blk.{}.ffn_down.weight", layer)].clone(),
            Shape::new(vec![D_FF, D_MODEL]),
        );

        let gate = linear_2d(&h_ffn, w_gate, D_FF);
        let up = linear_2d(&h_ffn, w_up, D_FF);
        let silu = gate.mul(gate.sigmoid());
        let ffn_out = linear_2d(&silu.mul(up), w_down, D_MODEL);
        h = h.add(ffn_out);
    }

    // Final norm + output projection
    let out_norm_w = graph.tensor(w["output_norm.weight"].clone(), Shape::new(vec![D_MODEL]));
    let h_final = h.rmsnorm(out_norm_w, 1e-5);

    let out_w = graph.tensor(
        w["output.weight"].clone(),
        Shape::new(vec![D_MODEL, VOCAB_SIZE]),
    );
    let logits = linear_2d(&h_final, out_w, VOCAB_SIZE).reshape(Shape::new(vec![VOCAB_SIZE]));

    // Mega tensor construction: concatenate logits and all new KV cache tensors
    let mut concat_parts = Vec::new();
    concat_parts.push(logits.clone());

    let kv_len = N_KV_HEADS * HEAD_DIM;
    for (k_tensor, v_tensor) in &new_kv_tensors {
        concat_parts.push(k_tensor.reshape(Shape::new(vec![kv_len])));
        concat_parts.push(v_tensor.reshape(Shape::new(vec![kv_len])));
    }

    let mega_tensor = GraphTensor::concat(&concat_parts, 0);
    let mega_result = mega_tensor.run(Device::Cpu).unwrap();
    let mega_data = mega_result.data();

    // Slicing mega_data to reconstruct outputs
    let mut offset = 0;
    let logits_data = mega_data[offset..offset + VOCAB_SIZE].to_vec();
    offset += VOCAB_SIZE;

    for layer in 0..N_LAYERS {
        let k_data = &mega_data[offset..offset + kv_len];
        offset += kv_len;
        let v_data = &mega_data[offset..offset + kv_len];
        offset += kv_len;

        new_cache.push(layer, k_data, v_data);
    }

    let result = Tensor::new(logits_data, Shape::new(vec![VOCAB_SIZE]));
    let logit_data = result.data();
    let first5: Vec<f32> = logit_data.iter().take(5).copied().collect();
    let max_val = logit_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let max_idx = logit_data
        .iter()
        .position(|&x| x == max_val)
        .unwrap_or(99999);
    let min_val = logit_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let nan_count = logit_data.iter().filter(|x| x.is_nan()).count();
    eprintln!(
        "  [decode logits] first5: {:?} max={:.6} at idx={} min={:.6} NaN={}",
        first5, max_val, max_idx, min_val, nan_count
    );
    (result, new_cache)
}

// ── Sampling ──────────────────────────────────────────────────────────────

fn sample(logits: &[f32], temperature: f32, top_k: usize, top_p: f32, rng: &mut impl Rng) -> u32 {
    let mut scores = logits.to_vec();

    // Remove NaN
    for s in &mut scores {
        if s.is_nan() || !s.is_finite() {
            *s = f32::NEG_INFINITY;
        }
    }

    // Temperature scaling
    if temperature > 0.0 && temperature != 1.0 {
        for s in &mut scores {
            *s /= temperature;
        }
    }

    // Top-k filtering
    if top_k > 0 && top_k < scores.len() {
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &i in &idx[top_k..] {
            scores[i] = f32::NEG_INFINITY;
        }
    }

    // Softmax
    let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scores.iter().map(|l| (l - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return 0;
    }
    for p in &mut probs {
        *p /= sum;
    }

    // Top-p (nucleus) filtering
    if top_p > 0.0 && top_p < 1.0 {
        let mut pairs: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumsum = 0.0;
        let mut keep = vec![false; probs.len()];
        for (i, p) in &pairs {
            if cumsum >= top_p {
                break;
            }
            keep[*i] = true;
            cumsum += p;
        }
        // If nothing kept, keep at least the top one
        if !keep.iter().any(|&x| x) {
            keep[pairs[0].0] = true;
        }
        let mut new_sum = 0.0;
        for (i, k) in keep.iter().enumerate() {
            if !*k {
                probs[i] = 0.0;
            } else {
                new_sum += probs[i];
            }
        }
        if new_sum > 0.0 {
            for p in &mut probs {
                *p /= new_sum;
            }
        }
    }

    // Sample
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

// ── Main ──────────────────────────────────────────────────────────────────

fn main() {
    eprintln!("=== TinyLlama-1.1B Text Generation ===");
    eprintln!("Loading model...");
    let model = GGUFLoader::load("tinyllama-q4.gguf").expect("GGUF file not found.");
    eprintln!("{} tensors. Dequantizing...", model.tensors.len());
    let w = load_and_dequant(&model);

    // Diagnostics for block 0 Q weight:
    println!(
        "blk.0.attn_q shape: {:?}",
        model.tensors["blk.0.attn_q.weight"].shape
    );
    println!(
        "blk.0.attn_q first 4 values: {:?}",
        &w["blk.0.attn_q.weight"][..4]
    );

    eprintln!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_gguf(&model).expect("Failed to load tokenizer");
    eprintln!("Vocabulary: {} tokens", tokenizer.vocab_size());

    let prompt = "The meaning of life is";
    eprintln!("Prompt: \"{}\"", prompt);

    let input_ids = tokenizer.encode(prompt, true);
    eprintln!("Tokenized: {} tokens", input_ids.len());

    // Use real token embeddings from the GGUF file
    let emb_raw = &w["token_embd.weight"];
    let emb_data = emb_raw.clone();

    // ── Pre-fill phase (with KV cache extraction) ──
    eprintln!();
    eprintln!("═══ Pre-fill ═══");
    let prefill_start = Instant::now();
    let (logits, cache) = prefill_and_cache(&input_ids, &w, &emb_data);
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let prompt_tok_s = input_ids.len() as f64 / (prefill_ms / 1000.0);
    eprintln!(
        "  {} tokens in {:.0} ms = {:.1} tok/s",
        input_ids.len(),
        prefill_ms,
        prompt_tok_s
    );

    let next_token = sample(logits.data(), 0.9, 40, 0.9, &mut rand::thread_rng());
    eprintln!(
        "  First token: {} \"{}\"",
        next_token,
        tokenizer.id_to_token_str(next_token).unwrap_or("?")
    );

    // ── Decode phase ──
    eprintln!();
    eprintln!("═══ Decode (KV-cached, single-token forward passes) ═══");

    let max_new_tokens = 32;
    let mut generated = input_ids.clone();
    generated.push(next_token);
    let mut cache = cache;

    let decode_start = Instant::now();
    let mut decode_times = Vec::new();

    for step in 0..max_new_tokens {
        let step_start = Instant::now();

        let (logits, new_cache) =
            decode_with_cache(generated.last().copied().unwrap(), &cache, &w, &emb_data);
        let next = sample(logits.data(), 0.7, 40, 0.9, &mut rand::thread_rng());

        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        decode_times.push(step_ms);

        generated.push(next);
        cache = new_cache;

        let token_str = tokenizer.id_to_token_str(next).unwrap_or("<?>");
        let token_str_clean = token_str.replace('\u{2581}', " ");
        if step < 5 || step % 5 == 4 || step == max_new_tokens - 1 {
            eprintln!(
                "  step {:3} [{:6.1} ms] {:6} \"{}\"",
                step + 1,
                step_ms,
                next,
                token_str_clean
            );
        }
    }

    let decode_total_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let avg_decode_ms = decode_times.iter().sum::<f64>() / decode_times.len() as f64;
    let decode_tok_s = max_new_tokens as f64 / (decode_total_ms / 1000.0);

    // ── Output ──
    let output = tokenizer.decode(&generated);
    println!();
    println!("═══ Generated Text ═══");
    println!("{}", output);
    println!();

    // ── Benchmarks ──
    println!("═══ Performance ═══");
    println!("{:<30} {:>12}", "Prompt tokens", input_ids.len());
    println!("{:<30} {:>12.0} ms", "Pre-fill time", prefill_ms);
    println!(
        "{:<30} {:>12.1} tok/s",
        "Prompt processing speed", prompt_tok_s
    );
    println!("{:<30} {:>12}", "Generated tokens", max_new_tokens);
    println!("{:<30} {:>12.1} ms", "Average decode step", avg_decode_ms);
    println!("{:<30} {:>12.1} tok/s", "Decode speed", decode_tok_s);
    println!("{:<30} {:>12.1} ms", "Total decode time", decode_total_ms);

    // Memory bandwidth estimate
    // Each decode step: read all model weights + write all activations
    let weight_bytes = N_LAYERS as f64
        * (
            (D_MODEL * D_MODEL * 4) as f64 +           // Q
        (D_MODEL * D_MODEL * 4) as f64 +           // O
        (D_MODEL * N_KV_HEADS * HEAD_DIM * 2 * 4) as f64 + // K, V
        (D_MODEL * D_FF * 2 * 4) as f64 +          // gate, up
        (D_FF * D_MODEL * 4) as f64 +              // down
        (D_MODEL * 4) as f64
            // norm weights
        );
    let kv_rw =
        2.0 * N_LAYERS as f64 * cache.seq_len as f64 * N_KV_HEADS as f64 * HEAD_DIM as f64 * 4.0;
    let bytes_per_step = weight_bytes + kv_rw;
    let bandwidth = bytes_per_step / (avg_decode_ms / 1000.0) / 1e9;
    println!("{:<30} {:>12.1} GB/s", "Est. memory bandwidth", bandwidth);
}
