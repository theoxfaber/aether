#![allow(unused)]

use aether::loader::dequant::dequantize;
/// Debug: run one full layer with runner's quantized path vs f32 reference.
use aether::loader::gguf::GGUFLoader;
use aether::quant::quantized_matmul_impl;

fn main() {
    let model_path = "tinyllama-q4.gguf";
    let gguf = GGUFLoader::load(model_path).unwrap();

    // Read config from GGUF metadata
    let meta = &gguf.metadata;
    let d_model: usize = gguf_usize(meta, "llama.embedding_length").unwrap_or(2048);
    let n_heads: usize = gguf_usize(meta, "llama.attention.head_count").unwrap_or(32);
    let n_kv_heads: usize = gguf_usize(meta, "llama.attention.head_count_kv").unwrap_or(4);
    let head_dim = d_model / n_heads;
    let d_ff: usize = gguf_usize(meta, "llama.feed_forward_length").unwrap_or(5632);
    let eps: f32 = gguf_f32(meta, "llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-6);

    eprintln!(
        "d_model={} n_heads={} n_kv_heads={} head_dim={} d_ff={} eps={}",
        d_model, n_heads, n_kv_heads, head_dim, d_ff, eps
    );

    // Simple test input (random but deterministic)
    let test_hidden: Vec<f32> = (0..d_model)
        .map(|i| (i as f32) / d_model as f32 - 0.5)
        .collect();

    // Helper: get tensor by name
    let tensors = &gguf.tensors;
    let get = |name: &str| tensors.get(name).unwrap();

    // Get quantized weight tensors for layer 0
    let t_q = get("blk.0.attn_q.weight");
    let t_k = get("blk.0.attn_k.weight");
    let t_v = get("blk.0.attn_v.weight");
    let t_o = get("blk.0.attn_output.weight");
    let t_gate = get("blk.0.ffn_gate.weight");
    let t_up = get("blk.0.ffn_up.weight");
    let t_down = get("blk.0.ffn_down.weight");

    // Dequantize all weights to f32 reference
    let b_q = dequantize(&t_q.data, t_q.dtype, &t_q.shape);
    let b_k = dequantize(&t_k.data, t_k.dtype, &t_k.shape);
    let b_v = dequantize(&t_v.data, t_v.dtype, &t_v.shape);
    let b_o = dequantize(&t_o.data, t_o.dtype, &t_o.shape);
    let b_gate = dequantize(&t_gate.data, t_gate.dtype, &t_gate.shape);
    let b_up = dequantize(&t_up.data, t_up.dtype, &t_up.shape);
    let b_down = dequantize(&t_down.data, t_down.dtype, &t_down.shape);

    // Quantized weight shapes [out, in] (reversed from GGUF)
    let s_q = vec![d_model, d_model];
    let s_k = vec![n_kv_heads * head_dim, d_model];
    let s_v = vec![n_kv_heads * head_dim, d_model];
    let s_o = vec![d_model, d_model];
    let s_gate = vec![d_ff, d_model];
    let s_up = vec![d_ff, d_model];
    let s_down = vec![d_model, d_ff];

    // ── 1. RMSNorm ──
    let attn_norm = get("blk.0.attn_norm.weight");
    let norm_weight: Vec<f32> = dequantize(&attn_norm.data, attn_norm.dtype, &attn_norm.shape);
    let mut normed = vec![0.0f32; d_model];
    rmsnorm(&test_hidden, &norm_weight, eps, &mut normed);

    // ── 2. Quantized path ──
    let mut q = vec![0.0f32; d_model];
    let mut k = vec![0.0f32; n_kv_heads * head_dim];
    let mut v = vec![0.0f32; n_kv_heads * head_dim];
    let mut o = vec![0.0f32; d_model];

    quantized_matmul_impl(&normed, 1, &t_q.data, &s_q, t_q.dtype, &mut q, None);
    quantized_matmul_impl(&normed, 1, &t_k.data, &s_k, t_k.dtype, &mut k, None);
    quantized_matmul_impl(&normed, 1, &t_v.data, &s_v, t_v.dtype, &mut v, None);
    quantized_matmul_impl(&normed, 1, &t_o.data, &s_o, t_o.dtype, &mut o, None);

    eprintln!("\nquantized Q[0..5]: {:?}", &q[..5]);
    eprintln!("quantized K[0..5]: {:?}", &k[..5]);

    // ── 3. F32 reference path ──
    // RMSNorm already done (same as quantized path)
    let mut ref_q = vec![0.0f32; d_model];
    let mut ref_k = vec![0.0f32; n_kv_heads * head_dim];
    let mut ref_v = vec![0.0f32; n_kv_heads * head_dim];
    let mut ref_o = vec![0.0f32; d_model];

    // b_q is [d_model, d_model] (GGUF order), use matmul_f32_gguf formula
    matmul_f32_gguf(&normed, &b_q, 1, d_model, d_model, &mut ref_q);
    matmul_f32_gguf(&normed, &b_k, 1, n_kv_heads * head_dim, d_model, &mut ref_k);
    matmul_f32_gguf(&normed, &b_v, 1, n_kv_heads * head_dim, d_model, &mut ref_v);
    matmul_f32_gguf(&normed, &b_o, 1, d_model, d_model, &mut ref_o);

    eprintln!("\nref Q[0..5]: {:?}", &ref_q[..5]);
    eprintln!("ref K[0..5]: {:?}", &ref_k[..5]);

    eprintln!("\n=== Layer 0 projection comparisons ===");
    eprintln!("Q max_diff: {:.10}", max_diff(&q, &ref_q));
    eprintln!("K max_diff: {:.10}", max_diff(&k, &ref_k));
    eprintln!("V max_diff: {:.10}", max_diff(&v, &ref_v));
    eprintln!("O max_diff: {:.10}", max_diff(&o, &ref_o));

    // ── Now run the full QKV → attention → output pipeline ──
    // Just the projections, not a full layer, since attention depends on sequence length.
}

fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len() as f32;
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / n + eps).sqrt();
    for (i, (&xi, &wi)) in x.iter().zip(weight.iter()).enumerate() {
        out[i] = xi / rms * wi;
    }
}

/// matmul_f32_gguf: b is [k, n] = [in, out] GGUF order
/// Computes c[j] = Σᵢ a[i] · b[i·n + j]
fn matmul_f32_gguf(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, c: &mut [f32]) {
    c.fill(0.0);
    for i in 0..k {
        let s = a[i];
        if s != 0.0 {
            for j in 0..n {
                c[j] = s.mul_add(b[i * n + j], c[j]);
            }
        }
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn gguf_usize(
    meta: &std::collections::HashMap<String, aether::loader::gguf::GGUFValue>,
    key: &str,
) -> Option<usize> {
    use aether::loader::gguf::GGUFValue;
    match meta.get(key) {
        Some(GGUFValue::Uint32(v)) => Some(*v as usize),
        Some(GGUFValue::Uint64(v)) => Some(*v as usize),
        Some(GGUFValue::Int32(v)) => Some(*v as usize),
        Some(GGUFValue::Int64(v)) => Some(*v as usize),
        _ => None,
    }
}

fn gguf_f32(
    meta: &std::collections::HashMap<String, aether::loader::gguf::GGUFValue>,
    key: &str,
) -> Option<f32> {
    use aether::loader::gguf::GGUFValue;
    match meta.get(key) {
        Some(GGUFValue::Float32(v)) => Some(*v),
        Some(GGUFValue::Float64(v)) => Some(*v as f32),
        _ => None,
    }
}
