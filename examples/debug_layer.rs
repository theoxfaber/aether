use aether::loader::dequant::dequantize;
use aether::loader::gguf::{GGUFLoader, GGUFModel};
use aether::{Device, Graph, GraphTensor, Shape};
/// Debug: run just one layer and print intermediate values.
use std::collections::HashMap;

const D_MODEL: usize = 2048;
const N_HEADS: usize = 32;
const N_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const D_FF: usize = 5632;
const VOCAB_SIZE: usize = 32000;
const ROPE_BASE: f32 = 10000.0;
const MAX_SEQ: usize = 4;

type Dequantized = HashMap<String, Vec<f32>>;

fn load_and_dequant(model: &GGUFModel) -> Dequantized {
    let mut w = HashMap::new();
    for (name, tensor) in &model.tensors {
        let deq = dequantize(&tensor.data, tensor.dtype, &tensor.shape);
        w.insert(name.clone(), deq);
    }
    w
}

fn rope_cos_sin(seq_len: usize, head_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos_emb = vec![0.0; seq_len * head_dim];
    let mut sin_emb = vec![0.0; seq_len * head_dim];
    for pos in 0..seq_len {
        for i in 0..half {
            let theta = (pos as f32) * base.powf(-2.0 * (i as f32) / (head_dim as f32));
            let c = theta.cos();
            let s = theta.sin();
            let off = pos * head_dim;
            cos_emb[off + i] = c;
            cos_emb[off + i + half] = c;
            sin_emb[off + i] = -s;
            sin_emb[off + i + half] = s;
        }
    }
    (cos_emb, sin_emb)
}

fn apply_rope_2d(x: &GraphTensor, cos: &GraphTensor, sin: &GraphTensor) -> GraphTensor {
    let s = x.shape();
    let dims = s.dims();
    let d = dims[1];
    let half = d / 2;
    let first = x.slice(1, 0, half);
    let second = x.slice(1, half, d);
    let rotated = GraphTensor::concat(&[second, first], 1);
    x.mul(cos.clone()).add(rotated.mul(sin.clone()))
}

fn main() {
    let model = GGUFLoader::load("tinyllama-q4.gguf").unwrap();
    let w = load_and_dequant(&model);

    let (cos_host, sin_host) = rope_cos_sin(MAX_SEQ, HEAD_DIM, ROPE_BASE);
    let tokens = [1u32, 518, 25580, 29313];
    let seq_len = tokens.len();

    let graph = Graph::new();

    // Embedding: random matrix seed 42
    let mut state = 42u32;
    let mut emb_data = Vec::with_capacity(VOCAB_SIZE * D_MODEL);
    for _ in 0..(VOCAB_SIZE * D_MODEL) {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let r = (state as f32) / (u32::MAX as f32);
        emb_data.push(r * 0.02 - 0.01);
    }
    let emb_weight = graph.tensor(emb_data.clone(), Shape::new(vec![VOCAB_SIZE, D_MODEL]));

    let mut emb_parts = Vec::new();
    for &tok in &tokens {
        emb_parts.push(emb_weight.slice(0, tok as usize, (tok + 1) as usize));
    }
    let x_in = GraphTensor::concat(&emb_parts, 0).reshape(Shape::new(vec![1, seq_len, D_MODEL]));

    // Run embedding through
    let emb_result = x_in.clone().run(Device::Cpu).unwrap();
    let emb_val = emb_result.data();
    println!(
        "Input embedding norm: {:.6}, first 5: {:?}",
        emb_val.iter().map(|x| x * x).sum::<f32>().sqrt(),
        &emb_val[..5]
    );

    // Layer 0, step 1: RMSNorm
    let norm_w_host = &w["blk.0.attn_norm.weight"];
    // Trim last 32 garbage values and pad with 1.0
    let mut norm_w = norm_w_host[..2016].to_vec();
    norm_w.resize(2048, 1.0);
    let norm_w_tensor = graph.tensor(norm_w, Shape::new(vec![D_MODEL]));
    let h_norm = x_in.rmsnorm(norm_w_tensor, 1e-5);
    // Find outlier positions
    for (i, &v) in norm_w_host.iter().enumerate() {
        if v.abs() > 1.0 {
            println!("  outlier at [{}] = {}", i, v);
        }
    }
    let norm_result = h_norm.clone().run(Device::Cpu).unwrap();
    let norm_val = norm_result.data();
    let norm_nan = norm_val.iter().filter(|v| v.is_nan()).count();
    let norm_min = norm_val.iter().cloned().fold(f32::INFINITY, f32::min);
    let norm_max = norm_val.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "After RMSNorm: NaN={}, first 5: {:?}, min={:.6}, max={:.6}, norm={:.6}",
        norm_nan,
        &norm_val[..5],
        norm_min,
        norm_max,
        norm_val.iter().map(|x| x * x).sum::<f32>().sqrt()
    );
    if norm_nan > 0 {
        return;
    }

    // Step 2: Q projection
    let wq_host = &w["blk.0.attn_q.weight"];
    let wq_min = wq_host.iter().cloned().fold(f32::INFINITY, f32::min);
    let wq_max = wq_host.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let wq_mean = wq_host.iter().sum::<f32>() / wq_host.len() as f32;
    let wq_var = wq_host.iter().map(|v| (v - wq_mean).powi(2)).sum::<f32>() / wq_host.len() as f32;
    let wq_std = wq_var.sqrt();
    let wq_nan = wq_host.iter().filter(|v| v.is_nan()).count();
    println!(
        "Q weight: NaN={} min={:.4} max={:.4} mean={:.6} std={:.6}",
        wq_nan, wq_min, wq_max, wq_mean, wq_std
    );

    // Also check output.weight (also Q4_K)
    let wo_host = &w["output.weight"];
    let wo_min = wo_host.iter().cloned().fold(f32::INFINITY, f32::min);
    let wo_max = wo_host.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let wo_mean = wo_host.iter().sum::<f32>() / wo_host.len() as f32;
    let wo_var = wo_host.iter().map(|v| (v - wo_mean).powi(2)).sum::<f32>() / wo_host.len() as f32;
    let wo_std = wo_var.sqrt();
    let wo_nan = wo_host.iter().filter(|v| v.is_nan()).count();
    println!(
        "output.weight: NaN={} min={:.4} max={:.4} mean={:.6} std={:.6}",
        wo_nan, wo_min, wo_max, wo_mean, wo_std
    );
    println!("  first 20: {:?}", &wo_host[..20]);
    let wq = graph.tensor(
        w["blk.0.attn_q.weight"].clone(),
        Shape::new(vec![D_MODEL, D_MODEL]),
    );
    let q = h_norm
        .reshape(Shape::new(vec![seq_len, D_MODEL]))
        .matmul(wq)
        .reshape(Shape::new(vec![1, seq_len, D_MODEL]));
    let q_result = q.clone().run(Device::Cpu).unwrap();
    let q_val = q_result.data();
    let q_nan = q_val.iter().filter(|v| v.is_nan()).count();
    println!(
        "After Q proj: NaN={}, first 5: {:?}, norm={:.6}",
        q_nan,
        &q_val[..5],
        q_val.iter().map(|x| x * x).sum::<f32>().sqrt()
    );
    if q_nan > 0 {
        return;
    }

    // Step 3: RoPE on Q
    let cos_2d = graph.tensor(cos_host, Shape::new(vec![seq_len, HEAD_DIM]));
    let sin_2d = graph.tensor(sin_host, Shape::new(vec![seq_len, HEAD_DIM]));
    let mut cos_parts = Vec::new();
    let mut sin_parts = Vec::new();
    for _ in 0..N_HEADS {
        cos_parts.push(cos_2d.clone());
        sin_parts.push(sin_2d.clone());
    }
    let cos_q = GraphTensor::concat(&cos_parts, 0);
    let sin_q = GraphTensor::concat(&sin_parts, 0);
    let q_flat = q.reshape(Shape::new(vec![seq_len * N_HEADS, HEAD_DIM]));
    let q_rope = apply_rope_2d(&q_flat, &cos_q, &sin_q);
    let q_rope_r = q_rope.clone().run(Device::Cpu).unwrap();
    let q_rope_val = q_rope_r.data();
    let q_rope_nan = q_rope_val.iter().filter(|v| v.is_nan()).count();
    println!(
        "After RoPE on Q: NaN={}, first 5: {:?}, norm={:.6}",
        q_rope_nan,
        &q_rope_val[..5],
        q_rope_val.iter().map(|x| x * x).sum::<f32>().sqrt()
    );
    if q_rope_nan > 0 {
        return;
    }

    // Step 4: K projection
    let wk = graph.tensor(
        w["blk.0.attn_k.weight"].clone(),
        Shape::new(vec![D_MODEL, N_KV_HEADS * HEAD_DIM]),
    );
    let k = h_norm
        .reshape(Shape::new(vec![seq_len, D_MODEL]))
        .matmul(wk)
        .reshape(Shape::new(vec![1, seq_len, N_KV_HEADS * HEAD_DIM]));

    // Step 5: V projection
    let wv = graph.tensor(
        w["blk.0.attn_v.weight"].clone(),
        Shape::new(vec![D_MODEL, N_KV_HEADS * HEAD_DIM]),
    );
    let v = h_norm
        .reshape(Shape::new(vec![seq_len, D_MODEL]))
        .matmul(wv)
        .reshape(Shape::new(vec![1, seq_len, N_KV_HEADS * HEAD_DIM]));

    // Step 6: Reshape, GQA repeat, attention
    let q_attn = q_rope
        .reshape(Shape::new(vec![1, seq_len, N_HEADS, HEAD_DIM]))
        .reshape(Shape::new(vec![N_HEADS, seq_len, HEAD_DIM]));

    // RoPE on K
    let k_flat = k.reshape(Shape::new(vec![seq_len * N_KV_HEADS, HEAD_DIM]));
    let mut k_cos_parts = Vec::new();
    let mut k_sin_parts = Vec::new();
    for _ in 0..N_KV_HEADS {
        k_cos_parts.push(cos_2d.clone());
        k_sin_parts.push(sin_2d.clone());
    }
    let cos_k = GraphTensor::concat(&k_cos_parts, 0);
    let sin_k = GraphTensor::concat(&k_sin_parts, 0);
    let k_rope = apply_rope_2d(&k_flat, &cos_k, &sin_k);
    let k_reshaped = k_rope.reshape(Shape::new(vec![1, seq_len, N_KV_HEADS, HEAD_DIM]));

    let v_reshaped = v.reshape(Shape::new(vec![1, seq_len, N_KV_HEADS, HEAD_DIM]));

    let rep = N_HEADS / N_KV_HEADS;
    let k_exp = GraphTensor::concat(&vec![k_reshaped.clone(); rep], 2);
    let v_exp = GraphTensor::concat(&vec![v_reshaped.clone(); rep], 2);

    let k_attn = k_exp.reshape(Shape::new(vec![N_HEADS, seq_len, HEAD_DIM]));
    let v_attn = v_exp.reshape(Shape::new(vec![N_HEADS, seq_len, HEAD_DIM]));

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn_out = q_attn.flash_attention(k_attn, v_attn, scale, true);

    let attn_result = attn_out.clone().run(Device::Cpu).unwrap();
    let attn_val = attn_result.data();
    let attn_nan = attn_val.iter().filter(|v| v.is_nan()).count();
    println!(
        "After attention: NaN={}, first 5: {:?}, shape={:?}, norm={:.6}",
        attn_nan,
        &attn_val[..5],
        attn_result.shape(),
        attn_val.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    // Step 7: Output proj + residual
    let wo = graph.tensor(
        w["blk.0.attn_output.weight"].clone(),
        Shape::new(vec![D_MODEL, D_MODEL]),
    );
    let attn_out_2d = attn_out
        .reshape(Shape::new(vec![N_HEADS * seq_len, HEAD_DIM]))
        .matmul(wo)
        .reshape(Shape::new(vec![1, seq_len, D_MODEL]));
    let h_attn = x_in.add(attn_out_2d);
    let h_attn_r = h_attn.clone().run(Device::Cpu).unwrap();
    let h_attn_val = h_attn_r.data();
    let h_attn_nan = h_attn_val.iter().filter(|v| v.is_nan()).count();
    println!(
        "After output proj + residual: NaN={}, first 5: {:?}, norm={:.6}",
        h_attn_nan,
        &h_attn_val[..5],
        h_attn_val.iter().map(|x| x * x).sum::<f32>().sqrt()
    );
    if h_attn_nan > 0 {
        return;
    }

    // Step 8: SiLU FFN
    let mut ffn_norm_w = w["blk.0.ffn_norm.weight"].clone();
    ffn_norm_w.truncate(2016);
    ffn_norm_w.resize(2048, 1.0);
    let ffn_norm_tensor = graph.tensor(ffn_norm_w, Shape::new(vec![D_MODEL]));
    let h_ffn = h_attn.rmsnorm(ffn_norm_tensor, 1e-5);
    let w_gate = graph.tensor(
        w["blk.0.ffn_gate.weight"].clone(),
        Shape::new(vec![D_MODEL, D_FF]),
    );
    let w_up = graph.tensor(
        w["blk.0.ffn_up.weight"].clone(),
        Shape::new(vec![D_MODEL, D_FF]),
    );
    let w_down = graph.tensor(
        w["blk.0.ffn_down.weight"].clone(),
        Shape::new(vec![D_FF, D_MODEL]),
    );

    let gate = h_ffn
        .clone()
        .reshape(Shape::new(vec![seq_len, D_MODEL]))
        .matmul(w_gate);
    let up = h_ffn
        .clone()
        .reshape(Shape::new(vec![seq_len, D_MODEL]))
        .matmul(w_up);
    let silu = gate.clone().mul(gate.sigmoid());
    let ffn_out = silu
        .mul(up)
        .matmul(w_down)
        .reshape(Shape::new(vec![1, seq_len, D_MODEL]));
    let h_out = h_attn.add(ffn_out);

    let ffn_result = h_out.run(Device::Cpu).unwrap();
    let ffn_val = ffn_result.data();
    let ffn_nan = ffn_val.iter().filter(|v| v.is_nan()).count();
    println!(
        "After FFN: NaN={}, first 5: {:?}, norm={:.6}",
        ffn_nan,
        &ffn_val[..5],
        ffn_val.iter().map(|x| x * x).sum::<f32>().sqrt()
    );
}
