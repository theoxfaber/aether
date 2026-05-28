use aether::loader::dequant::dequantize;
use aether::loader::gguf::{GGUFLoader, GGUFModel};
use aether::{Device, Graph, GraphTensor, Shape};
use std::collections::HashMap;
/// TinyLlama-1.1B-Chat full inference pipeline.
///
/// Loads Q4_K_M weights via GGUF, dequantizes, builds the full 22-layer
/// decoder graph, runs a hardcoded 4-token sequence, and prints argmax token id.
use std::time::Instant;

const D_MODEL: usize = 2048;
const N_HEADS: usize = 32;
const N_KV_HEADS: usize = 4;
const HEAD_DIM: usize = D_MODEL / N_HEADS; // 64
const D_FF: usize = 5632;
const N_LAYERS: usize = 22;
const VOCAB_SIZE: usize = 32000;
const ROPE_BASE: f32 = 10000.0;
const MAX_SEQ: usize = 4;

type Dequantized = HashMap<String, Vec<f32>>;

fn load_and_dequant(_path: &str, model: &GGUFModel) -> Dequantized {
    let mut w = HashMap::new();
    let mut total = 0usize;
    for (name, tensor) in &model.tensors {
        let deq = dequantize(&tensor.data, tensor.dtype, &tensor.shape);
        total += deq.len();
        eprintln!(
            "  deq {}: {}x{} -> {} floats",
            name,
            tensor.shape[0],
            if tensor.shape.len() > 1 {
                tensor.shape[1]
            } else {
                1
            },
            deq.len()
        );
        w.insert(name.clone(), deq);
    }
    eprintln!(
        "  Total dequantized: {} floats ({:.2} GB)",
        total,
        total as f64 * 4.0 / 1e9
    );
    w
}

/// Precompute RoPE cos/sin using the split-half convention.
fn rope_cos_sin(seq_len: usize, head_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos_emb = vec![0.0f32; seq_len * head_dim];
    let mut sin_emb = vec![0.0f32; seq_len * head_dim];
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

/// Apply RoPE to a 2D tensor [N, head_dim]. cos, sin must have same shape.
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

/// Repeat a tensor along axis 0, `n` times.
fn repeat_axis0(t: &GraphTensor, n: usize) -> GraphTensor {
    let mut parts = Vec::with_capacity(n);
    for _ in 0..n {
        parts.push(t.clone());
    }
    GraphTensor::concat(&parts, 0)
}

/// Random embedding matrix with fixed seed.
fn random_embedding(_graph: &Graph, rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let r = (state as f32) / (u32::MAX as f32);
        data.push(r * 0.02 - 0.01);
    }
    data
}

/// 2D matmul with automatic flatten/reshape for 3D inputs.
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

fn main() {
    eprintln!("Loading GGUF...");
    let model =
        GGUFLoader::load("tinyllama-q4.gguf").expect("GGUF file not found. Download first.");
    eprintln!("Loaded {} tensors. Dequantizing...", model.tensors.len());
    let w = load_and_dequant("tinyllama-q4.gguf", &model);

    let (cos_emb_host, sin_emb_host) = rope_cos_sin(MAX_SEQ, HEAD_DIM, ROPE_BASE);

    let tokens = [1u32, 518, 25580, 29313];
    let seq_len = tokens.len();

    eprintln!("Building graph...");
    let graph = Graph::new();

    // ── Embedding ──
    let emb_data = random_embedding(&graph, VOCAB_SIZE, D_MODEL, 42);
    let emb_weight = graph.tensor(emb_data, Shape::new(vec![VOCAB_SIZE, D_MODEL]));

    let mut emb_parts = Vec::new();
    for &tok in &tokens {
        emb_parts.push(emb_weight.slice(0, tok as usize, (tok + 1) as usize));
    }
    let x_in = GraphTensor::concat(&emb_parts, 0).reshape(Shape::new(vec![1, seq_len, D_MODEL]));

    // ── RoPE tables: 2D [S, HEAD_DIM], will be expanded to match head count ──
    let cos_2d = graph.tensor(cos_emb_host, Shape::new(vec![seq_len, HEAD_DIM]));
    let sin_2d = graph.tensor(sin_emb_host, Shape::new(vec![seq_len, HEAD_DIM]));
    // Expanded for Q: [S*N_HEADS, HEAD_DIM]
    let cos_q_full = repeat_axis0(&cos_2d, N_HEADS);
    let sin_q_full = repeat_axis0(&sin_2d, N_HEADS);
    // Expanded for K: [S*N_KV_HEADS, HEAD_DIM]
    let cos_k_full = repeat_axis0(&cos_2d, N_KV_HEADS);
    let sin_k_full = repeat_axis0(&sin_2d, N_KV_HEADS);

    let mut h = x_in;

    for layer in 0..N_LAYERS {
        let norm_name = format!("blk.{}.attn_norm.weight", layer);
        let q_name = format!("blk.{}.attn_q.weight", layer);
        let k_name = format!("blk.{}.attn_k.weight", layer);
        let v_name = format!("blk.{}.attn_v.weight", layer);
        let o_name = format!("blk.{}.attn_output.weight", layer);
        let ffn_norm_name = format!("blk.{}.ffn_norm.weight", layer);
        let gate_name = format!("blk.{}.ffn_gate.weight", layer);
        let up_name = format!("blk.{}.ffn_up.weight", layer);
        let down_name = format!("blk.{}.ffn_down.weight", layer);

        // ── Pre-RMSNorm ──
        let norm_w = graph.tensor(w[&norm_name].clone(), Shape::new(vec![D_MODEL]));
        let h_norm = h.rmsnorm(norm_w, 1e-5);

        // ── QKV projections (2D matmul handles 3D flatten) ──
        let wq = graph.tensor(w[&q_name].clone(), Shape::new(vec![D_MODEL, D_MODEL]));
        let wk = graph.tensor(
            w[&k_name].clone(),
            Shape::new(vec![D_MODEL, N_KV_HEADS * HEAD_DIM]),
        );
        let wv = graph.tensor(
            w[&v_name].clone(),
            Shape::new(vec![D_MODEL, N_KV_HEADS * HEAD_DIM]),
        );

        let q = linear_2d(&h_norm, wq, D_MODEL); // [1, S, 2048]
        let k = linear_2d(&h_norm, wk, N_KV_HEADS * HEAD_DIM); // [1, S, 256]
        let v = linear_2d(&h_norm, wv, N_KV_HEADS * HEAD_DIM); // [1, S, 256]

        // Flatten to 2D for per-head RoPE: [B*S*H, HEAD_DIM]
        let q_flat = q.reshape(Shape::new(vec![seq_len * N_HEADS, HEAD_DIM])); // [128, 64]
        let k_flat = k.reshape(Shape::new(vec![seq_len * N_KV_HEADS, HEAD_DIM])); // [4, 64]

        // Apply RoPE on 2D (exact shapes, no broadcasting)
        let q_rope = apply_rope_2d(&q_flat, &cos_q_full, &sin_q_full); // [128, 64]
        let k_rope = apply_rope_2d(&k_flat, &cos_k_full, &sin_k_full); // [4, 64]

        // Reshape back to head structure: [1, S, H, HEAD_DIM]
        let q_reshaped = q_rope.reshape(Shape::new(vec![1, seq_len, N_HEADS, HEAD_DIM]));
        let k_reshaped = k_rope.reshape(Shape::new(vec![1, seq_len, N_KV_HEADS, HEAD_DIM]));
        let v_reshaped = v.reshape(Shape::new(vec![1, seq_len, N_KV_HEADS, HEAD_DIM]));

        // GQA: repeat K, V from N_KV_HEADS to N_HEADS
        let rep = N_HEADS / N_KV_HEADS;
        let k_exp = GraphTensor::concat(&vec![k_reshaped.clone(); rep], 2);
        let v_exp = GraphTensor::concat(&vec![v_reshaped.clone(); rep], 2);

        // Flatten heads into batch dim for flash_attention: [B*H, S, D]
        let q_attn = q_reshaped.reshape(Shape::new(vec![N_HEADS, seq_len, HEAD_DIM]));
        let k_attn = k_exp.reshape(Shape::new(vec![N_HEADS, seq_len, HEAD_DIM]));
        let v_attn = v_exp.reshape(Shape::new(vec![N_HEADS, seq_len, HEAD_DIM]));

        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let attn_out = q_attn
            .flash_attention(k_attn, v_attn, scale, true)
            .reshape(Shape::new(vec![1, seq_len, D_MODEL]));

        // Output projection and residual
        let wo = graph.tensor(w[&o_name].clone(), Shape::new(vec![D_MODEL, D_MODEL]));
        h = h.add(linear_2d(&attn_out, wo, D_MODEL));

        // ── FFN with SiLU gating ──
        let ffn_norm_w = graph.tensor(w[&ffn_norm_name].clone(), Shape::new(vec![D_MODEL]));
        let h_ffn = h.rmsnorm(ffn_norm_w, 1e-5);

        let w_gate = graph.tensor(w[&gate_name].clone(), Shape::new(vec![D_MODEL, D_FF]));
        let w_up = graph.tensor(w[&up_name].clone(), Shape::new(vec![D_MODEL, D_FF]));
        let w_down = graph.tensor(w[&down_name].clone(), Shape::new(vec![D_FF, D_MODEL]));

        let gate = linear_2d(&h_ffn, w_gate, D_FF);
        let up = linear_2d(&h_ffn, w_up, D_FF);
        let silu = gate.mul(gate.sigmoid());
        let ffn_out = linear_2d(&silu.mul(up), w_down, D_MODEL);

        h = h.add(ffn_out);

        if layer % 5 == 0 || layer == N_LAYERS - 1 {
            eprintln!("  Layer {} built", layer);
        }
    }

    // ── Final RMSNorm and LM head ──
    let out_norm_w = graph.tensor(w["output_norm.weight"].clone(), Shape::new(vec![D_MODEL]));
    let h_final = h.rmsnorm(out_norm_w, 1e-5);

    let out_w = graph.tensor(
        w["output.weight"].clone(),
        Shape::new(vec![D_MODEL, VOCAB_SIZE]),
    );
    let logits = linear_2d(&h_final, out_w, VOCAB_SIZE); // [1, S, V]

    // Last token logits
    let last = logits
        .slice(1, seq_len - 1, seq_len)
        .reshape(Shape::new(vec![VOCAB_SIZE]));

    eprintln!("Graph built. Executing...");
    let start = Instant::now();
    let result = last.run(Device::Cpu).unwrap();
    let elapsed = start.elapsed();

    let data = result.data();
    let nan_count = data.iter().filter(|v| v.is_nan()).count();
    let valid: Vec<f32> = data
        .iter()
        .copied()
        .filter(|v| !v.is_nan() && v.is_finite())
        .collect();
    let predicted = data
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i);

    println!();
    println!("=== TinyLlama-1.1B Inference ===");
    println!("  Tokens: {:?}", tokens);
    match predicted {
        Some(id) => println!("  Predicted next token id: {}", id),
        None => println!("  Predicted next token id: <all NaN>"),
    }
    println!("  Logits valid/finite: {}/{}", valid.len(), data.len());
    println!("  NaN count: {}", nan_count);
    if !valid.is_empty() {
        let mn = valid.iter().cloned().fold(f32::NAN, f32::min);
        let mx = valid.iter().cloned().fold(f32::NAN, f32::max);
        println!("  Valid range: [{:.4}, {:.4}]", mn, mx);
    }
    println!();
    println!("=== Performance ===");
    println!(
        "  Time:           {:8.3} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "  Peak GPU mem:   {:8.1} MB",
        graph.peak_gpu_bytes() as f64 / 1e6
    );
    println!("  Evictions:      {:8}", graph.eviction_count());
    println!("  Uploads:        {:8}", graph.upload_count());
}
