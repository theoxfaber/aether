use aether::loader::dequant::dequantize;
use aether::loader::gguf::GGUFLoader;
use aether::quant::quantized_matmul_impl;

fn main() {
    let gguf = GGUFLoader::load("tinyllama-q4.gguf").unwrap();
    let tensors = &gguf.tensors;

    // Check output.weight
    let output = tensors.get("output.weight").unwrap();
    let d_model = 2048usize;
    let vocab_size = 32000usize;

    eprintln!(
        "output.weight: shape={:?} dtype={:?}",
        output.shape, output.dtype
    );

    // The GGUF shape is [in_features, out_features] = [d_model, vocab_size]
    // load_quant reverses to [vocab_size, d_model]
    let shape_q = vec![vocab_size, d_model]; // what runner uses
    let shape_gguf = vec![d_model, vocab_size];

    // Create a test hidden state
    let hidden: Vec<f32> = (0..d_model)
        .map(|i| (i as f32) / d_model as f32 - 0.5)
        .collect();

    // Runner's quantized path
    let mut logits_q = vec![0.0f32; vocab_size];
    quantized_matmul_impl(
        &hidden,
        1,
        &output.data,
        &shape_q,
        output.dtype,
        &mut logits_q,
        None,
    );

    // F32 reference path: dequant with GGUF shape, then use matmul_f32_gguf
    let b_f32 = dequantize(&output.data, output.dtype, &shape_gguf);
    let mut logits_ref = vec![0.0f32; vocab_size];
    matmul_f32_gguf(&hidden, &b_f32, 1, vocab_size, d_model, &mut logits_ref);

    // Compare
    let mut max_diff = 0.0f32;
    for i in 0..vocab_size.min(100) {
        let diff = (logits_q[i] - logits_ref[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    eprintln!("LM head max_diff (first 100): {:.10}", max_diff);

    // Find argmax
    let argmax_q = logits_q
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let argmax_ref = logits_ref
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "argmax Q: {} val={}, argmax ref: {} val={}",
        argmax_q, logits_q[argmax_q], argmax_ref, logits_ref[argmax_ref]
    );

    // Top 10
    let mut top_q: Vec<_> = logits_q.iter().enumerate().collect();
    top_q.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    eprintln!(
        "Top 10 Q: {:?}",
        &top_q[..10]
            .iter()
            .map(|(i, v)| (i, **v))
            .collect::<Vec<_>>()
    );
}

fn matmul_f32_gguf(a: &[f32], b: &[f32], _m: usize, n: usize, k: usize, c: &mut [f32]) {
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
