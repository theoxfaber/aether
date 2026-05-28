use aether::loader::dequant::dequantize;
use aether::loader::gguf::GGUFLoader;
use aether::quant::quantized_matmul_impl;

fn main() {
    let gguf = GGUFLoader::load("tinyllama-q4.gguf").expect("GGUF file");

    let weights = [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ];

    for name in &weights {
        let tensor = gguf.tensors.get(*name).unwrap();
        let gguf_shape = &tensor.shape;
        let k_in = gguf_shape[0]; // in_features (GGUF: [in, out])
        let n_out = gguf_shape[1]; // out_features

        let test_input: Vec<f32> = (0..k_in).map(|i| (i as f32) / k_in as f32).collect();

        let b_f32 = dequantize(&tensor.data, tensor.dtype, gguf_shape);

        let shape = vec![n_out, k_in]; // [out, in] = quantized_matmul convention

        let mut q_out = vec![0.0f32; n_out];
        quantized_matmul_impl(
            &test_input,
            1,
            &tensor.data,
            &shape,
            tensor.dtype,
            &mut q_out,
            None,
        );

        // Reference: y = x @ W (GGUF: W is [in, out]), c[j] = Σᵢ a[i] · W[i][j] = Σᵢ a[i] · flat[i·n + j]
        let mut ref_out = vec![0.0f32; n_out];
        for j in 0..n_out {
            let mut sum = 0.0f32;
            for i in 0..k_in {
                sum += test_input[i] * b_f32[i * n_out + j];
            }
            ref_out[j] = sum;
        }

        let mut max_diff = 0.0f32;
        let mut max_rel = 0.0f32;
        for i in 0..n_out {
            let diff = (q_out[i] - ref_out[i]).abs();
            let rel = diff / ref_out[i].abs().max(1e-10);
            if diff > max_diff {
                max_diff = diff;
            }
            if rel > max_rel {
                max_rel = rel;
            }
        }
        eprintln!(
            "{:<35} n={:<5} k={:<5} dtype={:<8}  max_diff={:.10}  max_rel={:.6}",
            name,
            n_out,
            k_in,
            format!("{:?}", tensor.dtype),
            max_diff,
            max_rel
        );
    }
}
