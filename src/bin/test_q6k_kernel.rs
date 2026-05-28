use aether::loader::dequant::dequantize;
/// Directly test the Q6_K matmul kernel vs a reference f32 implementation.
use aether::loader::gguf::{GGUFDtype, GGUFLoader};
use aether::quant::matmul::quantized_matmul_impl;
use aether::Error;

fn main() -> Result<(), Error> {
    // Load the raw GGUF
    let gguf = GGUFLoader::load("mistral-7b-q4k.gguf")?;
    let output_tensor = gguf.tensors.get("output.weight").expect("output.weight");
    let emb_tensor = gguf
        .tensors
        .get("token_embd.weight")
        .expect("token_embd.weight");
    let d_model = 4096usize;
    let vocab_size = emb_tensor.shape[1];
    eprintln!(
        "output.weight: shape={:?} dtype={:?}",
        output_tensor.shape, output_tensor.dtype
    );

    // Dequantize the lm_head to f32
    let lm_f32_raw = dequantize(
        &output_tensor.data,
        output_tensor.dtype,
        &output_tensor.shape,
    );
    // Rearrange from [d_model, vocab_size] to [vocab_size, d_model]
    let mut lm_f32 = vec![0.0f32; vocab_size * d_model];
    for d in 0..d_model {
        for v in 0..vocab_size {
            lm_f32[v * d_model + d] = lm_f32_raw[d * vocab_size + v];
        }
    }

    // Create a test input vector (random, but same for both paths)
    let test_input: Vec<f32> = (0..d_model).map(|i| (i as f32) / d_model as f32).collect();

    // Reference: dequantize the lm_head and compute f32 matmul
    let mut ref_out = vec![0.0f32; vocab_size];
    // c[j] = sum_i test_input[i] * lm_f32[j * d_model + i]
    // This is a straightforward dot product
    for j in 0..vocab_size {
        let mut sum = 0.0f32;
        let row_start = j * d_model;
        for i in 0..d_model {
            sum += test_input[i] * lm_f32[row_start + i];
        }
        ref_out[j] = sum;
    }

    // Now use the Q6_K quantized matmul on the SAME input
    // We need the QUANTIZED lm_head data, not the f32 version
    // Build a QuantWeight for the output.weight with the correct shape
    let lm_head_q = aether::inference::model_loader::QuantWeight {
        data: output_tensor.data.clone(),
        dtype: GGUFDtype::Q6_K,
        shape: vec![vocab_size, d_model], // reversed from raw shape
        f32_data: None,
    };

    let mut q6k_out = vec![0.0f32; vocab_size];
    quantized_matmul_impl(
        &test_input,
        1,
        &lm_head_q.data,
        &lm_head_q.shape,
        lm_head_q.dtype,
        &mut q6k_out,
        None,
    );

    // Compare
    let dot: f64 = ref_out
        .iter()
        .zip(q6k_out.iter())
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum();
    let norm_a: f64 = ref_out.iter().map(|&a| (a as f64) * (a as f64)).sum();
    let norm_b: f64 = q6k_out.iter().map(|&b| (b as f64) * (b as f64)).sum();
    let cos_sim = dot / (norm_a.sqrt() * norm_b.sqrt());
    let mse: f64 = ref_out
        .iter()
        .zip(q6k_out.iter())
        .map(|(&a, &b)| ((a - b) as f64) * ((a - b) as f64))
        .sum::<f64>()
        / vocab_size as f64;
    let max_diff: f64 = ref_out
        .iter()
        .zip(q6k_out.iter())
        .map(|(&a, &b)| (a - b).abs() as f64)
        .fold(0.0f64, f64::max);

    eprintln!("\n=== Direct Q6_K kernel vs reference f32 matmul ===");
    eprintln!("cos_sim={:.10}", cos_sim);
    eprintln!("mse={:.10}", mse);
    eprintln!("max_abs_diff={:.6}", max_diff);

    let mut ref_top: Vec<_> = ref_out.iter().enumerate().collect();
    ref_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut q6k_top: Vec<_> = q6k_out.iter().enumerate().collect();
    q6k_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!(
        "Ref top-5: {:?}",
        ref_top
            .iter()
            .take(5)
            .map(|(i, v)| (i, **v))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "Q6K top-5: {:?}",
        q6k_top
            .iter()
            .take(5)
            .map(|(i, v)| (i, **v))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "ref[0] = {:?} q6k[0] = {:?} (first 10 elements)",
        &ref_out[..10],
        &q6k_out[..10]
    );

    // Also test with a subset (first 10 columns) to check more easily
    eprintln!("\n--- Testing first 10 columns ---");
    let n_test = 10;
    let mut ref_sub = vec![0.0f32; n_test];
    for j in 0..n_test {
        let mut sum = 0.0f32;
        for i in 0..d_model {
            sum += test_input[i] * lm_f32[j * d_model + i];
        }
        ref_sub[j] = sum;
    }

    // For Q6_K, we need to use the layout as-is with n=vocab_size, so we can't easily
    // get just the first 10 columns. But we already have q6k_out.
    eprintln!("Ref first 10: {:?}", &ref_sub);
    eprintln!("Q6K first 10: {:?}", &q6k_out[..10]);

    Ok(())
}
