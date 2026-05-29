use std::sync::Arc;

use aether::inference::model_loader::QuantWeight;
/// Compare Q6_K quantized matmul vs f32 dequantized matmul for lm_head.
use aether::inference::runner::LlamaRunner;
use aether::inference::telemetry::LayerTelemetry;
use aether::loader::dequant::dequantize;
use aether::loader::gguf::{GGUFLoader, SharedBytes};
use aether::Error;

fn main() -> Result<(), Error> {
    let gguf = GGUFLoader::load("mistral-7b-q4k.gguf")?;
    let output_tensor = gguf
        .tensors
        .get("output.weight")
        .expect("output.weight not found");
    let emb_tensor = gguf
        .tensors
        .get("token_embd.weight")
        .expect("token_embd.weight");
    let d_model = 4096usize;
    let vocab_size = emb_tensor.shape[1];
    eprintln!(
        "vocab_size={} output.weight: shape={:?} dtype={:?}",
        vocab_size, output_tensor.shape, output_tensor.dtype
    );

    let lm_f32_raw = dequantize(
        &output_tensor.data,
        output_tensor.dtype,
        &output_tensor.shape,
    );
    eprintln!("lm_f32_raw: {} floats", lm_f32_raw.len());

    // Rearrange from [d_model, vocab_size] to [vocab_size, d_model]
    let mut lm_f32 = vec![0.0f32; vocab_size * d_model];
    for d in 0..d_model {
        for v in 0..vocab_size {
            lm_f32[v * d_model + d] = lm_f32_raw[d * vocab_size + v];
        }
    }

    // Run with Q6_K lm_head
    let mut runner = LlamaRunner::from_gguf("mistral-7b-q4k.gguf")?;
    runner.kv.reset();
    let token_ids = runner.tokenizer.encode("The capital of France is", true);
    let mut last_logits_q6k = vec![0f32; vocab_size];
    for (pos, &tok) in token_ids.iter().enumerate() {
        let mut dummy = vec![LayerTelemetry::default(); runner.ctx.model.config.num_layers];
        last_logits_q6k = runner.forward_one_hook(tok, pos, &mut dummy)?;
    }
    drop(runner);

    // Run with F32 lm_head (dequantized weight, f32 matmul)
    // We rebuild the model so lm_head is F32 instead of Q6_K.
    let gguf = aether::loader::gguf::GGUFLoader::load("mistral-7b-q4k.gguf").unwrap();
    let config = aether::inference::model_loader::LlamaConfig::from_gguf(&gguf).unwrap();
    let mut model = aether::inference::model_loader::LlamaModel::from_gguf(&gguf, false).unwrap();
    model.lm_head = QuantWeight {
        data: SharedBytes::new_owned(bytemuck::cast_slice(&lm_f32).to_vec()),
        dtype: aether::loader::gguf::GGUFDtype::F32,
        shape: vec![vocab_size, d_model],
        f32_data: None,
    };
    let ctx = Arc::new(aether::inference::runner::InferenceContext {
        model: Arc::new(model),
        wgpu_backend: None,
        gpu_weights: vec![],
        rope_sin_gpu: None,
        rope_cos_gpu: None,
        gguf: None,
    });
    let la = aether::scheduler::memory_aware::LayerAssignment {
        layer_devices: vec![aether::Device::Cpu; config.num_layers],
        total_model_bytes: 0,
        gpu_budget: 0,
        gpu_layers: 0,
        cpu_layers: config.num_layers,
        total_ram: 0,
        memory_plan: aether::scheduler::memory_aware::MemoryPlan::InMemory {
            total_ram: 0,
            model_bytes: 0,
        },
    };
    let mut runner2 = LlamaRunner::new_with_context(
        ctx,
        &aether::tokenizer::Tokenizer::from_gguf(&gguf).unwrap(),
        &la,
    );
    runner2.kv.reset();
    let token_ids2 = runner2.tokenizer.encode("The capital of France is", true);
    let mut last_logits_f32 = vec![0f32; vocab_size];
    for (pos, &tok) in token_ids2.iter().enumerate() {
        let mut dummy = vec![LayerTelemetry::default(); runner2.ctx.model.config.num_layers];
        last_logits_f32 = runner2.forward_one_hook(tok, pos, &mut dummy)?;
    }

    // Compare
    let dot: f64 = last_logits_q6k
        .iter()
        .zip(last_logits_f32.iter())
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum();
    let norm_a: f64 = last_logits_q6k
        .iter()
        .map(|&a| (a as f64) * (a as f64))
        .sum();
    let norm_b: f64 = last_logits_f32
        .iter()
        .map(|&b| (b as f64) * (b as f64))
        .sum();
    let cos_sim = dot / (norm_a.sqrt() * norm_b.sqrt());
    let mse: f64 = last_logits_q6k
        .iter()
        .zip(last_logits_f32.iter())
        .map(|(&a, &b)| ((a - b) as f64) * ((a - b) as f64))
        .sum::<f64>()
        / vocab_size as f64;
    let max_diff: f64 = last_logits_q6k
        .iter()
        .zip(last_logits_f32.iter())
        .map(|(&a, &b)| (a - b).abs() as f64)
        .fold(0.0f64, f64::max);
    eprintln!("\n=== Q6_K vs F32 lm_head ===");
    eprintln!("cos_sim={:.10}", cos_sim);
    eprintln!("mse={:.10}", mse);
    eprintln!("max_abs_diff={:.6}", max_diff);

    let mut q6k_top: Vec<_> = last_logits_q6k.iter().enumerate().collect();
    q6k_top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    let mut f32_top: Vec<_> = last_logits_f32.iter().enumerate().collect();
    f32_top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    eprintln!(
        "Q6_K top-5: {:?}",
        q6k_top
            .iter()
            .take(5)
            .map(|(i, v)| (i, **v))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "F32 top-5:  {:?}",
        f32_top
            .iter()
            .take(5)
            .map(|(i, v)| (i, **v))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "Q6_K argmax: {} '{}'",
        q6k_top[0].0,
        runner2
            .tokenizer
            .decode_one(q6k_top[0].0 as u32)
            .replace('\n', "\\n")
    );
    eprintln!(
        "F32 argmax:  {} '{}'",
        f32_top[0].0,
        runner2
            .tokenizer
            .decode_one(f32_top[0].0 as u32)
            .replace('\n', "\\n")
    );

    Ok(())
}
