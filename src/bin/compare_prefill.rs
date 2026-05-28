#![allow(clippy::needless_range_loop)]
use aether::inference::runner::LlamaRunner;
use aether::inference::telemetry::LayerTelemetry;
use aether::Error;

fn main() -> Result<(), Error> {
    let path = "mistral-7b-q4k.gguf";
    let prompt = &[1u32, 28705u32, 28714u32]; // BOS, 'a', 'b'

    // === Batched prefill ===
    let mut r1 = LlamaRunner::from_gguf(path)?;
    r1.kv.reset();
    let logits_batch = r1.prefill(prompt)?;
    let max_b = logits_batch
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!(
        "batched prefill: max_idx={} max_val={:.4}",
        max_b, logits_batch[max_b]
    );

    // === Token-by-token ===
    let mut r2 = LlamaRunner::from_gguf(path)?;
    r2.kv.reset();
    let mut last_logits = vec![0.0f32; logits_batch.len()];
    for pos in 0..prompt.len() {
        let mut tel = vec![LayerTelemetry::default(); r2.model.config.num_layers];
        last_logits = r2.forward_one_hook(prompt[pos], pos, &mut tel)?;
    }
    let max_s = last_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!(
        "token-by-token:  max_idx={} max_val={:.4}",
        max_s, last_logits[max_s]
    );

    // Cosine similarity
    let dot: f32 = logits_batch
        .iter()
        .zip(last_logits.iter())
        .map(|(a, b)| a * b)
        .sum();
    let na: f32 = logits_batch.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = last_logits.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos_sim = dot / (na * nb);
    let mse: f32 = logits_batch
        .iter()
        .zip(last_logits.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / logits_batch.len() as f32;
    println!("\ncosine similarity: {:.6}", cos_sim);
    println!("MSE: {:.10}", mse);
    if cos_sim > 0.99 {
        println!("PASS: cos_sim > 0.99 ✓");
    } else {
        println!("FAIL: cos_sim < 0.99 — batched prefill is BUGGY");
    }

    // Show top-5 diff
    println!("\nTop 5 diff (batched vs token-by-token):");
    let mut diffs: Vec<(usize, f32)> = logits_batch
        .iter()
        .zip(last_logits.iter())
        .enumerate()
        .map(|(i, (a, b))| (i, (a - b).abs()))
        .collect();
    diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (idx, diff) in diffs.iter().take(10) {
        println!(
            "  idx={:5} diff={:.4} (batch={:.4}, single={:.4})",
            idx, diff, logits_batch[*idx], last_logits[*idx]
        );
    }

    Ok(())
}
