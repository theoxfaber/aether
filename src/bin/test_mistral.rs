use aether::inference::runner::{sample, LlamaRunner};
use aether::inference::telemetry::LayerTelemetry;
use aether::Error;
use std::collections::HashSet;

fn main() -> Result<(), Error> {
    let path = "mistral-7b-q4k.gguf";

    // Test different sampling strategies
    let configs = [
        ("greedy (temp=0)", 0.0, 1.0),
        ("temp=0.5", 0.5, 0.9),
        ("temp=0.8", 0.8, 0.9),
        ("temp=1.0", 1.0, 0.9),
    ];

    for &(label, temp, top_p) in &configs {
        eprintln!("\n=== {} ===", label);
        let mut runner = LlamaRunner::from_gguf(path)?;
        runner.kv.reset();
        let prompt = "The capital of France is";
        let mut token_ids = runner.tokenizer.encode(prompt, true);
        let prompt_len = token_ids.len();

        // forward_one_hook for all prompt tokens
        let mut last_logits = vec![0f32; runner.ctx.model.config.vocab_size];
        for (pos, &tok) in token_ids.iter().enumerate() {
            let mut dummy = vec![LayerTelemetry::default(); runner.ctx.model.config.num_layers];
            last_logits = runner.forward_one_hook(tok, pos, &mut dummy)?;
        }

        let prev_set: HashSet<u32> = token_ids.iter().copied().collect();
        let mut next_tok = sample(&last_logits, temp, top_p, &prev_set, 1.0);
        let decoded = runner.tokenizer.decode_one(next_tok);
        eprint!("{}", decoded);
        token_ids.push(next_tok);

        for step in 0..49 {
            let pos = prompt_len + step;
            let mut dummy = vec![LayerTelemetry::default(); runner.ctx.model.config.num_layers];
            last_logits = runner.forward_one_hook(next_tok, pos, &mut dummy)?;

            let prev_set: HashSet<u32> = token_ids.iter().copied().collect();
            next_tok = sample(&last_logits, temp, top_p, &prev_set, 1.0);
            if next_tok == runner.tokenizer.eos_id {
                break;
            }
            let decoded = runner.tokenizer.decode_one(next_tok);
            eprint!("{}", decoded);
            token_ids.push(next_tok);
        }
        eprintln!();
    }

    Ok(())
}
