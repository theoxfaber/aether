use aether::inference::runner::LlamaRunner;
use aether::inference::telemetry::LayerTelemetry;
use aether::Error;

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[allow(dead_code)]
fn topk(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut v: Vec<_> = logits.iter().enumerate().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v.iter().take(k).map(|(i, v)| (*i, **v)).collect()
}

fn main() -> Result<(), Error> {
    let path = "mistral-7b-q4k.gguf";
    let prompt = "The capital of France is";

    // ========== PATH 1: batch prefill + decode_step ==========
    eprintln!("=== PATH 1: batch prefill + decode_step ===");
    {
        let mut runner = LlamaRunner::from_gguf(path)?;
        runner.kv.reset();
        let mut token_ids = runner.tokenizer.encode(prompt, true);
        let prompt_len = token_ids.len();

        // batch prefill
        let mut last_logits = runner.prefill(&token_ids)?;
        let mx = argmax(&last_logits) as u32;
        eprintln!(
            "Sampled first: {} '{}'",
            mx,
            runner.tokenizer.decode_one(mx).replace('\n', "\\n")
        );
        token_ids.push(mx);

        for step in 0..30 {
            let pos = prompt_len + step;
            let tok = token_ids[token_ids.len() - 1];
            let mut dummy = vec![LayerTelemetry::default(); runner.model.config.num_layers];
            last_logits = runner.decode_step(tok, pos, &mut dummy)?;

            let mx = argmax(&last_logits) as u32;
            if mx == runner.tokenizer.eos_id {
                break;
            }
            token_ids.push(mx);
        }
        let output = runner.tokenizer.decode(&token_ids[prompt_len..]);
        eprintln!("Output: '{}'", output.replace('\n', "\\n"));
    }

    // ========== PATH 2: forward_one_hook for all (prefill + decode) ==========
    eprintln!("\n=== PATH 2: forward_one_hook for all (no batch prefill) ===");
    {
        let mut runner = LlamaRunner::from_gguf(path)?;
        runner.kv.reset();
        let mut token_ids = runner.tokenizer.encode(prompt, true);
        let prompt_len = token_ids.len();

        let mut last_logits = vec![0f32; runner.model.config.vocab_size];
        // Prefill token by token with forward_one_hook
        for (pos, &tok) in token_ids.iter().enumerate() {
            let mut dummy = vec![LayerTelemetry::default(); runner.model.config.num_layers];
            last_logits = runner.forward_one_hook(tok, pos, &mut dummy)?;
        }
        let mx = argmax(&last_logits) as u32;
        eprintln!(
            "Sampled first: {} '{}'",
            mx,
            runner.tokenizer.decode_one(mx).replace('\n', "\\n")
        );
        token_ids.push(mx);

        // Decode with forward_one_hook
        for step in 0..30 {
            let pos = prompt_len + step;
            let tok = token_ids[token_ids.len() - 1];
            let mut dummy = vec![LayerTelemetry::default(); runner.model.config.num_layers];
            last_logits = runner.forward_one_hook(tok, pos, &mut dummy)?;

            let mx = argmax(&last_logits) as u32;
            if mx == runner.tokenizer.eos_id {
                break;
            }
            token_ids.push(mx);
        }
        let output = runner.tokenizer.decode(&token_ids[prompt_len..]);
        eprintln!("Output: '{}'", output.replace('\n', "\\n"));
    }

    // ========== PATH 3: forward_one_hook for prefill, then decode_step for decode ==========
    eprintln!("\n=== PATH 3: fwd_one_hook prefill + decode_step decode ===");
    {
        let mut runner = LlamaRunner::from_gguf(path)?;
        runner.kv.reset();
        let mut token_ids = runner.tokenizer.encode(prompt, true);
        let prompt_len = token_ids.len();

        let mut last_logits = vec![0f32; runner.model.config.vocab_size];
        for (pos, &tok) in token_ids.iter().enumerate() {
            let mut dummy = vec![LayerTelemetry::default(); runner.model.config.num_layers];
            last_logits = runner.forward_one_hook(tok, pos, &mut dummy)?;
        }
        let mx = argmax(&last_logits) as u32;
        eprintln!(
            "Sampled first: {} '{}'",
            mx,
            runner.tokenizer.decode_one(mx).replace('\n', "\\n")
        );
        token_ids.push(mx);

        for step in 0..30 {
            let pos = prompt_len + step;
            let tok = token_ids[token_ids.len() - 1];
            let mut dummy = vec![LayerTelemetry::default(); runner.model.config.num_layers];
            last_logits = runner.decode_step(tok, pos, &mut dummy)?;

            let mx = argmax(&last_logits) as u32;
            if mx == runner.tokenizer.eos_id {
                break;
            }
            token_ids.push(mx);
        }
        let output = runner.tokenizer.decode(&token_ids[prompt_len..]);
        eprintln!("Output: '{}'", output.replace('\n', "\\n"));
    }

    println!("\n\n=== Summary ===");
    println!("PATH 1: batch prefill + decode_step");
    println!("PATH 2: forward_one_hook for all (single-token prefill + decode)");
    println!("PATH 3: forward_one_hook prefill + decode_step");
    println!("If PATH 1 != PATH 2, batch prefill corrupts KV cache.");
    println!("If PATH 2 != PATH 3, decode_step is different from forward_one_hook.");
    println!("If PATH 1 == PATH 3, the issue is in batch prefill's KV writes.");

    Ok(())
}
