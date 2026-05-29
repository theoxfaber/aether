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

fn main() -> Result<(), Error> {
    let path = "tinyllama-q4.gguf";
    let prompt = "The capital of France is";

    let mut runner = LlamaRunner::from_gguf(path)?;
    runner.kv.reset();
    let mut token_ids = runner.tokenizer.encode(prompt, true);
    let prompt_len = token_ids.len();

    let mut last_logits = vec![0f32; runner.ctx.model.config.vocab_size];
    for (pos, &tok) in token_ids.iter().enumerate() {
        let mut dummy = vec![LayerTelemetry::default(); runner.ctx.model.config.num_layers];
        last_logits = runner.forward_one_hook(tok, pos, &mut dummy)?;
    }

    let mx = argmax(&last_logits) as u32;
    let mut top: Vec<_> = last_logits.iter().enumerate().collect();
    top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    eprintln!(
        "Prefill done. argmax={} ('{}')",
        mx,
        runner.tokenizer.decode_one(mx).replace('\n', "\\n")
    );
    eprintln!(
        "Top-5: {:?}",
        top.iter()
            .take(5)
            .map(|(i, v)| (i, **v))
            .collect::<Vec<_>>()
    );
    token_ids.push(mx);

    for step in 0..30 {
        let pos = prompt_len + step;
        let tok = token_ids[token_ids.len() - 1];
        let mut dummy = vec![LayerTelemetry::default(); runner.ctx.model.config.num_layers];
        last_logits = runner.forward_one_hook(tok, pos, &mut dummy)?;
        let mx = argmax(&last_logits) as u32;
        if mx == runner.tokenizer.eos_id {
            break;
        }
        token_ids.push(mx);
    }

    let output = runner.tokenizer.decode(&token_ids[prompt_len..]);
    eprintln!("Output: '{}'", output.replace('\n', "\\n"));
    Ok(())
}
