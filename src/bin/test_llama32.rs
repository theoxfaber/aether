use aether::inference::runner::sample;
use aether::inference::runner::LlamaRunner;
use aether::inference::telemetry::LayerTelemetry;
use aether::Error;
use std::collections::HashSet;

fn main() -> Result<(), Error> {
    let path = "llama-3.2-1b-q5_k_m.gguf";
    let prompt = "The capital of France is";

    let mut runner = LlamaRunner::from_gguf(path)?;
    runner.kv.reset();
    let mut token_ids = runner.tokenizer.encode(prompt, true);
    let prompt_len = token_ids.len();
    eprintln!(
        "Prompt tokens ({}): {:?}",
        prompt_len,
        &token_ids[..prompt_len.min(5)]
    );

    let mut last_logits = vec![0f32; runner.ctx.model.config.vocab_size];
    for (pos, &tok) in token_ids.iter().enumerate() {
        let mut dummy = vec![LayerTelemetry::default(); runner.ctx.model.config.num_layers];
        last_logits = runner.forward_one_hook(tok, pos, &mut dummy)?;
    }

    let mut top: Vec<_> = last_logits.iter().enumerate().collect();
    top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    eprintln!(
        "After prefill top-5: {:?}",
        top.iter()
            .take(5)
            .map(|(i, v)| (i, **v))
            .collect::<Vec<_>>()
    );

    // greedy
    let mut next_tok = sample(
        &last_logits,
        0.0,
        1.0,
        &token_ids.iter().copied().collect::<HashSet<u32>>(),
        1.0,
    );
    eprintln!(
        "First token: {} '{}'",
        next_tok,
        runner.tokenizer.decode_one(next_tok).replace('\n', "\\n")
    );
    token_ids.push(next_tok);

    for step in 0..29 {
        let pos = prompt_len + step;
        let mut dummy = vec![LayerTelemetry::default(); runner.ctx.model.config.num_layers];
        last_logits = runner.forward_one_hook(next_tok, pos, &mut dummy)?;
        next_tok = sample(
            &last_logits,
            0.0,
            1.0,
            &token_ids.iter().copied().collect::<HashSet<u32>>(),
            1.0,
        );
        if next_tok == runner.tokenizer.eos_id {
            break;
        }
        token_ids.push(next_tok);
    }

    let output = runner.tokenizer.decode(&token_ids[prompt_len..]);
    eprintln!("Output (greedy): '{}'", output.replace('\n', "\\n"));

    // temp=0.8
    eprintln!("\n--- temp=0.8 ---");
    let mut runner2 = LlamaRunner::from_gguf(path)?;
    runner2.kv.reset();
    let mut token_ids2 = runner2.tokenizer.encode(prompt, true);
    let prompt_len2 = token_ids2.len();

    let mut last_logits2 = vec![0f32; runner2.ctx.model.config.vocab_size];
    for (pos, &tok) in token_ids2.iter().enumerate() {
        let mut dummy = vec![LayerTelemetry::default(); runner2.ctx.model.config.num_layers];
        last_logits2 = runner2.forward_one_hook(tok, pos, &mut dummy)?;
    }

    let mut next_tok2 = sample(
        &last_logits2,
        0.8,
        0.9,
        &token_ids2.iter().copied().collect::<HashSet<u32>>(),
        1.0,
    );
    eprint!("{}", runner2.tokenizer.decode_one(next_tok2));
    token_ids2.push(next_tok2);

    for step in 0..49 {
        let pos = prompt_len2 + step;
        let mut dummy = vec![LayerTelemetry::default(); runner2.ctx.model.config.num_layers];
        last_logits2 = runner2.forward_one_hook(next_tok2, pos, &mut dummy)?;
        next_tok2 = sample(
            &last_logits2,
            0.8,
            0.9,
            &token_ids2.iter().copied().collect::<HashSet<u32>>(),
            1.0,
        );
        if next_tok2 == runner2.tokenizer.eos_id {
            break;
        }
        eprint!("{}", runner2.tokenizer.decode_one(next_tok2));
        token_ids2.push(next_tok2);
    }
    eprintln!();

    Ok(())
}
