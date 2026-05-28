use aether::inference::runner::{sample, LlamaRunner};
use aether::inference::telemetry::LayerTelemetry;
use aether::Error;
use std::time::Instant;

fn main() -> Result<(), Error> {
    // Force streaming mode to avoid GPU upload overhead
    let mut runner = LlamaRunner::from_gguf_streaming("mistral-7b-q4k.gguf", 2)?;
    eprintln!("Loaded in {:.1}s", Instant::now().elapsed().as_secs_f64());

    let prompt = "The capital of France is";
    let tokens = runner.tokenizer.encode(prompt, true);
    eprintln!("Prompt: {} tokens", tokens.len());

    let t1 = Instant::now();
    let logits = runner.prefill(&tokens)?;
    eprintln!("Prefill done in {:.1}s", t1.elapsed().as_secs_f64());

    let mut last_logits = logits;
    let mut next_tok = sample(&last_logits, 0.0, 1.0, &tokens, 1.0);
    eprint!("{}", runner.tokenizer.decode_one(next_tok));

    for step in 0..9 {
        let pos = tokens.len() + step;
        let t2 = Instant::now();
        let mut dummy = vec![LayerTelemetry::default(); runner.model.config.num_layers];
        last_logits = runner.forward_one_hook(next_tok, pos, &mut dummy)?;
        eprint!(" ({}s)", t2.elapsed().as_secs_f32());
        next_tok = sample(&last_logits, 0.0, 1.0, &[], 1.0);
        if next_tok == runner.tokenizer.eos_id {
            break;
        }
        eprint!("{}", runner.tokenizer.decode_one(next_tok));
    }
    eprintln!();
    Ok(())
}
