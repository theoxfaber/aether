use aether::inference::runner::LlamaRunner;
use aether::Error;
use clap::Parser;
/// Dump logits from Aether for a given prompt token sequence.
/// Output: JSON lines - one per token with [vocab_size] logits.
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    model: String,
    #[arg(short, long)]
    prompt: String,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();
    let mut runner = LlamaRunner::from_gguf(&args.model)?;
    let token_ids = runner.tokenizer.encode(&args.prompt, true);
    eprintln!("Token IDs: {:?}", &token_ids);
    eprintln!("Num tokens: {}", token_ids.len());
    let _cfg_vocab = runner.ctx.model.config.vocab_size;

    // Prefill all tokens token-by-token, dumping logits for each
    for (pos, &tok) in token_ids.iter().enumerate() {
        let start = Instant::now();
        let mut dummy_tel = vec![
            aether::inference::telemetry::LayerTelemetry::default();
            runner.ctx.model.config.num_layers
        ];
        let logits = runner.forward_one_hook(tok, pos, &mut dummy_tel)?;
        let elapsed = start.elapsed().as_micros();
        // Emit JSON: {"pos": N, "token": N, "logits": [f32; vocab], "time_us": N}
        let logits_json: Vec<f64> = logits.iter().map(|&x| x as f64).collect();
        let obj = serde_json::json!({
            "pos": pos,
            "token": tok,
            "logits": logits_json,
            "time_us": elapsed,
        });
        println!("{}", serde_json::to_string(&obj).unwrap());
    }

    Ok(())
}
