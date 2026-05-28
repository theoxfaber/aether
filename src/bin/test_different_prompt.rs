use aether::inference::runner::LlamaRunner;
use aether::inference::telemetry::LayerTelemetry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompts = ["Hello", "The", "I am", "Once upon a time"];
    for prompt in &prompts {
        let mut runner = LlamaRunner::from_gguf("tinyllama-q4.gguf")?;
        let ids = runner.tokenizer.encode(prompt, true);
        let mut last_logits = vec![0.0f32; 32000];
        let mut tel = vec![LayerTelemetry::default(); 22];
        for (pos, &tok) in ids.iter().enumerate() {
            last_logits = runner.forward_one_hook(tok, pos, &mut tel)?;
        }
        let argmax = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let text = runner.tokenizer.decode_one(argmax as u32);
        eprintln!("prompt={:20} argmax={:6} ({})", prompt, argmax, text);
    }
    Ok(())
}
