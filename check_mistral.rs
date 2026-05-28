fn main() {
    // Quick check: does Mistral 7B load?
    let mut runner = aether::inference::runner::LlamaRunner::from_gguf(
        "mistral-7b-q4k.gguf"
    ).expect("Failed to load Mistral 7B");

    println!("Model loaded successfully!");
    println!("Layers: {}, d_model: {}, vocab: {}",
        runner.model.config.num_layers,
        runner.model.config.d_model,
        runner.model.config.vocab_size);

    // Encode a prompt
    let prompt = "The future of AI is";
    let tokens = runner.tokenizer.encode(prompt, true);
    println!("Prompt tokens: {:?}", &tokens[..tokens.len().min(5)]);

    // Prefill
    let mut tel = vec![aether::inference::telemetry::LayerTelemetry::default();
                        runner.model.config.num_layers];
    let start = std::time::Instant::now();
    let logits = runner.prefill(&tokens).expect("Prefill failed");
    let elapsed = start.elapsed();
    println!("Prefill: {:.2}s, {} tokens", elapsed.as_secs_f32(), tokens.len());
    println!("logits[0] = {:.4}, max = {:.4}",
        logits[0],
        logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // Generate a few tokens
    let mut pos = tokens.len();
    let mut generated = Vec::new();
    for step in 0..3 {
        let tok = aether::inference::runner::sample(&logits, 0.0, 1.0, &[], 1.0);
        let decoded = runner.tokenizer.decode_one(tok);
        println!("Step {}: token {} -> '{}'", step, tok, decoded.replace('\n', "\\n"));

        let mut tel = vec![aether::inference::telemetry::LayerTelemetry::default();
                            runner.model.config.num_layers];
        let start = std::time::Instant::now();
        let logits_res = runner.decode_step(tok, pos, &mut tel);
        match logits_res {
            Ok(l) => { logits = l; }
            Err(e) => { eprintln!("Decode error: {e}"); break; }
        }
        let elapsed = start.elapsed();
        println!("  decode: {:.1}ms", elapsed.as_secs_f32() * 1000.0);
        pos += 1;
        generated.push(tok);
    }

    let text = runner.tokenizer.decode(&generated);
    println!("Generated: '{}'", text);
}
