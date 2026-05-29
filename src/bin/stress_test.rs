use aether::inference::runner::LlamaRunner;
use aether::inference::runner::{estimate_model_bytes, sample};
use aether::inference::telemetry::LayerTelemetry;
use aether::Error;
use clap::Parser;
use std::collections::HashSet;
/// Stress test for Aether inference engine with real LLMs.
///
/// Tests normal and streaming inference paths on models larger than 1B
/// to catch OOM, correctness bugs, and performance regressions.
use std::time::Instant;

const SAMPLING_TEMPERATURE: f32 = 0.7;

#[derive(Parser)]
#[command(name = "stress-test", about = "Aether stress test for real LLMs")]
struct Cli {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: String,

    /// Prompt for inference
    #[arg(short, long, default_value = "The future of AI is")]
    prompt: String,

    /// Max tokens to generate per run
    #[arg(short = 'n', long, default_value = "50")]
    max_tokens: usize,

    /// Values for max_hot (streaming cache size). Comma-separated.
    /// Use "0" for normal (in-memory) mode.
    #[arg(long, default_value = "0,32,16,4,2")]
    scenarios: String,
}

struct ScenarioResult {
    label: String,
    load_time_s: f64,
    model_mb: f64,
    kv_mb: f64,
    peak_mem_mb: f64,
    prefill_tokens: usize,
    ttft_ms: f64,
    prefill_tok_s: f64,
    decode_tokens: usize,
    decode_tok_s: f64,
    output_text: String,
    logit_min: f32,
    logit_max: f32,
    first_token: String,
    prefill_logits: Vec<f32>,
    errors: Vec<String>,
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum();
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot as f64) / ((na as f64).sqrt() * (nb as f64).sqrt())
}

fn run_scenario(
    model_path: &str,
    prompt: &str,
    max_tokens: usize,
    max_hot: usize,
) -> ScenarioResult {
    let label = if max_hot == 0 {
        "in-memory".to_string()
    } else {
        format!("streaming(max_hot={})", max_hot)
    };

    let mut errors = Vec::new();
    let load_start = Instant::now();

    let runner_result = if max_hot == 0 {
        LlamaRunner::from_gguf(model_path)
    } else {
        LlamaRunner::from_gguf_streaming(model_path, max_hot)
    };

    let mut runner = match runner_result {
        Ok(r) => r,
        Err(e) => {
            errors.push(format!("Load failed: {}", e));
            return ScenarioResult {
                label,
                load_time_s: load_start.elapsed().as_secs_f64(),
                model_mb: 0.0,
                kv_mb: 0.0,
                peak_mem_mb: 0.0,
                prefill_tokens: 0,
                ttft_ms: 0.0,
                prefill_tok_s: 0.0,
                decode_tokens: 0,
                decode_tok_s: 0.0,
                output_text: String::new(),
                logit_min: 0.0,
                logit_max: 0.0,
                first_token: String::new(),
                prefill_logits: Vec::new(),
                errors,
            };
        }
    };

    let load_time_s = load_start.elapsed().as_secs_f64();

    let model_bytes = estimate_model_bytes(&runner.ctx.model);
    let kv_bytes = runner.kv.size_bytes();
    let model_mb = model_bytes as f64 / 1e6;
    let kv_mb = kv_bytes as f64 / 1e6;
    let peak_mem_mb = (model_bytes + kv_bytes) as f64 / 1e6;

    // Tokenize + prefill
    let token_ids = runner.tokenizer.encode(prompt, true);
    let prefill_tokens = token_ids.len();

    let ttft_start = Instant::now();
    let prefill_result = runner.prefill(&token_ids);
    let last_logits = match prefill_result {
        Ok(l) => l,
        Err(e) => {
            errors.push(format!("Prefill failed: {}", e));
            return ScenarioResult {
                label,
                load_time_s,
                model_mb,
                kv_mb,
                peak_mem_mb,
                prefill_tokens,
                ttft_ms: 0.0,
                prefill_tok_s: 0.0,
                decode_tokens: 0,
                decode_tok_s: 0.0,
                output_text: String::new(),
                logit_min: 0.0,
                logit_max: 0.0,
                first_token: String::new(),
                prefill_logits: Vec::new(),
                errors,
            };
        }
    };

    let ttft = ttft_start.elapsed();
    let ttft_ms = ttft.as_secs_f64() * 1000.0;
    let prefill_tok_s = prefill_tokens as f64 / ttft.as_secs_f64().max(1e-9);

    // Check logits for NaNs
    let logit_min = last_logits.iter().cloned().fold(f32::MAX, f32::min);
    let logit_max = last_logits.iter().cloned().fold(f32::MIN, f32::max);
    if logit_min.is_nan() || logit_max.is_nan() {
        errors.push("NaN detected in prefill logits!".to_string());
    }
    if logit_min.is_infinite() || logit_max.is_infinite() {
        errors.push("Infinity detected in prefill logits!".to_string());
    }

    let prefill_logits = last_logits.clone();

    // Sample first token (use temperature=0.7 for quality, not greedy)
    let next_token = sample(
        &last_logits,
        SAMPLING_TEMPERATURE,
        0.9,
        &token_ids.iter().copied().collect::<HashSet<u32>>(),
        1.0,
    );
    let first_token = runner.tokenizer.decode_one(next_token);

    // Decode loop
    let decode_start = Instant::now();
    let mut all_tokens = token_ids.clone();
    all_tokens.push(next_token);
    let mut prev_token = next_token;
    let mut decode_count = 0usize;

    for step in 0..max_tokens.saturating_sub(1) {
        if prev_token == runner.tokenizer.eos_id {
            break;
        }
        if runner.kv.seq_len >= runner.kv.max_seq {
            break;
        }

        let pos = prefill_tokens + step;
        let mut step_tel = vec![LayerTelemetry::default(); runner.ctx.model.config.num_layers];
        let logits = match runner.decode_step(prev_token, pos, &mut step_tel) {
            Ok(l) => l,
            Err(e) => {
                errors.push(format!("Decode step {} failed: {}", step, e));
                break;
            }
        };

        let next = sample(
            &logits,
            SAMPLING_TEMPERATURE,
            0.9,
            &all_tokens.iter().copied().collect::<HashSet<u32>>(),
            1.0,
        );
        if next == runner.tokenizer.eos_id {
            break;
        }
        all_tokens.push(next);
        prev_token = next;
        decode_count += 1;
    }

    let decode_time = decode_start.elapsed();
    let decode_tok_s = decode_count as f64 / decode_time.as_secs_f64().max(1e-9);

    let generated_ids = &all_tokens[prefill_tokens..];
    let output_text = runner.tokenizer.decode(generated_ids);

    ScenarioResult {
        label,
        load_time_s,
        model_mb,
        kv_mb,
        peak_mem_mb,
        prefill_tokens,
        ttft_ms,
        prefill_tok_s,
        decode_tokens: decode_count,
        decode_tok_s,
        output_text,
        logit_min,
        logit_max,
        first_token,
        prefill_logits,
        errors,
    }
}

fn print_table(results: &[ScenarioResult], prompt: &str, max_tokens: usize) {
    println!();
    println!("╔{}╗", "═".repeat(79));
    println!("║  Aether Stress Test Results                                  ║");
    println!("║  Prompt: {:<64} ║", prompt);
    println!(
        "║  Target tokens: {}                                           ║",
        max_tokens
    );
    println!("╠{}╣", "═".repeat(79));

    for r in results {
        if r.errors.is_empty() && r.prefill_tokens > 0 {
            println!("╔{}╗", "─".repeat(77));
            println!(
                "║  [{}]                                              ║",
                r.label
            );
            println!("╠{}╣", "─".repeat(77));
            println!(
                "║  Load time  : {:<8.2}s  Model: {:<8.0} MB  KV: {:<8.1} MB  Mem: {:<8.0} MB  ║",
                r.load_time_s, r.model_mb, r.kv_mb, r.peak_mem_mb
            );
            println!(
                "║  TTFT       : {:<8.1} ms  ({} prefill tokens @ {:.0} tok/s)             ║",
                r.ttft_ms, r.prefill_tokens, r.prefill_tok_s
            );
            println!(
                "║  Decode     : {:<5} tokens @ {:<8.1} tok/s                              ║",
                r.decode_tokens, r.decode_tok_s
            );
            println!(
                "║  First token: '{}'                                           ║",
                &r.first_token
            );
            println!(
                "║  Logits     : min={:.3}, max={:.3}                              ║",
                r.logit_min, r.logit_max
            );

            let output_preview: String = r.output_text.chars().take(60).collect();
            if !output_preview.is_empty() {
                println!("║  Output     : {:<60} ║", output_preview);
            }
            println!("╚{}╝", "─".repeat(77));
        } else {
            println!("╔{}╗", "─".repeat(77));
            println!("║  [{}]  FAILED                                ║", r.label);
            for e in &r.errors {
                println!("║    Error: {:<67} ║", e);
            }
            println!("╚{}╝", "─".repeat(77));
        }
        println!();
    }

    // Logit comparison between configurations
    let results_ok: Vec<&ScenarioResult> = results
        .iter()
        .filter(|r| r.errors.is_empty() && r.prefill_tokens > 0)
        .collect();
    if results_ok.len() >= 2 {
        println!("╔{}╗", "═".repeat(79));
        println!("║  Cosine Similarity Between Configurations                   ║");
        println!("╠{}╣", "═".repeat(79));
        for pair in results_ok.windows(2) {
            let a = &pair[0].prefill_logits;
            let b = &pair[1].prefill_logits;
            let sim = cosine_sim(a, b);
            let ok = if sim > 0.95 { "✓" } else { "✗" };
            println!(
                "║  {} {:<20} vs {:<20} = {:.8}  ║",
                ok, pair[0].label, pair[1].label, sim
            );
        }
        println!("╚{}╝", "═".repeat(79));
    }
}

fn main() -> Result<(), Error> {
    let cli = Cli::parse();
    println!("[stress] Hardware: Apple M2, 16 GB RAM");
    println!("[stress] Model: {}", cli.model);
    println!("[stress] Prompt: \"{}\"", cli.prompt);
    println!("[stress] Max tokens: {}", cli.max_tokens);
    println!();

    let scenarios: Vec<usize> = cli
        .scenarios
        .split(',')
        .map(|s| s.trim().parse().unwrap_or(0))
        .collect();

    let mut results = Vec::new();

    for (i, &max_hot) in scenarios.iter().enumerate() {
        println!(
            "[stress] Scenario {}/{}: {}",
            i + 1,
            scenarios.len(),
            if max_hot == 0 {
                "in-memory".to_string()
            } else {
                format!("streaming(max_hot={})", max_hot)
            }
        );

        let start = Instant::now();
        let result = run_scenario(&cli.model, &cli.prompt, cli.max_tokens, max_hot);
        let elapsed = start.elapsed();
        results.push(result);

        let last = results.last().expect("results is non-empty after push");
        if last.errors.is_empty() {
            println!(
                "[stress]   Done in {:.1}s — {} decode tokens, first='{}'",
                elapsed.as_secs_f64(),
                last.decode_tokens,
                last.first_token
            );
        } else {
            println!(
                "[stress]   FAILED in {:.1}s — {} error(s)",
                elapsed.as_secs_f64(),
                last.errors.len()
            );
            for e in &last.errors {
                println!("[stress]     {}", e);
            }
        }
        println!();
    }

    print_table(&results, &cli.prompt, cli.max_tokens);

    // Summary
    let passed = results
        .iter()
        .filter(|r| r.errors.is_empty() && r.prefill_tokens > 0)
        .count();
    let failed = results.len() - passed;
    println!();
    println!(
        "[stress] Results: {}/{} scenarios passed, {} failed",
        passed,
        results.len(),
        failed
    );

    if failed > 0 {
        eprintln!("[stress] WARNING: Some scenarios FAILED! See above for details.");
    }

    println!("[stress] Stress test complete.");
    Ok(())
}
