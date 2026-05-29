use clap::{Parser, Subcommand};
use std::collections::HashSet;
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use aether::inference::runner::LlamaRunner;
use aether::Error;

#[derive(Parser)]
#[command(name = "aether", version, about = "Aether LLM inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference with a prompt
    Run {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: String,
        /// Input prompt
        #[arg(short, long)]
        prompt: String,
        /// Maximum number of tokens to generate
        #[arg(short = 'n', long, default_value = "200")]
        max_tokens: usize,
        /// Sampling temperature (0 = greedy)
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,
        /// Top-p nucleus sampling threshold
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        /// Repetition penalty (>1.0 penalizes repeated tokens)
        #[arg(long, default_value = "1.0")]
        repetition_penalty: f32,
    },
    /// Benchmark model performance
    Bench {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: String,
        /// Benchmark prompt
        #[arg(short, long, default_value = "Once upon a time")]
        prompt: String,
        /// Number of tokens to generate
        #[arg(short = 'n', long, default_value = "100")]
        max_tokens: usize,
        /// Save results as JSON
        #[arg(long)]
        output_json: Option<String>,
        /// Compare with llama.cpp (requires llama-cli in PATH)
        #[arg(long)]
        compare_llama_cpp: bool,
        /// Path to llama-cli binary
        #[arg(long, default_value = "llama-cli")]
        llama_binary: String,
        /// Show memory-aware layer assignment plan
        #[arg(long)]
        memory_plan: bool,
    },
}

fn main() -> Result<(), Error> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            repetition_penalty,
        } => {
            let mut runner = LlamaRunner::from_gguf(&model)?;
            let output =
                runner.generate(&prompt, max_tokens, temperature, top_p, repetition_penalty)?;
            println!();
            println!("{}", output);
            runner.telemetry.print_summary();
        }
        Commands::Bench {
            model,
            prompt,
            max_tokens,
            output_json,
            compare_llama_cpp,
            llama_binary,
            memory_plan,
        } => {
            let bench_runner = aether::inference::LlamaRunner::from_gguf(&model)?;
            if memory_plan {
                aether::MemoryAwareScheduler::print_assignment(&bench_runner.layer_assignment);
            }
            let mut bench = BenchRun::new(&model, &prompt, max_tokens)?;
            bench.run(compare_llama_cpp, &llama_binary)?;

            if let Some(path) = output_json {
                bench.save_json(&path)?;
                eprintln!("[aether] Benchmark results saved to {}", path);
            }
        }
    }
    Ok(())
}

struct BenchRun {
    model_path: String,
    prompt: String,
    max_tokens: usize,
    /// Results populated after run()
    load_time_s: f64,
    model_mb: f64,
    kv_mb: f64,
    peak_memory_mb: f64,
    prefill_time_ms: f64,
    prefill_tokens: usize,
    prefill_tok_s: f64,
    decode_time_ms: f64,
    decode_tokens: usize,
    decode_tok_s: f64,
    ttft_ms: f64,
    first_token: String,
    output_text: String,
}

impl BenchRun {
    fn new(model_path: &str, prompt: &str, max_tokens: usize) -> Result<Self, Error> {
        Ok(Self {
            model_path: model_path.to_string(),
            prompt: prompt.to_string(),
            max_tokens,
            load_time_s: 0.0,
            model_mb: 0.0,
            kv_mb: 0.0,
            peak_memory_mb: 0.0,
            prefill_time_ms: 0.0,
            prefill_tokens: 0,
            prefill_tok_s: 0.0,
            decode_time_ms: 0.0,
            decode_tokens: 0,
            decode_tok_s: 0.0,
            ttft_ms: 0.0,
            first_token: String::new(),
            output_text: String::new(),
        })
    }

    fn run(&mut self, compare: bool, llama_bin: &str) -> Result<(), Error> {
        // ── Phase 1: Load ────────────────────────────────────────────────
        let load_start = Instant::now();
        eprint!("[aether] Loading model... ");
        std::io::stderr().flush().ok();
        let mut runner = LlamaRunner::from_gguf(&self.model_path)?;
        self.load_time_s = load_start.elapsed().as_secs_f64();
        eprintln!("{:.2}s", self.load_time_s);

        // ── Phase 2: Prefill (time to first token) ───────────────────────
        eprint!("[aether] Prefill... ");
        std::io::stderr().flush().ok();

        let ttft_start = Instant::now();
        // We need to run generate with 1 token to measure TTFT
        // But generate() measures total time including decode.
        // Let's run prefill directly and sample one token.
        runner.kv.reset();
        let token_ids = runner.tokenizer.encode(&self.prompt, true);
        self.prefill_tokens = token_ids.len();

        let prefill_start = Instant::now();
        let mut last_logits = runner.prefill(&token_ids)?;
        let prefill_time = prefill_start.elapsed();
        self.prefill_time_ms = prefill_time.as_secs_f64() * 1000.0;
        self.prefill_tok_s = self.prefill_tokens as f64 / prefill_time.as_secs_f64().max(1e-9);

        // Sample first token
        let next_token = aether::inference::runner::sample(
            &last_logits,
            0.0,
            0.0,
            &token_ids.iter().copied().collect::<HashSet<u32>>(),
            1.0,
        );
        let ttft = ttft_start.elapsed();
        self.ttft_ms = ttft.as_secs_f64() * 1000.0;
        self.first_token = runner.tokenizer.decode_one(next_token);
        eprintln!(
            "{:.1}ms (TTFT), first token: '{}'",
            self.ttft_ms, self.first_token
        );

        // ── Phase 3: Decode ──────────────────────────────────────────────
        eprint!("[aether] Decoding {} tokens... ", self.max_tokens);
        std::io::stderr().flush().ok();

        let decode_start = Instant::now();
        let mut all_tokens = token_ids.clone();
        let mut prev_token = next_token;
        all_tokens.push(prev_token);

        for step in 0..self.max_tokens.saturating_sub(1) {
            if prev_token == runner.tokenizer.eos_id {
                break;
            }
            if runner.kv.seq_len >= runner.kv.max_seq {
                break;
            }

            let pos = self.prefill_tokens + step;
            let mut step_tel = vec![
                aether::inference::telemetry::LayerTelemetry::default();
                runner.model.config.num_layers
            ];
            last_logits = runner.decode_step(prev_token, pos, &mut step_tel)?;

            let next = aether::inference::runner::sample(
                &last_logits,
                0.0,
                0.0,
                &all_tokens.iter().copied().collect::<HashSet<u32>>(),
                1.0,
            );
            all_tokens.push(next);
            if next == runner.tokenizer.eos_id {
                break;
            }
            prev_token = next;
        }

        let decode_time = decode_start.elapsed();
        self.decode_time_ms = decode_time.as_secs_f64() * 1000.0;
        self.decode_tokens = all_tokens.len().saturating_sub(self.prefill_tokens + 1);
        self.decode_tok_s = self.decode_tokens as f64 / decode_time.as_secs_f64().max(1e-9);
        eprintln!(
            "{:.1}ms ({:.1} tok/s)",
            self.decode_time_ms, self.decode_tok_s
        );

        // Decode output
        let generated_ids = &all_tokens[self.prefill_tokens..];
        self.output_text = runner.tokenizer.decode(generated_ids);

        // ── Memory ──────────────────────────────────────────────────────
        let model_bytes = aether::inference::runner::estimate_model_bytes(&runner.model);
        let kv_bytes = runner.kv.size_bytes();
        self.model_mb = model_bytes as f64 / 1e6;
        self.kv_mb = kv_bytes as f64 / 1e6;
        self.peak_memory_mb = (model_bytes + kv_bytes) as f64 / 1e6;

        // ── Print results ────────────────────────────────────────────────
        self.print_human();

        // ── Compare with llama.cpp if requested ──────────────────────────
        if compare {
            self.compare_with_llama_cpp(llama_bin)?;
        }

        Ok(())
    }

    fn print_human(&self) {
        println!();
        println!("╔════════════════════════════════════════════════════╗");
        println!("║           Aether Benchmark Results                ║");
        println!("╠════════════════════════════════════════════════════╣");
        println!("║  Model       : {:<38}", self.model_path);
        println!("║  Prompt      : {:<38}", self.prompt);
        println!("║  Load time   : {:.2}s", self.load_time_s);
        println!("║  Model size  : {:.0} MB", self.model_mb);
        println!("║  KV cache    : {:.1} MB", self.kv_mb);
        println!("║  Total mem   : {:.0} MB", self.peak_memory_mb);
        println!("╠════════════════════════════════════════════════════╣");
        println!(
            "║  Prefill     : {} tokens  ({:.1} tok/s)",
            self.prefill_tokens, self.prefill_tok_s
        );
        println!("║  TTFT        : {:.1} ms", self.ttft_ms);
        println!(
            "║  Decode      : {} tokens  {:.1}ms  ({:.1} tok/s)",
            self.decode_tokens, self.decode_time_ms, self.decode_tok_s
        );
        println!("║  First token : '{}'", self.first_token);
        println!("╚════════════════════════════════════════════════════╝");
        println!("{}", self.output_text);
    }

    fn save_json(&self, path: &str) -> Result<(), Error> {
        let obj = serde_json::json!({
            "benchmark": {
                "model_path": self.model_path,
                "prompt": self.prompt,
                "max_tokens": self.max_tokens,
            },
            "results": {
                "load_time_s": self.load_time_s,
                "model_mb": self.model_mb,
                "kv_cache_mb": self.kv_mb,
                "peak_memory_mb": self.peak_memory_mb,
                "prefill_tokens": self.prefill_tokens,
                "prefill_tok_s": self.prefill_tok_s,
                "prefill_time_ms": self.prefill_time_ms,
                "ttft_ms": self.ttft_ms,
                "decode_tokens": self.decode_tokens,
                "decode_time_ms": self.decode_time_ms,
                "decode_tok_s": self.decode_tok_s,
                "first_token": self.first_token,
                "output": self.output_text,
            }
        });
        let json_str = serde_json::to_string_pretty(&obj)
            .map_err(|e| Error::ExecutionError(format!("JSON serialize: {}", e)))?;
        std::fs::write(path, json_str)
            .map_err(|e| Error::ExecutionError(format!("Write JSON: {}", e)))?;
        Ok(())
    }

    fn compare_with_llama_cpp(&self, llama_bin: &str) -> Result<(), Error> {
        eprintln!("\n[aether] Running llama.cpp benchmark for comparison...");

        // Build llama.cpp command — single-turn + perf for timing output
        let output = Command::new(llama_bin)
            .args([
                "-m",
                &self.model_path,
                "-p",
                &self.prompt,
                "-n",
                &self.max_tokens.to_string(),
                "-t",
                &std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(8)
                    .to_string(),
                "-st",
                "--perf",
            ])
            .output()
            .map_err(|e| Error::ExecutionError(format!("llama-cli exec: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("[aether] llama.cpp stderr:\n{}", stderr);
            return Err(Error::ExecutionError("llama-cli failed".into()));
        }

        // Parse timing from llama.cpp stdout (format: "[ Prompt: XXX.X t/s | Generation: YYY.Y t/s ]")
        let stdout_str = String::from_utf8_lossy(&output.stdout);
        let mut llama_prefill_tok_s = 0.0f64;
        let mut llama_decode_tok_s = 0.0f64;

        for line in stdout_str.lines() {
            if line.contains("Prompt:") && line.contains("Generation:") {
                // "Prompt: 416.0"
                if let Some(p_start) = line.find("Prompt:") {
                    let after = line[p_start + 7..].trim();
                    if let Some(val) = after.split_whitespace().next() {
                        llama_prefill_tok_s = val.parse().unwrap_or(0.0);
                    }
                }
                // "Generation: 119.5"
                if let Some(g_start) = line.find("Generation:") {
                    let after = line[g_start + 11..].trim();
                    if let Some(val) = after.split_whitespace().next() {
                        llama_decode_tok_s = val.parse().unwrap_or(0.0);
                    }
                }
                break;
            }
        }

        // Print comparison
        println!();
        println!("╔══════════════════════════════════════════════════════════╗");
        println!("║              Aether vs llama.cpp                        ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!(
            "║  {:>15}  {:>15}  {:>15}  {:>10}║",
            "", "Aether", "llama.cpp", "Ratio"
        );
        println!("║  ────────────────────────────────────────────────────────── ║");
        println!(
            "║  {:>15}  {:>15.1}  {:>15.1}  {:>10.2}║",
            "Prefill tok/s",
            self.prefill_tok_s,
            llama_prefill_tok_s,
            if llama_prefill_tok_s > 0.0 {
                self.prefill_tok_s / llama_prefill_tok_s
            } else {
                0.0
            }
        );
        println!(
            "║  {:>15}  {:>15.1}  {:>15.1}  {:>10.2}║",
            "Decode tok/s",
            self.decode_tok_s,
            llama_decode_tok_s,
            if llama_decode_tok_s > 0.0 {
                self.decode_tok_s / llama_decode_tok_s
            } else {
                0.0
            }
        );
        println!(
            "║  {:>15}  {:>15.1}  {:>15}  {:>10}║",
            "TTFT (ms)", self.ttft_ms, "-", "-"
        );
        println!("╚══════════════════════════════════════════════════════════╝");

        // Also save comparison to JSON if output_json was specified
        Ok(())
    }
}
