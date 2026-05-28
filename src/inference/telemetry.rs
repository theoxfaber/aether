/// Runtime execution telemetry for Aether inference.
///
/// Tracks per-layer wall-clock latency, total throughput, and memory usage.
/// Designed to be zero-overhead when not printing: all measurements are
/// simple `Instant::now()` + duration accumulation.
use std::time::{Duration, Instant};
use tracing::info;

/// Per-layer timing breakdown collected during a single decode step.
#[derive(Default, Clone)]
pub struct LayerTelemetry {
    pub attn_us: u64, // microseconds for attention (QKV + softmax + output)
    pub mlp_us: u64,  // microseconds for FFN
    pub norm_us: u64, // microseconds for both RMSNorm calls
}

/// Full inference session telemetry.
pub struct ExecutionTelemetry {
    pub load_time: Duration,
    pub prefill_time: Duration,
    pub prefill_tokens: usize,
    pub decode_time: Duration,
    pub decode_tokens: usize,
    pub layer_avg: Vec<LayerTelemetry>,
    pub peak_memory_bytes: usize,

    // Internal accumulators (not printed directly)
    layer_sums: Vec<LayerTelemetry>,
    decode_steps: usize,
}

impl ExecutionTelemetry {
    pub fn new(num_layers: usize) -> Self {
        Self {
            load_time: Duration::ZERO,
            prefill_time: Duration::ZERO,
            prefill_tokens: 0,
            decode_time: Duration::ZERO,
            decode_tokens: 0,
            layer_avg: vec![LayerTelemetry::default(); num_layers],
            peak_memory_bytes: 0,
            layer_sums: vec![LayerTelemetry::default(); num_layers],
            decode_steps: 0,
        }
    }

    pub fn record_load(&mut self, t: Duration) {
        self.load_time = t;
    }

    pub fn record_prefill(&mut self, t: Duration, tokens: usize) {
        self.prefill_time = t;
        self.prefill_tokens = tokens;
    }

    pub fn record_decode_step(&mut self, t: Duration, layer_data: &[LayerTelemetry]) {
        self.decode_time += t;
        self.decode_tokens += 1;
        self.decode_steps += 1;
        for (i, ld) in layer_data.iter().enumerate() {
            if i < self.layer_sums.len() {
                self.layer_sums[i].attn_us += ld.attn_us;
                self.layer_sums[i].mlp_us += ld.mlp_us;
                self.layer_sums[i].norm_us += ld.norm_us;
            }
        }
    }

    pub fn record_memory(&mut self, bytes: usize) {
        self.peak_memory_bytes = self.peak_memory_bytes.max(bytes);
    }

    /// Compute averages and print the telemetry summary.
    pub fn print_summary(&mut self) {
        // Compute layer averages
        let steps = self.decode_steps.max(1) as u64;
        for (avg, sum) in self.layer_avg.iter_mut().zip(self.layer_sums.iter()) {
            avg.attn_us = sum.attn_us / steps;
            avg.mlp_us = sum.mlp_us / steps;
            avg.norm_us = sum.norm_us / steps;
        }

        info!("");
        info!("╔══════════════════════════════════════════════════════════╗");
        info!("║              Aether Inference Telemetry                  ║");
        info!("╠══════════════════════════════════════════════════════════╣");
        info!("║  Load time      : {:>8.2}s", self.load_time.as_secs_f64());
        info!(
            "║  Peak memory    : {:>8.1} MB",
            self.peak_memory_bytes as f64 / 1e6
        );

        if self.prefill_tokens > 0 {
            let prefill_tps =
                self.prefill_tokens as f64 / self.prefill_time.as_secs_f64().max(1e-9);
            info!(
                "║  Prefill        : {:>4} tokens  {:.1}ms  ({:.1} tok/s)",
                self.prefill_tokens,
                self.prefill_time.as_secs_f64() * 1000.0,
                prefill_tps
            );
        }

        if self.decode_tokens > 0 {
            let decode_tps = self.decode_tokens as f64 / self.decode_time.as_secs_f64().max(1e-9);
            info!(
                "║  Decode         : {:>4} tokens  {:.1}ms  ({:.1} tok/s)",
                self.decode_tokens,
                self.decode_time.as_secs_f64() * 1000.0,
                decode_tps
            );
        }

        info!("╠══════════════════════════════════════════════════════════╣");
        info!("║  Layer breakdown (avg per decode step):                  ║");
        info!(
            "║  {:>3}  {:>8}  {:>8}  {:>8}  {:>8}",
            "Lyr", "Attn(us)", "MLP(us)", "Norm(us)", "Tot(us)"
        );

        // Print first 4 and last 2 layers to keep output compact
        let n = self.layer_avg.len();
        let show: Vec<usize> = (0..4.min(n))
            .chain(if n > 6 { vec![n - 2, n - 1] } else { vec![] })
            .collect();
        let mut last_shown = None;
        for &i in &show {
            if let Some(prev) = last_shown {
                if i > prev + 1 {
                    info!("║  ...");
                }
            }
            let la = &self.layer_avg[i];
            let tot = la.attn_us + la.mlp_us + la.norm_us;
            info!(
                "║  {:>3}  {:>8}  {:>8}  {:>8}  {:>8}",
                i, la.attn_us, la.mlp_us, la.norm_us, tot
            );
            last_shown = Some(i);
        }
        info!("╚══════════════════════════════════════════════════════════╝");
    }
}

/// Simple stopwatch utility.
pub struct Stopwatch(Instant);

impl Stopwatch {
    pub fn start() -> Self {
        Self(Instant::now())
    }
    pub fn elapsed_us(&self) -> u64 {
        self.0.elapsed().as_micros() as u64
    }
    pub fn elapsed(&self) -> Duration {
        self.0.elapsed()
    }
}
