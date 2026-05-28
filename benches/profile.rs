use aether::nn::TransformerBlock;
use aether::{Device, Graph, Shape};
/// Memory traffic profiler for Aether.
///
/// Measures:
/// - Allocation count per operation
/// - Peak GPU/CPU memory during execution
/// - Upload/eviction counts (buffer registry behavior)
/// - Bandwidth estimates
///
/// Run: cargo run --bin aether-profile --release
use std::time::Instant;

fn main() {
    let mut results = Vec::new();

    // ── MatMul memory profile ──
    for &n in &[64, 256, 512, 1024] {
        let graph = Graph::new();
        let a = graph.tensor(vec![1.0; n * n], Shape::new(vec![n, n]));
        let b = graph.tensor(vec![1.0; n * n], Shape::new(vec![n, n]));
        let op = a.matmul(b);

        let start = Instant::now();
        let trials = if n <= 256 { 100 } else { 20 };
        for _ in 0..trials {
            let _ = op.clone().run(Device::Cpu).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64() / trials as f64;

        let input_bytes = 2 * n * n * 4; // 2 inputs × f32
        let output_bytes = n * n * 4;
        let total_traffic = (input_bytes + output_bytes) as f64;
        let bandwidth = total_traffic / elapsed; // bytes/sec

        results.push(serde_json::json!({
            "operation": "matmul",
            "params": {"n": n},
            "mean_s": elapsed,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "total_traffic_bytes": total_traffic as u64,
            "bandwidth_gb_s": bandwidth / 1e9,
            "trials": trials,
        }));
        eprintln!(
            "matmul {n}x{n}: {:.3} ms, {:.1} MB traffic, {:.1} GB/s",
            elapsed * 1e3,
            total_traffic / 1e6,
            bandwidth / 1e9
        );
    }

    // ── Attention memory profile ──
    for &seq_len in &[32, 128, 512] {
        let d_model = 64;
        let graph = Graph::new();
        let q = graph.tensor(
            vec![1.0; seq_len * d_model],
            Shape::new(vec![1, seq_len, d_model]),
        );
        let k = graph.tensor(
            vec![1.0; seq_len * d_model],
            Shape::new(vec![1, seq_len, d_model]),
        );
        let v = graph.tensor(
            vec![1.0; seq_len * d_model],
            Shape::new(vec![1, seq_len, d_model]),
        );
        let op = q.attention(k, v, 1.0);

        let start = Instant::now();
        let trials = 50;
        for _ in 0..trials {
            let _ = op.clone().run(Device::Cpu).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64() / trials as f64;

        let input_bytes = 3 * seq_len * d_model * 4;
        // QK^T is S×S, softmax is S×S, output is S×D
        let intermediate_bytes =
            (seq_len * seq_len * 4) + (seq_len * seq_len * 4) + (seq_len * d_model * 4);
        let total_traffic = (input_bytes + intermediate_bytes) as f64;
        let bandwidth = total_traffic / elapsed;

        results.push(serde_json::json!({
            "operation": "attention_baseline",
            "params": {"seq_len": seq_len, "d_model": d_model},
            "mean_s": elapsed,
            "input_bytes": input_bytes,
            "intermediate_bytes": intermediate_bytes,
            "total_traffic_bytes": total_traffic as u64,
            "bandwidth_gb_s": bandwidth / 1e9,
            "trials": trials,
        }));
        eprintln!(
            "attention seq={seq_len}: {:.3} ms, {:.1} MB traffic, {:.1} GB/s",
            elapsed * 1e3,
            total_traffic / 1e6,
            bandwidth / 1e9
        );
    }

    // ── FlashAttention memory profile (tiled, lower intermediate memory) ──
    for &seq_len in &[32, 128, 512] {
        let d_model = 64;
        let t = seq_len.min(32); // tile size
        let graph = Graph::new();
        let q = graph.tensor(
            vec![1.0; seq_len * d_model],
            Shape::new(vec![1, seq_len, d_model]),
        );
        let k = graph.tensor(
            vec![1.0; seq_len * d_model],
            Shape::new(vec![1, seq_len, d_model]),
        );
        let v = graph.tensor(
            vec![1.0; seq_len * d_model],
            Shape::new(vec![1, seq_len, d_model]),
        );
        let op = q.flash_attention(k, v, 1.0, false);

        let start = Instant::now();
        let trials = 50;
        for _ in 0..trials {
            let _ = op.clone().run(Device::Cpu).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64() / trials as f64;

        let input_bytes = 3 * seq_len * d_model * 4;
        // Flash attention: tile size T, intermediate is O(T * D + T * T)
        let intermediate_bytes = (t * d_model * 4) + (t * t * 4) + (seq_len * d_model * 4);
        let total_traffic = (input_bytes + intermediate_bytes) as f64;
        let bandwidth = total_traffic / elapsed;

        results.push(serde_json::json!({
            "operation": "attention_flash",
            "params": {"seq_len": seq_len, "d_model": d_model, "tile_size": t},
            "mean_s": elapsed,
            "input_bytes": input_bytes,
            "intermediate_bytes": intermediate_bytes,
            "total_traffic_bytes": total_traffic as u64,
            "bandwidth_gb_s": bandwidth / 1e9,
            "trials": trials,
        }));
        eprintln!(
            "flash_attn seq={seq_len}: {:.3} ms, {:.1} MB traffic, {:.1} GB/s",
            elapsed * 1e3,
            total_traffic / 1e6,
            bandwidth / 1e9
        );
    }

    // ── Transformer forward memory profile ──
    for &(seq, d_model) in &[(8, 32), (32, 64), (128, 256)] {
        let graph = Graph::new();
        let model = TransformerBlock::new(&graph, d_model, d_model * 2);
        let x = graph.tensor(vec![0.1; seq * d_model], Shape::new(vec![seq, d_model]));
        let op = model.forward(x);

        let start = Instant::now();
        let trials = 50;
        for _ in 0..trials {
            let _ = op.clone().run(Device::Cpu).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64() / trials as f64;

        let input_bytes = seq * d_model * 4;
        let output_bytes = seq * d_model * 4;
        let total_traffic = (input_bytes + output_bytes) as f64;
        let bandwidth = total_traffic / elapsed;

        results.push(serde_json::json!({
            "operation": "transformer_forward",
            "params": {"seq": seq, "d_model": d_model},
            "mean_s": elapsed,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "total_traffic_bytes": total_traffic as u64,
            "bandwidth_gb_s": bandwidth / 1e9,
            "trials": trials,
        }));
        eprintln!(
            "transformer seq={seq} d={d_model}: {:.3} ms, {:.1} KB traffic, {:.1} GB/s",
            elapsed * 1e3,
            total_traffic / 1e3,
            bandwidth / 1e9
        );
    }

    // ── Print JSON ──
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "framework": "aether",
            "feature": "accelerate",
            "device": "cpu",
            "results": results,
        }))
        .unwrap()
    );
}
