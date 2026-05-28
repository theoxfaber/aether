use aether::{Device, Dtype, Graph, Shape};
use std::time::Instant;

fn main() {
    let mut results: Vec<serde_json::Value> = Vec::new();

    // ── MatMul ──
    for &n in &[4, 16, 64, 256, 512, 1024] {
        let graph = Graph::new();
        let a = graph.tensor(vec![1.0; n * n], Shape::new(vec![n, n]));
        let b = graph.tensor(vec![1.0; n * n], Shape::new(vec![n, n]));
        let op = a.matmul(b);
        // warmup
        let _ = op.clone().run(Device::Cpu).unwrap();
        let start = Instant::now();
        let trials = if n <= 64 {
            500
        } else if n <= 256 {
            100
        } else {
            20
        };
        for _ in 0..trials {
            let _ = op.clone().run(Device::Cpu).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64() / trials as f64;
        results.push(serde_json::json!({
            "benchmark": "matmul",
            "params": {"n": n},
            "mean_s": elapsed,
            "trials": trials,
        }));
        eprintln!("matmul {n}x{n}: {:.3} ms", elapsed * 1e3);
    }

    // ── Attention variants (seq_len=32, d_model=32) ──
    for &seq_len in &[8, 32, 128] {
        let d_model = 64;
        for (variant, attn_fn) in [
            (
                "baseline",
                Box::new(|q: aether::GraphTensor, k, v| q.attention(k, v, 1.0))
                    as Box<dyn Fn(_, _, _) -> _>,
            ),
            (
                "causal",
                Box::new(|q: aether::GraphTensor, k, v| q.causal_attention(k, v, 1.0, 4)),
            ),
            (
                "flash",
                Box::new(|q: aether::GraphTensor, k, v| q.flash_attention(k, v, 1.0, false)),
            ),
        ] {
            let compute = |seq| {
                let graph = Graph::new();
                let q = graph.tensor(
                    vec![1.0; seq * d_model],
                    Shape::new(vec![1, seq, d_model]),
                );
                let k = graph.tensor(
                    vec![1.0; seq * d_model],
                    Shape::new(vec![1, seq, d_model]),
                );
                let v = graph.tensor(
                    vec![1.0; seq * d_model],
                    Shape::new(vec![1, seq, d_model]),
                );
                attn_fn(q, k, v)
            };
            let op = compute(seq_len);
            let _ = op.clone().run(Device::Cpu).unwrap();
            let trials = 200;
            let start = Instant::now();
            for _ in 0..trials {
                let _ = op.clone().run(Device::Cpu).unwrap();
            }
            let elapsed = start.elapsed().as_secs_f64() / trials as f64;
            results.push(serde_json::json!({
                "benchmark": format!("attention_{variant}"),
                "params": {"seq_len": seq_len, "d_model": d_model},
                "mean_s": elapsed,
                "trials": trials,
            }));
            eprintln!("attention/{variant} seq={seq_len}: {:.3} ms", elapsed * 1e3);
        }
    }

    // ── Cast operations (forward only: F32 -> F16, F32 -> BF16) ──
    for &n in &[64, 1024, 4096] {
        for (variant, cast_to) in [("f32_to_f16", Dtype::F16), ("f32_to_bf16", Dtype::BF16)] {
            let graph = Graph::new();
            let a = graph.tensor(vec![1.5; n], Shape::new(vec![n]));
            let op = a.cast(cast_to);
            let _ = op.clone().run(Device::Cpu).unwrap();
            let trials = 500;
            let start = Instant::now();
            for _ in 0..trials {
                let out = op.clone().run(Device::Cpu).unwrap();
                let _ = out.data_raw();
            }
            let elapsed = start.elapsed().as_secs_f64() / trials as f64;
            results.push(serde_json::json!({
                "benchmark": format!("cast_{variant}"),
                "params": {"n": n},
                "mean_s": elapsed,
                "trials": trials,
            }));
            eprintln!("cast/{variant} n={n}: {:.3} µs", elapsed * 1e6);
        }
    }

    // ── Transformer forward ──
    for &(seq, d_model) in &[(8, 32), (32, 64)] {
        use aether::nn::TransformerBlock;
        let graph = Graph::new();
        let model = TransformerBlock::new(&graph, d_model, d_model * 2);
        let x = graph.tensor(vec![0.1; seq * d_model], Shape::new(vec![seq, d_model]));
        let op = model.forward(x);
        let _ = op.clone().run(Device::Cpu).unwrap();
        let trials = 100;
        let start = Instant::now();
        for _ in 0..trials {
            let _ = op.clone().run(Device::Cpu).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64() / trials as f64;
        results.push(serde_json::json!({
            "benchmark": "transformer_forward",
            "params": {"seq": seq, "d_model": d_model},
            "mean_s": elapsed,
            "trials": trials,
        }));
        eprintln!(
            "transformer_forward seq={seq} d={d_model}: {:.3} ms",
            elapsed * 1e3
        );
    }

    // ── Print final JSON ──
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "framework": "aether",
            "device": "cpu",
            "timestamp": chrono_now(),
            "results": results,
        }))
        .unwrap()
    );
}

fn chrono_now() -> String {
    // Simple UTC timestamp without external crate
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", d.as_secs())
}
