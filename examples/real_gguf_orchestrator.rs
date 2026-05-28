use aether::loader::dequant::dequantize;
use aether::loader::gguf::GGUFLoader;
use aether::{Device, Graph, Shape};
use std::collections::HashMap;
use std::time::Instant;

const D_MODEL: usize = 2048;
const N_KV_HEADS: usize = 4;
const HEAD_DIM: usize = D_MODEL / 32;
const D_FF: usize = 5632;
const N_LAYERS: usize = 22;
const SEQ_LEN: usize = 4;

type Dequantized = HashMap<String, Vec<f32>>;

fn load_and_dequant(model: &aether::loader::gguf::GGUFModel) -> Dequantized {
    let mut w = HashMap::new();
    let mut total = 0usize;
    for (name, tensor) in &model.tensors {
        let deq = dequantize(&tensor.data, tensor.dtype, &tensor.shape);
        total += deq.len();
        eprintln!("  {} -> {} floats", name, deq.len());
        w.insert(name.clone(), deq);
    }
    eprintln!(
        "  Total: {} floats ({:.1} GB)",
        total,
        total as f64 * 4.0 / 1e9
    );
    w
}

/// Number of f32 bytes per weight tensor across all 22 layers.
fn weight_bytes_per_layer() -> usize {
    let qkv = D_MODEL * D_MODEL + D_MODEL * (N_KV_HEADS * HEAD_DIM) * 2;
    let ffn = D_MODEL * D_FF * 2 + D_FF * D_MODEL;
    (qkv + ffn) * 4
}

fn main() {
    eprintln!("=== Real GGUF → Memory Orchestrator ===");
    eprintln!("Loading...");
    let model =
        GGUFLoader::load("tinyllama-q4.gguf").expect("GGUF file not found at project root.");
    eprintln!("{} tensors. Dequantizing...", model.tensors.len());
    let w = load_and_dequant(&model);

    let budgets: &[(usize, &str)] = &[
        (128_000_000, "128 MB"),
        (256_000_000, "256 MB"),
        (512_000_000, "512 MB"),
        (1_000_000_000, "1 GB"),
        (3_000_000_000, "3 GB"),
        (6_000_000_000, "6 GB"),
    ];

    println!();
    println!(
        "Single-layer weight data: ~{:.0} MB ({} layers = ~{:.0} GB f32)",
        weight_bytes_per_layer() as f64 / 1e6,
        N_LAYERS,
        (weight_bytes_per_layer() * N_LAYERS) as f64 / 1e9
    );
    println!();
    println!(
        "{:<10} {:>10} {:>14} {:>10} {:>10}  Notes",
        "Budget", "Time(ms)", "Peak GPU", "Evicts", "Uploads"
    );
    println!(
        "{:-<10}-+-{:-<10}-+-{:-<14}-+-{:-<10}-+-{:-<10}",
        "", "", "", "", ""
    );

    for &(budget, label) in budgets {
        let mut graph = Graph::new();
        graph.set_gpu_memory_limit(budget);

        let input_data = vec![0.1; SEQ_LEN * D_MODEL];
        let mut layer_outputs = Vec::new();

        for layer in 0..N_LAYERS {
            let x = graph.tensor(input_data.clone(), Shape::new(vec![SEQ_LEN, D_MODEL]));

            let wq = graph.tensor(
                w[&format!("blk.{}.attn_q.weight", layer)].clone(),
                Shape::new(vec![D_MODEL, D_MODEL]),
            );
            let wo = graph.tensor(
                w[&format!("blk.{}.attn_output.weight", layer)].clone(),
                Shape::new(vec![D_MODEL, D_MODEL]),
            );

            let w_gate = graph.tensor(
                w[&format!("blk.{}.ffn_gate.weight", layer)].clone(),
                Shape::new(vec![D_MODEL, D_FF]),
            );
            let w_up = graph.tensor(
                w[&format!("blk.{}.ffn_up.weight", layer)].clone(),
                Shape::new(vec![D_MODEL, D_FF]),
            );
            let w_down = graph.tensor(
                w[&format!("blk.{}.ffn_down.weight", layer)].clone(),
                Shape::new(vec![D_FF, D_MODEL]),
            );

            let q = x.matmul(wq);
            let attn_out = q.matmul(wo);

            let gate = x.matmul(w_gate);
            let up = x.matmul(w_up);
            let silu = gate.sigmoid().mul(up);
            let ffn_out = silu.matmul(w_down);

            layer_outputs.push(attn_out.add(ffn_out));
        }

        let output = layer_outputs.into_iter().reduce(|a, b| a.add(b)).unwrap();

        let start = Instant::now();
        let result = match output.run(Device::Wgpu) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("{:<10} {:>10}  ERROR: {}", label, "-", e);
                continue;
            }
        };
        let ms = start.elapsed().as_secs_f64() * 1000.0;

        let evictions = graph.eviction_count();
        let uploads = graph.upload_count();

        let note = if evictions > 0 {
            format!("evictions: {}", evictions)
        } else {
            "".into()
        };
        println!(
            "{:<10} {:>10.2} {:>9.1} MB {:>10} {:>10}  {}",
            label,
            ms,
            graph.peak_gpu_bytes() as f64 / 1e6,
            evictions,
            uploads,
            note
        );

        if budget == budgets.last().unwrap().0 {
            let data = result.data();
            let _nan_count = data.iter().filter(|v| v.is_nan()).count();
            let valid: Vec<f32> = data
                .iter()
                .copied()
                .filter(|v| !v.is_nan() && v.is_finite())
                .collect();
            if !valid.is_empty() {
                let mn = valid.iter().cloned().fold(f32::NAN, f32::min);
                let mx = valid.iter().cloned().fold(f32::NAN, f32::max);
                println!(
                    "  Output range: [{:.6}, {:.6}]  (valid: {}/{})",
                    mn,
                    mx,
                    valid.len(),
                    data.len()
                );
            }
        }

        // Reset stats for next iteration (Graph stays alive to hold stats)
    }

    println!();
    println!("=== Memory Orchestrator Summary ===");
    println!("- Each layer has 7 weight tensors (attn Q/K/V/O + ffn gate/up/down)");
    println!(
        "- Total dequantized f32 weights: ~{:.1} GB",
        N_LAYERS as f64 * weight_bytes_per_layer() as f64 / 1e9
    );
    println!("- LRU eviction kicks in when budget < f32 weight size across live layers");
    println!("- The orchestrator uses BufferRegistry (LRU) + LivenessMap (tensor lifetime)");
}
