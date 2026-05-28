use aether::{Device, Graph, Shape};
use std::time::Instant;

fn main() {
    println!("=== Aether Memory Tiering & Prefetching Transformer Simulation ===");

    let mut graph = Graph::new();
    // 12MB soft limit. A single 1024x1024 f32 tensor is 4MB.
    // Total weight tensors across 8 layers far exceed 12MB, demonstrating active eviction/prefetching.
    graph.set_gpu_memory_limit(12 * 1024 * 1024);

    let batch_size = 1024;
    let d_model = 1024;

    // Input sequence: [1024, 1024]
    let mut x = graph.tensor(
        vec![0.1; batch_size * d_model],
        Shape::new(vec![batch_size, d_model]),
    );

    // Construct 8-layer Transformer FFN simulation
    // Layer structure: Output = Relu(X @ W1 + b1) @ W2 + b2 + X
    for _layer in 0..8 {
        let w1 = graph.tensor(
            vec![0.001; d_model * d_model],
            Shape::new(vec![d_model, d_model]),
        );
        let b1 = graph.tensor(
            vec![0.005; batch_size * d_model],
            Shape::new(vec![batch_size, d_model]),
        );
        let w2 = graph.tensor(
            vec![0.002; d_model * d_model],
            Shape::new(vec![d_model, d_model]),
        );
        let b2 = graph.tensor(
            vec![0.01; batch_size * d_model],
            Shape::new(vec![batch_size, d_model]),
        );

        let h1 = x.matmul(w1).add(b1).relu();
        let h2 = h1.matmul(w2).add(b2);
        x = x.add(h2);
    }

    println!("Graph constructed. Executing 8-layer Transformer Simulation on M2 GPU...");

    let start = Instant::now();
    let result = x.run(Device::Wgpu).unwrap();
    let elapsed = start.elapsed();

    println!("Simulation completed successfully in {:?}", elapsed);
    println!("Result shape: {:?}", result.shape());
    println!(
        "Peak GPU memory usage: {:.2} MB",
        graph.peak_gpu_bytes() as f64 / (1024.0 * 1024.0)
    );
    println!("Total CPU->GPU uploads: {}", graph.upload_count());
    println!("Total GPU->CPU evictions: {}", graph.eviction_count());

    // Print first few elements of result
    let sample = &result.data()[0..5];
    println!("First 5 output values: {:?}", sample);
}
