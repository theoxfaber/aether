use aether::{Device, Graph, Shape};
use std::time::Instant;

fn bench(device: Device, label: &str, size: usize) {
    let data: Vec<f32> = (0..size * size).map(|x| x as f32 * 0.01).collect();

    let graph = Graph::new();
    let a = graph.tensor(data.clone(), Shape::new(vec![size, size]));
    let b = graph.tensor(data.clone(), Shape::new(vec![size, size]));

    let start = Instant::now();
    let _result = a.matmul(b).relu().run(device).unwrap();
    let elapsed = start.elapsed();

    println!("{}: {}x{} matmul+relu = {:?}", label, size, size, elapsed);
}

fn main() {
    // Warm up the GPU backend (device initialization and shader compilation)
    println!("Warming up GPU backend...");
    let warmup_graph = Graph::new();
    let wa = warmup_graph.tensor(vec![1.0], Shape::new(vec![1, 1]));
    let wb = warmup_graph.tensor(vec![1.0], Shape::new(vec![1, 1]));
    let _ = wa.matmul(wb).relu().run(Device::Wgpu).unwrap();
    println!(
        "Warmup complete. Running benchmark...
"
    );

    bench(Device::Cpu, "CPU", 4096);
    bench(Device::Wgpu, "GPU (Metal/M2)", 4096);
}
