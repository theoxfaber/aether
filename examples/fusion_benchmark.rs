use aether::{Device, Graph, Shape};
use std::time::Instant;

fn run_benchmark(size: usize, enable_fusion: bool, iterations: usize) -> std::time::Duration {
    let data: Vec<f32> = (0..size * size).map(|x| x as f32 * 0.001).collect();

    // Warm up the specific configuration first
    {
        let graph = Graph::new();
        graph.enable_fusion(enable_fusion);
        let a = graph.tensor(data.clone(), Shape::new(vec![size, size]));
        let b = graph.tensor(data.clone(), Shape::new(vec![size, size]));
        let _ = a.matmul(b).relu().run(Device::Wgpu).unwrap();
    }

    let mut total_duration = std::time::Duration::ZERO;

    for _ in 0..iterations {
        let graph = Graph::new();
        graph.enable_fusion(enable_fusion);
        let a = graph.tensor(data.clone(), Shape::new(vec![size, size]));
        let b = graph.tensor(data.clone(), Shape::new(vec![size, size]));

        let start = Instant::now();
        let _ = a.matmul(b).relu().run(Device::Wgpu).unwrap();
        total_duration += start.elapsed();
    }

    total_duration / (iterations as u32)
}

fn bench_size(size: usize, iterations: usize) {
    println!("--- Size {}x{} ---", size, size);
    let unfused_time = run_benchmark(size, false, iterations);
    println!("GPU Unfused Time: {:?}", unfused_time);

    let fused_time = run_benchmark(size, true, iterations);
    println!("GPU Fused Time  : {:?}", fused_time);

    let speedup = unfused_time.as_secs_f64() / fused_time.as_secs_f64();
    println!("Speedup         : {:.2}x\n", speedup);
}

fn main() {
    println!("Warming up GPU backend...");
    // Initial global warmup
    let warmup_graph = Graph::new();
    let wa = warmup_graph.tensor(vec![1.0], Shape::new(vec![1, 1]));
    let wb = warmup_graph.tensor(vec![1.0], Shape::new(vec![1, 1]));
    let _ = wa.matmul(wb).relu().run(Device::Wgpu).unwrap();
    println!("Warmup complete.\n");

    if let Ok(backend) = aether::WgpuBackend::get_or_init() {
        println!(
            "Calibrated GPU Bandwidth: {:.2} GB/s",
            backend.memory_bandwidth_gbps()
        );
        println!(
            "Calibrated GPU Compute:   {:.2} GFLOPS",
            backend.compute_flops_gflops()
        );
        println!();
    }

    bench_size(512, 10);
    bench_size(1024, 10);
    bench_size(2048, 5);
}
