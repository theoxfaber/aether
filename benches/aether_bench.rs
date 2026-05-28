use aether::{Device, Graph, Shape};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

// ── Problem 1: GPU vs CPU — separate compute from readback ──
//
// The original benchmark timed `attn.run(Device::Wgpu)` which includes the
// GPU→CPU buffer readback (device.poll + map_async). This readback dominates
// the wall time and makes the GPU appear slower than CPU even though its
// compute kernels are faster.
//
// Fix: measure compute-only time via `run_no_readback()` (submits all GPU
// work but does NOT read back), then measure total time via `run()` which
// includes the readback. Report both.
fn memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    let size = 1024usize;
    let a_data = vec![0.5f32; size * size];
    let b_data = vec![-0.3f32; size * size];

    // CPU baseline
    group.bench_function("cpu_total", |bench| {
        bench.iter(|| {
            let graph = Graph::new();
            let a = graph.tensor(black_box(a_data.clone()), Shape::new(vec![size, size]));
            let b = graph.tensor(black_box(b_data.clone()), Shape::new(vec![size, size]));
            for _ in 0..4 {
                let w = graph.tensor(black_box(b_data.clone()), Shape::new(vec![size, size]));
                let _ = a.matmul(w).relu();
            }
            let r = a.matmul(b).run(Device::Cpu).unwrap();
            black_box(r);
        })
    });

    // GPU compute-only (no readback)
    group.bench_function("gpu_compute_only", |bench| {
        bench.iter(|| {
            let graph = Graph::new();
            let a = graph.tensor(black_box(a_data.clone()), Shape::new(vec![size, size]));
            let b = graph.tensor(black_box(b_data.clone()), Shape::new(vec![size, size]));
            for _ in 0..4 {
                let w = graph.tensor(black_box(b_data.clone()), Shape::new(vec![size, size]));
                let _ = a.matmul(w).relu();
            }
            let final_op = a.matmul(b);
            final_op.run_no_readback(Device::Wgpu).unwrap();
            black_box(());
        })
    });

    // GPU total time (includes readback)
    group.bench_function("gpu_total", |bench| {
        bench.iter(|| {
            let graph = Graph::new();
            let a = graph.tensor(black_box(a_data.clone()), Shape::new(vec![size, size]));
            let b = graph.tensor(black_box(b_data.clone()), Shape::new(vec![size, size]));
            for _ in 0..4 {
                let w = graph.tensor(black_box(b_data.clone()), Shape::new(vec![size, size]));
                let _ = a.matmul(w).relu();
            }
            let r = a.matmul(b).run(Device::Wgpu).unwrap();
            black_box(r);
        })
    });

    group.finish();
}

// ── Problem 2: Fusion making no difference ──
//
// The original benchmark on `Device::Cpu` cannot show fusion benefit because
// fused CPU kernels use the default Backend trait impl (decompose + execute
// separately). Only the Wgpu backend has single-kernel fused implementations.
//
// Additionally the original workload was `matmul → add → relu`. The fusion
// pass only fused `matmul+add` (MatMulAdd), leaving relu as a separate op.
// Month 3 tested `matmul → relu` which creates a single MatMulRelu fused op.
//
// Fix:
//   1  Use `matmul → relu` (no bias) so the pass creates MatMulRelu
//   2  Run on Wgpu where fused GPU kernels live
//   3  Size 1024×1024 so the cost model decides fusion is profitable
fn fusion_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_speedup");

    // Use 512×512 where the calibrated cost model (high bandwidth on Mac)
    // decides fusion is profitable.  At 1024×1024 the memory-bandwidth
    // savings per byte are too small relative to compute overhead.
    let size = 512usize;
    let layers = 8;
    let a_data = vec![0.5f32; size * size];
    let w_data = vec![-0.3f32; size * size];

    for fusion_enabled in &[false, true] {
        let label = if *fusion_enabled {
            "fusion_on"
        } else {
            "fusion_off"
        };
        let enabled = *fusion_enabled;
        group.bench_function(label, |bench| {
            bench.iter(|| {
                let graph = Graph::new();
                graph.enable_fusion(enabled);
                let mut x = graph.tensor(black_box(a_data.clone()), Shape::new(vec![size, size]));
                let w = graph.tensor(black_box(w_data.clone()), Shape::new(vec![size, size]));

                for _ in 0..layers {
                    x = x.matmul(w.clone()).relu();
                }

                let result = x.run(Device::Wgpu).unwrap();
                black_box(result);
            })
        });
    }
    group.finish();
}

// ── Problem 3: Memory pressure not triggering eviction ──
//
// The original benchmark used tiny 32×32 tensors on `Device::Cpu`. Since GPU
// memory tracking is only populated when `target_device == Device::Wgpu`,
// eviction_count() was always 0. The tensors were also far too small to
// exceed even the 512 MB budget.
//
// Fix:
//   1  Run on Device::Wgpu so that GPU memory tracking is active
//   2  Use 1024×1024 f32 tensors (4 MB each) with 200 parallel branches
//      (≈800 MB total) to guarantee budget exhaustion
//   3  Print eviction_count() / upload_count() for each budget level
//   4  Assert eviction_count() > 0 at 512 MB
fn pressure_graceful_degradation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pressure_graceful_degradation");
    // Each 512×512 f32 tensor = 1 MB.  With 600 parallel branches, all relu
    // outputs accumulate (≈600 MB peak) before the reduction chain consumes
    // them.  Budgets: 8 GB (no eviction), 2 GB (no eviction — peak < 2 GB),
    // 512 MB (eviction guaranteed because peak ≈600 MB > 512 MB).
    group.sample_size(10);

    let size = 512usize;
    let n_branches = 600usize;
    let x_data = vec![0.5f32; size * size];
    let w_data = vec![-0.3f32; size * size];

    let budgets: [(usize, &str); 3] = [
        (8 * 1024 * 1024 * 1024, "8GB"),
        (2 * 1024 * 1024 * 1024, "2GB"),
        (512 * 1024 * 1024, "512MB"),
    ];

    for (budget, label) in &budgets {
        let bval = *budget;
        let label_str = *label;

        group.bench_function(label_str, |bench| {
            bench.iter(|| {
                let mut graph = Graph::new();
                graph.set_gpu_memory_limit(bval);
                let input = graph.tensor(black_box(x_data.clone()), Shape::new(vec![size, size]));

                // Each branch has its own weight so that ensure_gpu fires
                // for every branch, giving the eviction mechanism a chance.
                let mut branches = Vec::with_capacity(n_branches);
                for _ in 0..n_branches {
                    let w = graph.tensor(black_box(w_data.clone()), Shape::new(vec![size, size]));
                    branches.push(input.matmul(w).relu());
                }

                let sum = branches
                    .into_iter()
                    .fold(input.clone(), |acc, b| acc.add(b));

                let result = sum.run(Device::Wgpu).unwrap();

                let evictions = graph.eviction_count();
                let uploads = graph.upload_count();
                eprintln!(
                    "[pressure_degradation {}] budget={}  evictions={}  uploads={}",
                    label_str, bval, evictions, uploads
                );

                if bval <= 512 * 1024 * 1024 {
                    assert!(
                        evictions > 0,
                        "Expected evictions at {} budget, got 0",
                        label_str
                    );
                }

                black_box(result);
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    memory_efficiency,
    fusion_speedup,
    pressure_graceful_degradation,
);
criterion_main!(benches);
