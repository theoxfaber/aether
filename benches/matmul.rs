use aether::{Device, Graph, Shape};
use criterion::{black_box, criterion_group, Criterion};

fn bench_matmul_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/small");
    group.sample_size(100);

    for &n in &[4, 8, 16, 32] {
        let graph = Graph::new();
        let a = graph.tensor(vec![1.0; n * n], Shape::new(vec![n, n]));
        let b = graph.tensor(vec![1.0; n * n], Shape::new(vec![n, n]));
        let res = a.matmul(b);

        group.bench_function(format!("{}x{}", n, n), |bencher| {
            bencher.iter(|| {
                let out = black_box(res.clone()).run(Device::Cpu).unwrap();
                black_box(out.data());
            });
        });
    }
    group.finish();
}

fn bench_matmul_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/medium");
    group.sample_size(50);

    for &n in &[64, 128, 256] {
        let graph = Graph::new();
        let a = graph.tensor(vec![0.5; n * n], Shape::new(vec![n, n]));
        let b = graph.tensor(vec![1.5; n * n], Shape::new(vec![n, n]));
        let res = a.matmul(b);

        group.bench_function(format!("{}x{}", n, n), |bencher| {
            bencher.iter(|| {
                let out = black_box(res.clone()).run(Device::Cpu).unwrap();
                black_box(out.data());
            });
        });
    }
    group.finish();
}

fn bench_matmul_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/large");
    group.sample_size(20);

    for &n in &[512, 1024] {
        let graph = Graph::new();
        let a = graph.tensor(vec![1.0; n * n], Shape::new(vec![n, n]));
        let b = graph.tensor(vec![1.0; n * n], Shape::new(vec![n, n]));
        let res = a.matmul(b);

        group.bench_function(format!("{}x{}", n, n), |bencher| {
            bencher.iter(|| {
                let out = black_box(res.clone()).run(Device::Cpu).unwrap();
                black_box(out.data());
            });
        });
    }
    group.finish();
}

fn bench_batched_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/batched");
    group.sample_size(50);

    let graph = Graph::new();
    let batch = 4;
    let m = 64;
    let k = 64;
    let n = 64;
    let lhs = graph.tensor(vec![1.0; batch * m * k], Shape::new(vec![batch, m, k]));
    let rhs = graph.tensor(vec![1.0; batch * k * n], Shape::new(vec![batch, k, n]));
    let res = lhs.batched_matmul(rhs);

    group.bench_function("4x64x64", |bencher| {
        bencher.iter(|| {
            let out = black_box(res.clone()).run(Device::Cpu).unwrap();
            black_box(out.data());
        });
    });
    group.finish();
}

fn bench_matmul_relu_fused(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/fusion");
    group.sample_size(50);

    for &n in &[16, 64] {
        let res_fused = {
            let g = Graph::new();
            g.enable_fusion(true);
            let a = g.tensor(vec![1.0; n * n], Shape::new(vec![n, n]));
            let b = g.tensor(vec![2.0; n * n], Shape::new(vec![n, n]));
            a.matmul(b).relu()
        };

        group.bench_function(format!("fused_{}x{}", n, n), |bencher| {
            bencher.iter(|| {
                let out = black_box(res_fused.clone()).run(Device::Cpu).unwrap();
                black_box(out.data());
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_small,
    bench_matmul_medium,
    bench_matmul_large,
    bench_batched_matmul,
    bench_matmul_relu_fused,
);
