use aether::{Device, Graph, Shape};
use criterion::{black_box, criterion_group, Criterion};

fn bench_graph_construction_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_build/construction");
    group.sample_size(100);

    group.bench_function("single_matmul", |bencher| {
        bencher.iter(|| {
            let graph = Graph::new();
            let a = graph.tensor(vec![1.0; 16], Shape::new(vec![4, 4]));
            let b = graph.tensor(vec![2.0; 16], Shape::new(vec![4, 4]));
            let _ = black_box(a.matmul(b));
        });
    });
    group.finish();
}

fn bench_graph_construction_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_build/chain_construction");
    group.sample_size(50);

    group.bench_function("10_layer_mlp", |bencher| {
        bencher.iter(|| {
            let graph = Graph::new();
            let mut prev = graph.tensor(vec![1.0; 32], Shape::new(vec![32]));
            for _ in 0..10 {
                let w = graph.tensor(vec![0.5; 32 * 32], Shape::new(vec![32, 32]));
                let b = graph.tensor(vec![0.1; 32], Shape::new(vec![32]));
                prev = prev.matmul(w).add(b).relu();
            }
            let _ = black_box(prev);
        });
    });
    group.finish();
}

fn bench_graph_construction_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_build/attention_construction");
    group.sample_size(50);

    group.bench_function("attention_graph", |bencher| {
        bencher.iter(|| {
            let graph = Graph::new();
            let q = graph.tensor(vec![1.0; 128], Shape::new(vec![1, 4, 32]));
            let k = graph.tensor(vec![1.0; 128], Shape::new(vec![1, 4, 32]));
            let v = graph.tensor(vec![1.0; 128], Shape::new(vec![1, 4, 32]));
            let _ = black_box(q.attention(k, v, 1.0));
        });
    });

    group.bench_function("flash_attention_graph", |bencher| {
        bencher.iter(|| {
            let graph = Graph::new();
            let q = graph.tensor(vec![1.0; 128], Shape::new(vec![1, 4, 32]));
            let k = graph.tensor(vec![1.0; 128], Shape::new(vec![1, 4, 32]));
            let v = graph.tensor(vec![1.0; 128], Shape::new(vec![1, 4, 32]));
            let _ = black_box(q.flash_attention(k, v, 1.0, false));
        });
    });

    group.finish();
}

fn bench_graph_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_build/clone");
    group.sample_size(50);

    let graph = Graph::new();
    let mut prev = graph.tensor(vec![1.0; 64], Shape::new(vec![8, 8]));
    for _ in 0..5 {
        let w = graph.tensor(vec![0.5; 64], Shape::new(vec![8, 8]));
        prev = prev.matmul(w).relu();
    }
    let _ = prev.run(Device::Cpu).unwrap();

    group.bench_function("clone_5_layer_graph", |bencher| {
        bencher.iter(|| {
            let cloned = black_box(graph.clone());
            black_box(cloned);
        });
    });
    group.finish();
}

fn bench_execution_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_build/execution_setup");
    group.sample_size(100);

    let graph = Graph::new();
    let a = graph.tensor(vec![1.0; 16], Shape::new(vec![4, 4]));
    let b = graph.tensor(vec![2.0; 16], Shape::new(vec![4, 4]));
    let c = a.matmul(b);

    // Measure just the run() overhead for already-built graph
    group.bench_function("run_4x4_matmul", |bencher| {
        bencher.iter(|| {
            let out = black_box(c.clone()).run(Device::Cpu).unwrap();
            black_box(out.data());
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_graph_construction_small,
    bench_graph_construction_chain,
    bench_graph_construction_attention,
    bench_graph_clone,
    bench_execution_overhead,
);
