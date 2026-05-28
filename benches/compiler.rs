use aether::{Device, Graph, GraphCompiler, Shape};
use criterion::{black_box, criterion_group, Criterion};

fn bench_compiler_small_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiler/small_graph");
    group.sample_size(50);

    let graph = Graph::new();
    let a = graph.tensor(vec![1.0; 16], Shape::new(vec![4, 4]));
    let b = graph.tensor(vec![2.0; 16], Shape::new(vec![4, 4]));
    let c = a.matmul(b).relu().sum_all();
    let _ = c.run(Device::Cpu).unwrap();

    group.bench_function("compile_4x4_matmul_relu", |bencher| {
        bencher.iter(|| {
            let result = black_box(GraphCompiler::compile(&graph)).unwrap();
            black_box(result.graph);
        });
    });
    group.finish();
}

fn bench_compiler_medium_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiler/medium_graph");
    group.sample_size(30);

    let graph = Graph::new();
    let a = graph.tensor(vec![1.0; 256], Shape::new(vec![16, 16]));
    let b = graph.tensor(vec![2.0; 256], Shape::new(vec![16, 16]));
    let c = graph.tensor(vec![3.0; 256], Shape::new(vec![16, 16]));
    let d = a.matmul(b).relu();
    let e = d.add(c.matmul(d.clone()));
    let loss = e.sum_all();
    let _ = loss.run(Device::Cpu).unwrap();

    group.bench_function("compile_2_matmul_chain", |bencher| {
        bencher.iter(|| {
            let result = black_box(GraphCompiler::compile(&graph)).unwrap();
            black_box(result.graph);
        });
    });
    group.finish();
}

fn bench_compiler_large_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiler/large_graph");
    group.sample_size(20);

    let graph = Graph::new();
    let n = 32;
    let mut prev = graph.tensor(vec![1.0; n], Shape::new(vec![n]));
    for i in 0..10 {
        let w = graph.tensor(vec![0.5; n * n], Shape::new(vec![n, n]));
        let b = graph.tensor(vec![0.1; n], Shape::new(vec![n]));
        prev = prev.matmul(w).add(b).relu();
    }
    let loss = prev.sum_all();
    let _ = loss.run(Device::Cpu).unwrap();

    group.bench_function("compile_10_layer_mlp", |bencher| {
        bencher.iter(|| {
            let result = black_box(GraphCompiler::compile(&graph)).unwrap();
            black_box(result.graph);
        });
    });
    group.finish();
}

fn bench_compiler_per_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiler/per_pass");
    group.sample_size(30);

    // Build a graph with redundancy for CSE and constant folding opportunities
    let graph = Graph::new();
    let a = graph.tensor(vec![1.0; 64], Shape::new(vec![8, 8]));
    let b = graph.tensor(vec![2.0; 64], Shape::new(vec![8, 8]));
    let c = graph.tensor(vec![3.0; 64], Shape::new(vec![8, 8]));
    // a * b + a * b + c * 1 -> CSE should merge the two a*b, simplify should fold c*1
    let ab1 = a.matmul(b.clone());
    let ab2 = a.matmul(b);
    let c1 = c.mul(graph.tensor(vec![1.0; 64], Shape::new(vec![8, 8])));
    let out = ab1.add(ab2).add(c1).sum_all();
    let _ = out.run(Device::Cpu).unwrap();

    let compiled = graph.clone();

    group.bench_function("simplify_pass", |bencher| {
        bencher.iter(|| {
            let g = compiled.clone();
            aether::compiler::simplify::run_simplify_pass(&g).unwrap();
            black_box(g);
        });
    });

    group.bench_function("dce_pass", |bencher| {
        bencher.iter(|| {
            let g = compiled.clone();
            aether::compiler::dce::run_dce_pass(&g).unwrap();
            black_box(g);
        });
    });

    group.bench_function("cse_pass", |bencher| {
        bencher.iter(|| {
            let g = compiled.clone();
            aether::compiler::cse::run_cse_pass(&g).unwrap();
            black_box(g);
        });
    });

    group.bench_function("constant_fold_pass", |bencher| {
        bencher.iter(|| {
            let g = compiled.clone();
            aether::compiler::constant_fold::run_constant_fold_pass(&g).unwrap();
            black_box(g);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_compiler_small_graph,
    bench_compiler_medium_graph,
    bench_compiler_large_graph,
    bench_compiler_per_pass,
);
