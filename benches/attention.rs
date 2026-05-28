use aether::{Device, Graph, Shape};
use criterion::{black_box, criterion_group, Criterion};

fn bench_attention_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/baseline");
    group.sample_size(50);

    for &seq_len in &[4, 8, 16, 32] {
        let d_model = 32;
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
        let attn = q.attention(k, v, 1.0);

        group.bench_function(format!("seq_len={}", seq_len), |bencher| {
            bencher.iter(|| {
                let out = black_box(attn.clone()).run(Device::Cpu).unwrap();
                black_box(out.data());
            });
        });
    }
    group.finish();
}

fn bench_causal_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/causal");
    group.sample_size(50);

    for &seq_len in &[4, 8, 16, 32] {
        let d_model = 32;
        let num_heads = 4;
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
        let attn = q.causal_attention(k, v, 1.0, num_heads);

        group.bench_function(format!("seq_len={}", seq_len), |bencher| {
            bencher.iter(|| {
                let out = black_box(attn.clone()).run(Device::Cpu).unwrap();
                black_box(out.data());
            });
        });
    }
    group.finish();
}

fn bench_multi_head_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/multi_head");
    group.sample_size(50);

    for &(seq_len, num_heads) in &[(4, 2), (8, 4), (16, 4), (32, 8)] {
        let d_model = num_heads * 8;
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
        let attn = q.multi_head_attention(k, v, 1.0, num_heads);

        group.bench_function(format!("seq={}_heads={}", seq_len, num_heads), |bencher| {
            bencher.iter(|| {
                let out = black_box(attn.clone()).run(Device::Cpu).unwrap();
                black_box(out.data());
            });
        });
    }
    group.finish();
}

fn bench_flash_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/flash");
    group.sample_size(50);

    for &seq_len in &[4, 8, 16, 32] {
        let d_model = 32;
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
        let attn = q.flash_attention(k, v, 1.0, false);

        group.bench_function(format!("seq_len={}", seq_len), |bencher| {
            bencher.iter(|| {
                let out = black_box(attn.clone()).run(Device::Cpu).unwrap();
                black_box(out.data());
            });
        });
    }
    group.finish();
}

fn bench_flash_attention_causal(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/flash_causal");
    group.sample_size(50);

    for &seq_len in &[4, 8, 16, 32] {
        let d_model = 32;
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
        let attn = q.flash_attention(k, v, 1.0, true);

        group.bench_function(format!("seq_len={}", seq_len), |bencher| {
            bencher.iter(|| {
                let out = black_box(attn.clone()).run(Device::Cpu).unwrap();
                black_box(out.data());
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_attention_baseline,
    bench_causal_attention,
    bench_multi_head_attention,
    bench_flash_attention,
    bench_flash_attention_causal,
);
