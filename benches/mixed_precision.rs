use aether::{AnyData, Device, Dtype, GradScaler, Graph, Shape};
use criterion::{black_box, criterion_group, Criterion};

fn bench_cast_f32_to_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_precision/cast_f32_f16");
    group.sample_size(50);

    for &n in &[64, 256, 1024, 4096] {
        let graph = Graph::new();
        let a = graph.tensor(vec![1.5; n], Shape::new(vec![n]));
        let casted = a.cast(Dtype::F16);

        group.bench_function(format!("n={}", n), |bencher| {
            bencher.iter(|| {
                let out = black_box(casted.clone()).run(Device::Cpu).unwrap();
                black_box(out.data_raw());
            });
        });
    }
    group.finish();
}

fn bench_cast_f16_to_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_precision/cast_f16_f32");
    group.sample_size(50);

    for &n in &[64, 256, 1024, 4096] {
        let graph = Graph::new();
        let data: Vec<half::f16> = vec![half::f16::from_f32(1.5); n];
        let a = graph.tensor_with_data(AnyData::F16(data), Shape::new(vec![n]), Dtype::F16);
        let casted = a.cast(Dtype::F32);

        group.bench_function(format!("n={}", n), |bencher| {
            bencher.iter(|| {
                let out = black_box(casted.clone()).run(Device::Cpu).unwrap();
                black_box(out.data());
            });
        });
    }
    group.finish();
}

fn bench_cast_f32_to_bf16(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_precision/cast_f32_bf16");
    group.sample_size(50);

    for &n in &[64, 256, 1024, 4096] {
        let graph = Graph::new();
        let a = graph.tensor(vec![1.5; n], Shape::new(vec![n]));
        let casted = a.cast(Dtype::BF16);

        group.bench_function(format!("n={}", n), |bencher| {
            bencher.iter(|| {
                let out = black_box(casted.clone()).run(Device::Cpu).unwrap();
                black_box(out.data_raw());
            });
        });
    }
    group.finish();
}

fn bench_grad_scaler_scale_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_precision/grad_scaler");
    group.sample_size(50);

    let scaler = GradScaler::default();
    let graph = Graph::new();
    let a = graph.tensor(vec![1.0; 256], Shape::new(vec![16, 16]));
    let b = graph.tensor(vec![2.0; 256], Shape::new(vec![16, 16]));
    let loss = a.matmul(b).sum_all();

    group.bench_function("scale_loss", |bencher| {
        bencher.iter(|| {
            let scaled = black_box(scaler.scale_loss(loss.clone()));
            let out = black_box(scaled).run(Device::Cpu).unwrap();
            black_box(out.data());
        });
    });
    group.finish();
}

fn bench_grad_scaler_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_precision/update");
    group.sample_size(100);

    let mut scaler = GradScaler::default();

    group.bench_function("update_finite", |bencher| {
        bencher.iter(|| {
            black_box(scaler.update(true));
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_cast_f32_to_f16,
    bench_cast_f16_to_f32,
    bench_cast_f32_to_bf16,
    bench_grad_scaler_scale_loss,
    bench_grad_scaler_update,
);
