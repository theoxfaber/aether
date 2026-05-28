use aether::loader::gguf::GGUFDtype;
use aether::quant::matmul::quantized_matmul_impl;
use aether::quant::requantize;
use criterion::{black_box, criterion_group, Criterion};

fn bench_quant_matmul_q8_0(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant_matmul/q8_0");
    group.sample_size(30);

    for &(m, n, k) in &[(1, 1024, 1024), (1, 4096, 4096)] {
        let a = vec![0.5f32; m * k];
        let b_f32 = vec![1.0f32; k * n];
        let b_raw = requantize(&b_f32, GGUFDtype::Q8_0, &[n, k]).unwrap();
        let mut c_buf = vec![0.0f32; m * n];
        let b_shape = vec![n, k];

        group.bench_function(format!("{}x{}x{}", m, n, k), |bencher| {
            bencher.iter(|| {
                c_buf.fill(0.0);
                quantized_matmul_impl(
                    black_box(&a),
                    black_box(m),
                    black_box(&b_raw),
                    black_box(&b_shape),
                    black_box(GGUFDtype::Q8_0),
                    black_box(&mut c_buf),
                    black_box(None),
                );
            });
        });
    }
    group.finish();
}

fn bench_quant_matmul_q4_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant_matmul/q4_k");
    group.sample_size(20);

    for &(m, n, k) in &[(1, 1024, 1024), (1, 4096, 4096)] {
        let a = vec![0.5f32; m * k];
        let b_f32 = vec![1.0f32; k * n];
        let b_raw = requantize(&b_f32, GGUFDtype::Q4_K, &[n, k]).unwrap();
        let mut c_buf = vec![0.0f32; m * n];
        let b_shape = vec![n, k];

        group.bench_function(format!("{}x{}x{}", m, n, k), |bencher| {
            bencher.iter(|| {
                c_buf.fill(0.0);
                quantized_matmul_impl(
                    black_box(&a),
                    black_box(m),
                    black_box(&b_raw),
                    black_box(&b_shape),
                    black_box(GGUFDtype::Q4_K),
                    black_box(&mut c_buf),
                    black_box(None),
                );
            });
        });
    }
    group.finish();
}

fn bench_quant_matmul_q6_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant_matmul/q6_k");
    group.sample_size(20);

    for &(m, n, k) in &[(1, 1024, 1024), (1, 4096, 4096)] {
        let a = vec![0.5f32; m * k];
        let b_f32 = vec![1.0f32; k * n];
        let b_raw = requantize(&b_f32, GGUFDtype::Q6_K, &[n, k]).unwrap();
        let mut c_buf = vec![0.0f32; m * n];
        let b_shape = vec![n, k];

        group.bench_function(format!("{}x{}x{}", m, n, k), |bencher| {
            bencher.iter(|| {
                c_buf.fill(0.0);
                quantized_matmul_impl(
                    black_box(&a),
                    black_box(m),
                    black_box(&b_raw),
                    black_box(&b_shape),
                    black_box(GGUFDtype::Q6_K),
                    black_box(&mut c_buf),
                    black_box(None),
                );
            });
        });
    }
    group.finish();
}

fn bench_quant_matmul_f32_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant_matmul/f32_baseline");
    group.sample_size(30);

    for &(m, n, k) in &[(1, 1024, 1024), (1, 4096, 4096)] {
        let a = vec![0.5f32; m * k];
        let b = vec![1.0f32; k * n];
        let b_shape = vec![n, k];
        let mut c_buf = vec![0.0f32; m * n];

        group.bench_function(format!("{}x{}x{}", m, n, k), |bencher| {
            bencher.iter(|| {
                c_buf.fill(0.0);
                quantized_matmul_impl(
                    black_box(&a),
                    black_box(m),
                    black_box(bytemuck::cast_slice(&b)),
                    black_box(&b_shape),
                    black_box(GGUFDtype::F32),
                    black_box(&mut c_buf),
                    black_box(None),
                );
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_quant_matmul_f32_baseline,
    bench_quant_matmul_q8_0,
    bench_quant_matmul_q4_k,
    bench_quant_matmul_q6_k,
);
