use criterion::criterion_main;

mod attention;
mod compiler;
mod graph_build;
mod matmul;
mod mixed_precision;
mod quant_matmul;

criterion_main!(
    matmul::benches,
    attention::benches,
    mixed_precision::benches,
    compiler::benches,
    graph_build::benches,
    quant_matmul::benches,
);
