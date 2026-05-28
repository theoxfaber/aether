use aether::quant::matmul::matmul_q6_k;
use std::time::Instant;

fn main() {
    let shapes = [
        ("attn (4096x4096)", 1, 4096usize, 4096usize),
        ("ffn (4096x14336)", 1, 4096, 14336),
    ];

    let warmup = 3;
    let iterations = 10;

    for &(name, m, n, k) in &shapes {
        println!("\n═══ {} ({}x{}x{}) ═══", name, m, n, k);

        let a_size = m * k;
        let b_quant_size = n * k * 210 / 256; // 210 bytes per 256 weights for Q6_K
        let c_size = m * n;

        let a: Vec<f32> = (0..a_size)
            .map(|i| (i as f32) / (a_size as f32) * 2.0 - 1.0)
            .collect();
        let b_quant: Vec<u8> = (0..b_quant_size).map(|i| (i % 256) as u8).collect();
        let mut c = vec![0.0f32; c_size];

        for _ in 0..warmup {
            matmul_q6_k(&a, &b_quant, m, n, k, &mut c);
        }

        let start = Instant::now();
        for _ in 0..iterations {
            matmul_q6_k(&a, &b_quant, m, n, k, &mut c);
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let tot_sec = elapsed.as_secs_f64() / iterations as f64;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = flops / tot_sec / 1e9;
        let m_bytes = (b_quant_size as f64) / 1e6;
        let bw = m_bytes / tot_sec;

        println!(
            "  {:.3} ms/iter, {:.1} GFLOPS, {:.0} MB/s bandwidth",
            avg_ms, gflops, bw
        );
    }
}
