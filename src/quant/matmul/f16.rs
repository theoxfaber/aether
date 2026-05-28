use rayon::prelude::*;

pub(crate) fn matmul_f16_scalar(a: &[f32], b_bytes: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if m > 1 {
        return matmul_f16_batched_scalar(a, b_bytes, m, n, k, c);
    }
    c.par_iter_mut().enumerate().for_each(|(idx, out)| {
        let col_b = idx % n;
        let a_row = &a[..k];
        let mut acc = 0.0f32;
        let b_row_start = col_b * k * 2;
        for i in 0..k {
            let bo = b_row_start + i * 2;
            let v = half::f16::from_le_bytes([b_bytes[bo], b_bytes[bo + 1]]).to_f32();
            acc += a_row[i] * v;
        }
        *out = acc;
    });
}

fn matmul_f16_batched_scalar(
    a: &[f32],
    b_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
) {
    let mut flat_results = vec![0.0f32; m * n];
    let tasks: Vec<&mut [f32]> = flat_results.chunks_mut(m).collect();
    tasks
        .into_par_iter()
        .enumerate()
        .for_each(|(col_b, results)| {
            let b_row_start = col_b * k * 2;
            for i in 0..k {
                let bo = b_row_start + i * 2;
                let w = half::f16::from_le_bytes([b_bytes[bo], b_bytes[bo + 1]]).to_f32();
                for row_a in 0..m {
                    results[row_a] += a[row_a * k + i] * w;
                }
            }
        });
    for col_b in 0..n {
        for row_a in 0..m {
            c[row_a * n + col_b] = flat_results[col_b * m + row_a];
        }
    }
}
