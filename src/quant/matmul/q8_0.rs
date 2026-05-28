use rayon::prelude::*;
use crate::quant::matmul::common::*;

#[allow(dead_code)]
pub(crate) fn matmul_q8_0_scalar(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if m > 1 {
        return matmul_q8_0_batched_scalar(a, b_quant, m, n, k, c);
    }
    let blocks_per_row = k.div_ceil(Q8_BLOCK_SIZE);

    c.par_iter_mut().enumerate().for_each(|(idx, out)| {
        let col_b = idx % n;
        let a_row = &a[..k];
        let mut acc = 0.0f32;
        let b_row_start = col_b * blocks_per_row * Q8_BLOCK_BYTES;

        for block_idx in 0..blocks_per_row {
            let bo = b_row_start + block_idx * Q8_BLOCK_BYTES;
            if bo + Q8_BLOCK_BYTES > b_quant.len() {
                break;
            }
            let d = half::f16::from_le_bytes([b_quant[bo], b_quant[bo + 1]]).to_f32();
            let elem_base = block_idx * Q8_BLOCK_SIZE;
            let elem_count = Q8_BLOCK_SIZE.min(k - elem_base);
            let quant = &b_quant[bo + 2..bo + 2 + elem_count];

            let mut sum = 0.0f32;
            for i in 0..elem_count {
                sum += a_row[elem_base + i] * quant[i] as i8 as f32;
            }
            acc += d * sum;
        }
        *out = acc;
    });
}

#[allow(dead_code)]
fn matmul_q8_0_batched_scalar(
    a: &[f32],
    b_quant: &[u8],
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
) {
    let blocks_per_row = k.div_ceil(Q8_BLOCK_SIZE);
    let mut flat_results = vec![0.0f32; m * n];

    let tasks: Vec<&mut [f32]> = flat_results.chunks_mut(m).collect();
    tasks
        .into_par_iter()
        .enumerate()
        .for_each(|(col_b, results)| {
            let b_row_start = col_b * blocks_per_row * Q8_BLOCK_BYTES;

            for block_idx in 0..blocks_per_row {
                let bo = b_row_start + block_idx * Q8_BLOCK_BYTES;
                if bo + Q8_BLOCK_BYTES > b_quant.len() {
                    break;
                }
                let d = half::f16::from_le_bytes([b_quant[bo], b_quant[bo + 1]]).to_f32();
                let elem_base = block_idx * Q8_BLOCK_SIZE;
                let elem_count = Q8_BLOCK_SIZE.min(k - elem_base);
                let quant = &b_quant[bo + 2..bo + 2 + elem_count];

                for row_a in 0..m {
                    let act = &a[row_a * k..];
                    let mut sum = 0.0f32;
                    for i in 0..elem_count {
                        sum += act[elem_base + i] * quant[i] as i8 as f32;
                    }
                    results[row_a] += d * sum;
                }
            }
        });

    for col_b in 0..n {
        for row_a in 0..m {
            c[row_a * n + col_b] = flat_results[col_b * m + row_a];
        }
    }
}
