use rayon::prelude::*;
use crate::quant::matmul::common::*;

pub(crate) fn matmul_q2_k_scalar(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if m > 1 {
        return matmul_q2_k_batched_scalar(a, b_quant, m, n, k, c);
    }
    let blocks_per_row = k / Q2K_BLOCK_SIZE;
    c.par_iter_mut().enumerate().for_each(|(idx, out)| {
        let col_b = idx % n;
        let a_row = &a[..k];
        let mut acc = 0.0f32;
        let b_row_start = col_b * blocks_per_row * Q2K_BLOCK_BYTES;
        for block_idx in 0..blocks_per_row {
            let bo = b_row_start + block_idx * Q2K_BLOCK_BYTES;
            if bo + Q2K_BLOCK_BYTES > b_quant.len() {
                break;
            }
            let d = decode_f16_scale(b_quant[bo], b_quant[bo + 1]);
            let dmin = decode_f16_scale(b_quant[bo + 2], b_quant[bo + 3]);
            let scales = &b_quant[bo + 4..bo + 20];
            let qs = &b_quant[bo + 20..bo + 84];
            let elem_base = block_idx * Q2K_BLOCK_SIZE;
            for s in 0..16usize {
                let half = s / 8;
                let q_off = half * 32;
                let shift = (s as i32 / 2 % 4) * 2;
                let sc_val = scales[s];
                let dl = d * (sc_val & 0xF) as f32;
                let ml = dmin * (sc_val >> 4) as f32;
                let byte_base = (s % 2) * 16;
                let mut acc_aq = 0.0f32;
                let mut acc_a = 0.0f32;
                for e in 0..16usize {
                    let q = ((qs[q_off + byte_base + e] >> shift) & 3) as i8;
                    let a_val = a_row[elem_base + s * 16 + e];
                    acc_aq += a_val * q as f32;
                    acc_a += a_val;
                }
                acc += dl * acc_aq - ml * acc_a;
            }
        }
        *out = acc;
    });
}

pub(crate) fn matmul_q2_k_batched_scalar(
    a: &[f32],
    b_quant: &[u8],
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
) {
    let blocks_per_row = k / Q2K_BLOCK_SIZE;
    let mut flat = vec![0.0f32; m * n];
    let tasks: Vec<&mut [f32]> = flat.chunks_mut(m).collect();
    tasks
        .into_par_iter()
        .enumerate()
        .for_each(|(col_b, results)| {
            let b_row_start = col_b * blocks_per_row * Q2K_BLOCK_BYTES;
            for block_idx in 0..blocks_per_row {
                let bo = b_row_start + block_idx * Q2K_BLOCK_BYTES;
                if bo + Q2K_BLOCK_BYTES > b_quant.len() {
                    break;
                }
                let d = decode_f16_scale(b_quant[bo], b_quant[bo + 1]);
                let dmin = decode_f16_scale(b_quant[bo + 2], b_quant[bo + 3]);
                let scales = &b_quant[bo + 4..bo + 20];
                let qs = &b_quant[bo + 20..bo + 84];
                let elem_base = block_idx * Q2K_BLOCK_SIZE;
                for s in 0..16usize {
                    let half = s / 8;
                    let q_off = half * 32;
                    let shift = (s as i32 / 2 % 4) * 2;
                    let sc_val = scales[s];
                    let dl = d * (sc_val & 0xF) as f32;
                    let ml = dmin * (sc_val >> 4) as f32;
                    let byte_base = (s % 2) * 16;
                    for row_a in 0..m {
                        let act = &a[row_a * k..];
                        let mut acc_aq = 0.0f32;
                        let mut acc_a = 0.0f32;
                        for e in 0..16usize {
                            let q = ((qs[q_off + byte_base + e] >> shift) & 3) as f32;
                            let a_val = act[elem_base + s * 16 + e];
                            acc_aq += a_val * q;
                            acc_a += a_val;
                        }
                        results[row_a] += dl * acc_aq - ml * acc_a;
                    }
                }
            }
        });
    for col_b in 0..n {
        for row_a in 0..m {
            c[row_a * n + col_b] = flat[col_b * m + row_a];
        }
    }
}
