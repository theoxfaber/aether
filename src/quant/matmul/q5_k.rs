use rayon::prelude::*;
use crate::quant::matmul::common::*;

#[allow(dead_code)]
pub(crate) fn matmul_q5_k_scalar(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if m > 1 {
        return matmul_q5_k_batched_scalar(a, b_quant, m, n, k, c);
    }
    let blocks_per_row = k / Q5K_BLOCK_SIZE;

    c.par_iter_mut().enumerate().for_each(|(idx, out)| {
        let col_b = idx % n;
        let a_row = &a[..k];
        let mut acc = 0.0f32;
        let b_row_start = col_b * blocks_per_row * Q5K_BLOCK_BYTES;

        for block_idx in 0..blocks_per_row {
            let bo = b_row_start + block_idx * Q5K_BLOCK_BYTES;
            if bo + Q5K_BLOCK_BYTES > b_quant.len() {
                break;
            }

            let d = decode_f16_scale(b_quant[bo], b_quant[bo + 1]);
            let dmin = decode_f16_scale(b_quant[bo + 2], b_quant[bo + 3]);
            let scales = &b_quant[bo + 4..bo + 16];
            let qh = &b_quant[bo + 16..bo + 48];
            let qs = &b_quant[bo + 48..bo + 176];

            let elem_base = block_idx * Q5K_BLOCK_SIZE;
            let mut qs_ptr = 0usize;
            let mut is = 0usize;

            for _j in (0..Q5K_BLOCK_SIZE).step_by(64) {
                let (sc0, mm0) = get_scale_min_k4(is, scales);
                let d1 = d * sc0 as f32;
                let m1 = dmin * mm0 as f32;
                let (sc1, mm1) = get_scale_min_k4(is + 1, scales);
                let d2 = d * sc1 as f32;
                let m2 = dmin * mm1 as f32;

                for l in 0..32 {
                    let byte_l = qs[qs_ptr + l];
                    let ql_lo = (byte_l & 0x0F) as f32;
                    let ql_hi = ((byte_l >> 4) & 0x0F) as f32;
                    let qh_byte_lo = qh[(is * 32 + l) / 8];
                    let qh_bit_lo = ((qh_byte_lo >> ((is * 32 + l) % 8)) & 1) as f32;
                    let qh_byte_hi = qh[((is + 1) * 32 + l) / 8];
                    let qh_bit_hi = ((qh_byte_hi >> (((is + 1) * 32 + l) % 8)) & 1) as f32;

                    let q_lo = ql_lo + qh_bit_lo * 16.0;
                    let q_hi = ql_hi + qh_bit_hi * 16.0;

                    let global_lo = elem_base + is * 32 + l;
                    let global_hi = elem_base + (is + 1) * 32 + l;

                    if global_lo < k {
                        acc += a_row[global_lo] * (d1 * q_lo - m1);
                    }
                    if global_hi < k {
                        acc += a_row[global_hi] * (d2 * q_hi - m2);
                    }
                }
                qs_ptr += 32;
                is += 2;
            }
        }
        *out = acc;
    });
}

#[allow(dead_code)]
pub(crate) fn matmul_q5_k_batched_scalar(
    a: &[f32],
    b_quant: &[u8],
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
) {
    let blocks_per_row = k / Q5K_BLOCK_SIZE;
    let mut flat_results = vec![0.0f32; m * n];

    let tasks: Vec<&mut [f32]> = flat_results.chunks_mut(m).collect();
    tasks
        .into_par_iter()
        .enumerate()
        .for_each(|(col_b, results)| {
            let b_row_start = col_b * blocks_per_row * Q5K_BLOCK_BYTES;

            for block_idx in 0..blocks_per_row {
                let bo = b_row_start + block_idx * Q5K_BLOCK_BYTES;
                if bo + Q5K_BLOCK_BYTES > b_quant.len() {
                    break;
                }

                let d = decode_f16_scale(b_quant[bo], b_quant[bo + 1]);
                let dmin = decode_f16_scale(b_quant[bo + 2], b_quant[bo + 3]);
                let scales = &b_quant[bo + 4..bo + 16];
                let qh = &b_quant[bo + 16..bo + 48];
                let qs = &b_quant[bo + 48..bo + 176];

                let mut qs_ptr = 0usize;
                let mut is = 0usize;

                for _j in (0..Q5K_BLOCK_SIZE).step_by(64) {
                    let (sc0, mm0) = get_scale_min_k4(is, scales);
                    let d1 = d * sc0 as f32;
                    let m1 = dmin * mm0 as f32;
                    let (sc1, mm1) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * sc1 as f32;
                    let m2 = dmin * mm1 as f32;

                    for row_a in 0..m {
                        let act = &a[row_a * k..];
                        let mut acc_lo = 0.0f32;
                        let mut acc_a_lo = 0.0f32;
                        let mut acc_hi = 0.0f32;
                        let mut acc_a_hi = 0.0f32;

                        for l in 0..32 {
                            let byte_l = qs[qs_ptr + l];
                            let ql_lo = (byte_l & 0x0F) as f32;
                            let ql_hi = ((byte_l >> 4) & 0x0F) as f32;
                            let qh_byte_lo = qh[(is * 32 + l) / 8];
                            let qh_bit_lo = ((qh_byte_lo >> ((is * 32 + l) % 8)) & 1) as f32;
                            let qh_byte_hi = qh[((is + 1) * 32 + l) / 8];
                            let qh_bit_hi = ((qh_byte_hi >> (((is + 1) * 32 + l) % 8)) & 1) as f32;

                            let q_lo = ql_lo + qh_bit_lo * 16.0;
                            let q_hi = ql_hi + qh_bit_hi * 16.0;
                            let a_val_lo = act[block_idx * Q5K_BLOCK_SIZE + is * 32 + l];
                            let a_val_hi = act[block_idx * Q5K_BLOCK_SIZE + (is + 1) * 32 + l];

                            acc_lo += a_val_lo * q_lo;
                            acc_a_lo += a_val_lo;
                            acc_hi += a_val_hi * q_hi;
                            acc_a_hi += a_val_hi;
                        }
                        results[row_a] += d1 * acc_lo - m1 * acc_a_lo;
                        results[row_a] += d2 * acc_hi - m2 * acc_a_hi;
                    }
                    qs_ptr += 32;
                    is += 2;
                }
            }
        });

    for col_b in 0..n {
        for row_a in 0..m {
            c[row_a * n + col_b] = flat_results[col_b * m + row_a];
        }
    }
}
