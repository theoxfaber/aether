use crate::quant::matmul::common::*;
use rayon::prelude::*;

#[allow(dead_code)]
pub(crate) fn matmul_q6_k_scalar(
    a: &[f32],
    b_quant: &[u8],
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
) {
    if m > 1 {
        return matmul_q6_k_batched_scalar(a, b_quant, m, n, k, c);
    }
    let blocks_per_row = k / Q6K_BLOCK_SIZE;

    c.par_iter_mut().enumerate().for_each(|(idx, out)| {
        let col_b = idx % n;
        let a_row = &a[..k];
        let mut acc = 0.0f32;
        let b_row_start = col_b * blocks_per_row * Q6K_BLOCK_BYTES;

        for block_idx in 0..blocks_per_row {
            let bo = b_row_start + block_idx * Q6K_BLOCK_BYTES;
            if bo + Q6K_BLOCK_BYTES > b_quant.len() {
                break;
            }

            let d = decode_f16_scale(b_quant[bo + 208], b_quant[bo + 209]);

            let ql = &b_quant[bo..bo + 128];
            let qh = &b_quant[bo + 128..bo + 192];
            let sc = &b_quant[bo + 192..bo + 208];

            let elem_base = block_idx * Q6K_BLOCK_SIZE;

            for half in 0..2 {
                let ql_off = half * 64;
                let qh_off = half * 32;
                let sc_off = half * 8;

                for l in (0..32).step_by(16) {
                    let is = l / 16;
                    let s1_val = sc[sc_off + is] as i8 as f32;
                    let s2_val = sc[sc_off + is + 2] as i8 as f32;
                    let s3_val = sc[sc_off + is + 4] as i8 as f32;
                    let s4_val = sc[sc_off + is + 6] as i8 as f32;

                    let ds1 = d * s1_val;
                    let ds2 = d * s2_val;
                    let ds3 = d * s3_val;
                    let ds4 = d * s4_val;

                    let hbase = elem_base + half * 128 + l;

                    for idx in 0..16 {
                        let ql_val = ql[ql_off + l + idx];
                        let ql_val32 = ql[ql_off + l + 32 + idx];
                        let qh_val = qh[qh_off + l + idx];

                        let q1_low = ql_val & 0x0F;
                        let q1_high = (qh_val & 0x03) << 4;
                        let q1 = (q1_low | q1_high) as i8 - 32;

                        let q2_low = ql_val32 & 0x0F;
                        let q2_high = (qh_val & 0x0C) << 2;
                        let q2 = (q2_low | q2_high) as i8 - 32;

                        let q3_low = ql_val >> 4;
                        let q3_high = qh_val & 0x30;
                        let q3 = (q3_low | q3_high) as i8 - 32;

                        let q4_low = ql_val32 >> 4;
                        let q4_high = (qh_val & 0xC0) >> 2;
                        let q4 = (q4_low | q4_high) as i8 - 32;

                        let a_val1 = a_row[hbase + idx];
                        let a_val2 = a_row[hbase + 32 + idx];
                        let a_val3 = a_row[hbase + 64 + idx];
                        let a_val4 = a_row[hbase + 96 + idx];

                        acc += a_val1 * (ds1 * q1 as f32);
                        acc += a_val2 * (ds2 * q2 as f32);
                        acc += a_val3 * (ds3 * q3 as f32);
                        acc += a_val4 * (ds4 * q4 as f32);
                    }
                }
            }
        }
        *out = acc;
    });
}

#[allow(dead_code)]
fn matmul_q6_k_batched_scalar(
    a: &[f32],
    b_quant: &[u8],
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
) {
    let blocks_per_row = k / Q6K_BLOCK_SIZE;
    let mut flat_results = vec![0.0f32; m * n];

    let tasks: Vec<&mut [f32]> = flat_results.chunks_mut(m).collect();
    tasks
        .into_par_iter()
        .enumerate()
        .for_each(|(col_b, results)| {
            let b_row_start = col_b * blocks_per_row * Q6K_BLOCK_BYTES;

            for block_idx in 0..blocks_per_row {
                let bo = b_row_start + block_idx * Q6K_BLOCK_BYTES;
                if bo + Q6K_BLOCK_BYTES > b_quant.len() {
                    break;
                }

                let d = decode_f16_scale(b_quant[bo + 208], b_quant[bo + 209]);

                let ql = &b_quant[bo..bo + 128];
                let qh = &b_quant[bo + 128..bo + 192];
                let sc = &b_quant[bo + 192..bo + 208];

                let elem_base = block_idx * Q6K_BLOCK_SIZE;

                for half in 0..2 {
                    let ql_off = half * 64;
                    let qh_off = half * 32;
                    let sc_off = half * 8;

                    for l in (0..32).step_by(16) {
                        let is = l / 16;
                        let s1_val = sc[sc_off + is] as i8 as f32;
                        let s2_val = sc[sc_off + is + 2] as i8 as f32;
                        let s3_val = sc[sc_off + is + 4] as i8 as f32;
                        let s4_val = sc[sc_off + is + 6] as i8 as f32;

                        let ds1 = d * s1_val;
                        let ds2 = d * s2_val;
                        let ds3 = d * s3_val;
                        let ds4 = d * s4_val;

                        let hbase = elem_base + half * 128 + l;

                        let mut q1_arr = [0i8; 16];
                        let mut q2_arr = [0i8; 16];
                        let mut q3_arr = [0i8; 16];
                        let mut q4_arr = [0i8; 16];

                        for idx in 0..16 {
                            let ql_val = ql[ql_off + l + idx];
                            let ql_val32 = ql[ql_off + l + 32 + idx];
                            let qh_val = qh[qh_off + l + idx];

                            q1_arr[idx] = ((ql_val & 0x0F) | ((qh_val & 0x03) << 4)) as i8 - 32;
                            q2_arr[idx] = ((ql_val32 & 0x0F) | ((qh_val & 0x0C) << 2)) as i8 - 32;
                            q3_arr[idx] = ((ql_val >> 4) | (qh_val & 0x30)) as i8 - 32;
                            q4_arr[idx] = ((ql_val32 >> 4) | ((qh_val & 0xC0) >> 2)) as i8 - 32;
                        }

                        for row_a in 0..m {
                            let act = &a[row_a * k..];
                            let mut acc1 = 0.0f32;
                            let mut acc2 = 0.0f32;
                            let mut acc3 = 0.0f32;
                            let mut acc4 = 0.0f32;

                            for idx in 0..16 {
                                let a_idx = hbase + idx;
                                acc1 += act[a_idx] * q1_arr[idx] as f32;
                                acc2 += act[a_idx + 32] * q2_arr[idx] as f32;
                                acc3 += act[a_idx + 64] * q3_arr[idx] as f32;
                                acc4 += act[a_idx + 96] * q4_arr[idx] as f32;
                            }

                            results[row_a] += ds1 * acc1 + ds2 * acc2 + ds3 * acc3 + ds4 * acc4;
                        }
                    }
                }
            }
        });

    for col_b in 0..n {
        for row_a in 0..m {
            c[row_a * n + col_b] = flat_results[col_b * m + row_a];
        }
    }
}
