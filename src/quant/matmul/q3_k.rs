use crate::quant::matmul::common::*;
use rayon::prelude::*;

pub(crate) fn matmul_q3_k_scalar(
    a: &[f32],
    b_quant: &[u8],
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
) {
    if m > 1 {
        return matmul_q3_k_batched_scalar(a, b_quant, m, n, k, c);
    }
    let blocks_per_row = k / Q3K_BLOCK_SIZE;
    c.par_iter_mut().enumerate().for_each(|(idx, out)| {
        let col_b = idx % n;
        let a_row = &a[..k];
        let mut acc = 0.0f32;
        let b_row_start = col_b * blocks_per_row * Q3K_BLOCK_BYTES;
        for block_idx in 0..blocks_per_row {
            let bo = b_row_start + block_idx * Q3K_BLOCK_BYTES;
            if bo + Q3K_BLOCK_BYTES > b_quant.len() {
                break;
            }
            let hmask = &b_quant[bo..bo + 32];
            let qs = &b_quant[bo + 32..bo + 96];
            let sr = &b_quant[bo + 96..bo + 108];
            let d = decode_f16_scale(b_quant[bo + 108], b_quant[bo + 109]);
            let elem_base = block_idx * Q3K_BLOCK_SIZE;
            let mut sc = [0i16; 16];
            for i in 0..8 {
                sc[i] = (sr[i] & 0x0F) as i16;
                sc[i + 8] = (sr[i] >> 4) as i16;
            }
            for i in 0..4 {
                let b = sr[i + 8];
                sc[i] |= (b as i16 & 0x03) << 4;
                sc[i + 4] |= ((b >> 2) as i16 & 0x03) << 4;
                sc[i + 8] |= ((b >> 4) as i16 & 0x03) << 4;
                sc[i + 12] |= ((b >> 6) as i16 & 0x03) << 4;
            }
            for s in &mut sc {
                *s -= 32;
            }
            for s in 0..16usize {
                let half = s / 8;
                let q_off = half * 32;
                let shift = (s as i32 / 2 % 4) * 2;
                let dl = d * sc[s] as f32;
                let byte_base = (s % 2) * 16;
                let hm_bit = s as u8 / 2;
                let mut acc_q = 0.0f32;
                for e in 0..16usize {
                    let ql = (qs[q_off + byte_base + e] >> shift) & 3;
                    let qh = ((hmask[byte_base + e] >> hm_bit) & 1) ^ 1;
                    let q = (ql as i8) - ((qh as i8) << 2);
                    acc_q += a_row[elem_base + s * 16 + e] * q as f32;
                }
                acc += dl * acc_q;
            }
        }
        *out = acc;
    });
}

pub(crate) fn matmul_q3_k_batched_scalar(
    a: &[f32],
    b_quant: &[u8],
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
) {
    let blocks_per_row = k / Q3K_BLOCK_SIZE;
    let mut flat = vec![0.0f32; m * n];
    let tasks: Vec<&mut [f32]> = flat.chunks_mut(m).collect();
    tasks
        .into_par_iter()
        .enumerate()
        .for_each(|(col_b, results)| {
            let b_row_start = col_b * blocks_per_row * Q3K_BLOCK_BYTES;
            for block_idx in 0..blocks_per_row {
                let bo = b_row_start + block_idx * Q3K_BLOCK_BYTES;
                if bo + Q3K_BLOCK_BYTES > b_quant.len() {
                    break;
                }
                let hmask = &b_quant[bo..bo + 32];
                let qs = &b_quant[bo + 32..bo + 96];
                let sr = &b_quant[bo + 96..bo + 108];
                let d = decode_f16_scale(b_quant[bo + 108], b_quant[bo + 109]);
                let elem_base = block_idx * Q3K_BLOCK_SIZE;
                let mut sc = [0i16; 16];
                for i in 0..8 {
                    sc[i] = (sr[i] & 0x0F) as i16;
                    sc[i + 8] = (sr[i] >> 4) as i16;
                }
                for i in 0..4 {
                    let b = sr[i + 8];
                    sc[i] |= (b as i16 & 0x03) << 4;
                    sc[i + 4] |= ((b >> 2) as i16 & 0x03) << 4;
                    sc[i + 8] |= ((b >> 4) as i16 & 0x03) << 4;
                    sc[i + 12] |= ((b >> 6) as i16 & 0x03) << 4;
                }
                for s in &mut sc {
                    *s -= 32;
                }
                for s in 0..16usize {
                    let half = s / 8;
                    let q_off = half * 32;
                    let shift = (s as i32 / 2 % 4) * 2;
                    let dl = d * sc[s] as f32;
                    let byte_base = (s % 2) * 16;
                    let hm_bit = s as u8 / 2;
                    for row_a in 0..m {
                        let act = &a[row_a * k..];
                        let mut acc_q = 0.0f32;
                        for e in 0..16usize {
                            let ql = (qs[q_off + byte_base + e] >> shift) & 3;
                            let qh = ((hmask[byte_base + e] >> hm_bit) & 1) ^ 1;
                            let q = (ql as i8) - ((qh as i8) << 2);
                            acc_q += act[elem_base + s * 16 + e] * q as f32;
                        }
                        results[row_a] += dl * acc_q;
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
