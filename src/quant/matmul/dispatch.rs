use crate::loader::dequant::dequantize;
use crate::loader::gguf::GGUFDtype;
use crate::quant::matmul::f16;
use crate::quant::matmul::neon;
use crate::quant::matmul::q2_k;
use crate::quant::matmul::q3_k;
use rayon::prelude::*;

// ── Q5_K dispatch ─────────────────────────────────────────────────────

pub fn matmul_q5_k(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    matmul_q5_k_impl(a, b, m, n, k, c);
}

#[cfg(target_arch = "aarch64")]
fn matmul_q5_k_impl(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        neon::matmul_q5_k_dotprod(a, b, m, n, k, c);
    } else {
        neon::matmul_q5_k(a, b, m, n, k, c);
    }
}
#[cfg(not(target_arch = "aarch64"))]
fn matmul_q5_k_impl(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return unsafe { crate::quant::avx2::matmul_q5_k(a, b, m, n, k, c) };
    }
    crate::quant::matmul::q5_k::matmul_q5_k_scalar(a, b, m, n, k, c);
}

// ── Q2_K dispatch ────────────────────────────────────────────────────

pub fn matmul_q2_k(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    matmul_q2_k_impl(a, b, m, n, k, c);
}

fn matmul_q2_k_impl(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return unsafe { crate::quant::avx2::matmul_q2_k(a, b, m, n, k, c) };
    }
    q2_k::matmul_q2_k_scalar(a, b, m, n, k, c);
}

// ── Q3_K dispatch ────────────────────────────────────────────────────

pub fn matmul_q3_k(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    matmul_q3_k_impl(a, b, m, n, k, c);
}

fn matmul_q3_k_impl(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return unsafe { crate::quant::avx2::matmul_q3_k(a, b, m, n, k, c) };
    }
    q3_k::matmul_q3_k_scalar(a, b, m, n, k, c);
}

// ── F16 dispatch ──────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub(crate) fn matmul_f16_impl(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if std::arch::is_aarch64_feature_detected!("fp16") {
        neon::matmul_f16_neon(a, b, m, n, k, c);
    } else {
        f16::matmul_f16_scalar(a, b, m, n, k, c);
    }
}
#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn matmul_f16_impl(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("f16c") {
        return unsafe { crate::quant::avx2::matmul_f16(a, b, m, n, k, c) };
    }
    f16::matmul_f16_scalar(a, b, m, n, k, c);
}

pub fn matmul_f16(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    matmul_f16_impl(a, b, m, n, k, c);
}

// ─── Convenience: auto-dispatch by dtype ────────────────────────────

/// Compute C = A × B^T where B is stored in any quantized format.
///
/// For Q8_0, Q4_K, and Q6_K, uses the fused inline dequant kernel
/// (with NEON SIMD on aarch64, falling back if needed).
/// For all other formats, falls back to dequantize → f32 matmul.
///
/// A:      [M × K] f32 activations (K = in_features)
/// b_raw:  raw quantized bytes for B, stored GGUF [in, out] row-major
/// shape:  [N, K] = [out, in] — weight matrix shape (reversed from GGUF)
pub fn quantized_matmul_impl(
    a: &[f32],
    m: usize,
    b_raw: &[u8],
    b_shape: &[usize],
    b_dtype: GGUFDtype,
    c: &mut [f32],
    b_f32_cache: Option<&[f32]>,
) {
    debug_assert_eq!(b_shape.len(), 2);
    let n = b_shape[0];
    let k = b_shape[1];
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(c.len(), m * n);

    if let Some(cached) = b_f32_cache {
        c.fill(0.0f32);
        matmul_f32_gguf(a, cached, m, n, k, c);
        return;
    }

    // Try registered kernel first; fall back to builtin dispatch.
    if super::registry::dispatch_matmul(a, m, b_raw, n, k, b_dtype, c) {
        return;
    }

    match b_dtype {
        GGUFDtype::Q8_0 => crate::quant::matmul::avx2_dispatch::matmul_q8_0(a, b_raw, m, n, k, c),
        GGUFDtype::Q4_K => crate::quant::matmul::avx2_dispatch::matmul_q4_k(a, b_raw, m, n, k, c),
        GGUFDtype::Q5_K => matmul_q5_k(a, b_raw, m, n, k, c),
        GGUFDtype::Q6_K => crate::quant::matmul::avx2_dispatch::matmul_q6_k(a, b_raw, m, n, k, c),
        _ => {
            let b_f32 = dequantize(b_raw, b_dtype, &[k, n]);
            c.fill(0.0f32);
            matmul_f32_gguf(a, &b_f32, m, n, k, c);
        }
    }
}

/// Re-quantize f32 data back to a quantized format.
/// Used to transpose quantized weights at load time.
pub fn requantize(data: &[f32], dtype: GGUFDtype, shape: &[usize]) -> Result<Vec<u8>, String> {
    match dtype {
        GGUFDtype::Q8_0 => Ok(requantize_q8_0(data, shape)),
        GGUFDtype::Q4_K => Ok(requantize_q4_k(data, shape)),
        GGUFDtype::Q5_K => Ok(requantize_q5_k(data, shape)),
        GGUFDtype::Q6_K => Ok(requantize_q6_k(data, shape)),
        _ => Err(format!("requantize not implemented for {:?}", dtype)),
    }
}

fn requantize_q8_0(data: &[f32], shape: &[usize]) -> Vec<u8> {
    let rows = shape[0];
    let cols = shape[1];
    let block_size = 32;
    let blocks_per_row = cols.div_ceil(block_size);
    let total_blocks = rows * blocks_per_row;
    let mut out = vec![0u8; total_blocks * 34];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let start = row * cols + blk * block_size;
            let end = std::cmp::min(start + block_size, row * cols + cols);
            let actual_block = end - start;

            let block = &data[start..start + actual_block];
            let off = (row * blocks_per_row + blk) * 34;

            let max_val = block.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
            let d = if max_val == 0.0 { 1.0 } else { max_val / 127.0 };
            let d_bits = half::f16::from_f32(d).to_le_bytes();
            out[off] = d_bits[0];
            out[off + 1] = d_bits[1];

            for j in 0..block_size {
                let idx = start + j;
                let q = if j < actual_block {
                    (data[idx] / d).round().max(-128.0).min(127.0) as i8 as u8
                } else {
                    0
                };
                out[off + 2 + j] = q;
            }
        }
    }
    out
}

fn requantize_q4_k(data: &[f32], shape: &[usize]) -> Vec<u8> {
    let rows = shape[0];
    let cols = shape[1];
    let block_size = 256;
    let blocks_per_row = cols.div_ceil(block_size);
    let total_blocks = rows * blocks_per_row;
    let mut out = vec![0u8; total_blocks * 144];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let start = row * cols + blk * block_size;
            let end = std::cmp::min(start + block_size, row * cols + cols);
            let actual_block = end - start;

            let block = &data[start..start + actual_block];
            let off = (row * blocks_per_row + blk) * 144;

            let mut sub_mins = [0.0f32; 8];
            let mut sub_maxs = [0.0f32; 8];
            for sb in 0..8 {
                let sb_start = sb * 32;
                let sb_end = std::cmp::min(sb_start + 32, actual_block);
                if sb_start < actual_block {
                    let sb_slice = &block[sb_start..sb_end];
                    let mut mn = sb_slice[0];
                    let mut mx = sb_slice[0];
                    for &x in sb_slice {
                        mn = mn.min(x);
                        mx = mx.max(x);
                    }
                    sub_mins[sb] = mn;
                    sub_maxs[sb] = mx;
                }
            }

            let mut m_subs = [0.0f32; 8];
            let mut d_subs = [0.0f32; 8];
            for sb in 0..8 {
                m_subs[sb] = sub_mins[sb].min(0.0).abs();
                d_subs[sb] = (sub_maxs[sb] + m_subs[sb]) / 15.0;
            }

            let max_d = d_subs.iter().copied().fold(0.0f32, f32::max);
            let max_m = m_subs.iter().copied().fold(0.0f32, f32::max);

            let d = if max_d == 0.0 { 1.0 } else { max_d / 63.0 };
            let dmin = if max_m == 0.0 { 1.0 } else { max_m / 63.0 };

            let d_bits = half::f16::from_f32(d).to_le_bytes();
            let dmin_bits = half::f16::from_f32(dmin).to_le_bytes();
            out[off] = d_bits[0];
            out[off + 1] = d_bits[1];
            out[off + 2] = dmin_bits[0];
            out[off + 3] = dmin_bits[1];

            let mut sc = [0u8; 8];
            let mut mm = [0u8; 8];
            for sb in 0..8 {
                sc[sb] = (d_subs[sb] / d).round().max(0.0).min(63.0) as u8;
                mm[sb] = (m_subs[sb] / dmin).round().max(0.0).min(63.0) as u8;
            }

            for k in 0..4 {
                let sc_high = sc[k + 4] >> 4;
                let mm_high = mm[k + 4] >> 4;
                out[off + 4 + k] = (sc[k] & 63) | (sc_high << 6);
                out[off + 8 + k] = (mm[k] & 63) | (mm_high << 6);
                let sc_low = sc[k + 4] & 0x0F;
                let mm_low = mm[k + 4] & 0x0F;
                out[off + 12 + k] = sc_low | (mm_low << 4);
            }

            for chunk in 0..4 {
                let is = chunk * 2;
                let sc0 = sc[is] as f32;
                let mm0 = mm[is] as f32;
                let sc1 = sc[is + 1] as f32;
                let mm1 = mm[is + 1] as f32;

                let d1 = d * sc0;
                let m1 = dmin * mm0;
                let d2 = d * sc1;
                let m2 = dmin * mm1;

                for l in 0..32 {
                    let idx0 = is * 32 + l;
                    let idx1 = (is + 1) * 32 + l;

                    let q0 = if idx0 < actual_block {
                        if d1 == 0.0 {
                            0
                        } else {
                            ((block[idx0] + m1) / d1).round().max(0.0).min(15.0) as u8
                        }
                    } else {
                        0
                    };

                    let q1 = if idx1 < actual_block {
                        if d2 == 0.0 {
                            0
                        } else {
                            ((block[idx1] + m2) / d2).round().max(0.0).min(15.0) as u8
                        }
                    } else {
                        0
                    };

                    out[off + 16 + chunk * 32 + l] = q0 | (q1 << 4);
                }
            }
        }
    }
    out
}

fn requantize_q5_k(data: &[f32], shape: &[usize]) -> Vec<u8> {
    let rows = shape[0];
    let cols = shape[1];
    let block_size = 256;
    let blocks_per_row = cols.div_ceil(block_size);
    let total_blocks = rows * blocks_per_row;
    let mut out = vec![0u8; total_blocks * 176];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let start = row * cols + blk * block_size;
            let end = std::cmp::min(start + block_size, row * cols + cols);
            let actual_block = end - start;

            let block = &data[start..start + actual_block];
            let off = (row * blocks_per_row + blk) * 176;

            let mut sub_mins = [0.0f32; 8];
            let mut sub_maxs = [0.0f32; 8];
            for sb in 0..8 {
                let sb_start = sb * 32;
                let sb_end = std::cmp::min(sb_start + 32, actual_block);
                if sb_start < actual_block {
                    let sb_slice = &block[sb_start..sb_end];
                    let mut mn = sb_slice[0];
                    let mut mx = sb_slice[0];
                    for &x in sb_slice {
                        mn = mn.min(x);
                        mx = mx.max(x);
                    }
                    sub_mins[sb] = mn;
                    sub_maxs[sb] = mx;
                }
            }

            let mut m_subs = [0.0f32; 8];
            let mut d_subs = [0.0f32; 8];
            for sb in 0..8 {
                m_subs[sb] = sub_mins[sb].min(0.0).abs();
                d_subs[sb] = (sub_maxs[sb] + m_subs[sb]) / 31.0;
            }

            let max_d = d_subs.iter().copied().fold(0.0f32, f32::max);
            let max_m = m_subs.iter().copied().fold(0.0f32, f32::max);

            let d = if max_d == 0.0 { 1.0 } else { max_d / 63.0 };
            let dmin = if max_m == 0.0 { 1.0 } else { max_m / 63.0 };

            let d_bits = half::f16::from_f32(d).to_le_bytes();
            let dmin_bits = half::f16::from_f32(dmin).to_le_bytes();
            out[off] = d_bits[0];
            out[off + 1] = d_bits[1];
            out[off + 2] = dmin_bits[0];
            out[off + 3] = dmin_bits[1];

            let mut sc = [0u8; 8];
            let mut mm = [0u8; 8];
            for sb in 0..8 {
                sc[sb] = (d_subs[sb] / d).round().max(0.0).min(63.0) as u8;
                mm[sb] = (m_subs[sb] / dmin).round().max(0.0).min(63.0) as u8;
            }

            for k in 0..4 {
                let sc_high = sc[k + 4] >> 4;
                let mm_high = mm[k + 4] >> 4;
                out[off + 4 + k] = (sc[k] & 63) | (sc_high << 6);
                out[off + 8 + k] = (mm[k] & 63) | (mm_high << 6);
                let sc_low = sc[k + 4] & 0x0F;
                let mm_low = mm[k + 4] & 0x0F;
                out[off + 12 + k] = sc_low | (mm_low << 4);
            }

            for chunk in 0..4 {
                let is = chunk * 2;
                let sc0 = sc[is] as f32;
                let mm0 = mm[is] as f32;
                let sc1 = sc[is + 1] as f32;
                let mm1 = mm[is + 1] as f32;

                let d1 = d * sc0;
                let m1 = dmin * mm0;
                let d2 = d * sc1;
                let m2 = dmin * mm1;

                for l in 0..32 {
                    let idx0 = is * 32 + l;
                    let idx1 = (is + 1) * 32 + l;

                    let q0 = if idx0 < actual_block {
                        if d1 == 0.0 {
                            0
                        } else {
                            ((block[idx0] + m1) / d1).round().max(0.0).min(31.0) as i32
                        }
                    } else {
                        0
                    };

                    let q1 = if idx1 < actual_block {
                        if d2 == 0.0 {
                            0
                        } else {
                            ((block[idx1] + m2) / d2).round().max(0.0).min(31.0) as i32
                        }
                    } else {
                        0
                    };

                    let quant0 = q0 as u8;
                    let quant1 = q1 as u8;
                    let ql_lo = quant0 & 0x0F;
                    let qh_bit_lo = quant0 >> 4;
                    let ql_hi = quant1 & 0x0F;
                    let qh_bit_hi = quant1 >> 4;

                    out[off + 48 + chunk * 32 + l] = ql_lo | (ql_hi << 4);

                    let bit_idx_lo = is * 32 + l;
                    out[off + 16 + bit_idx_lo / 8] |= qh_bit_lo << (bit_idx_lo % 8);

                    let bit_idx_hi = (is + 1) * 32 + l;
                    out[off + 16 + bit_idx_hi / 8] |= qh_bit_hi << (bit_idx_hi % 8);
                }
            }
        }
    }
    out
}

fn requantize_q6_k(data: &[f32], shape: &[usize]) -> Vec<u8> {
    let rows = shape[0];
    let cols = shape[1];
    let block_size = 256;
    let blocks_per_row = cols.div_ceil(block_size);
    let total_blocks = rows * blocks_per_row;
    let mut out = vec![0u8; total_blocks * 210];

    for row in 0..rows {
        for blk in 0..blocks_per_row {
            let start = row * cols + blk * block_size;
            let end = std::cmp::min(start + block_size, row * cols + cols);
            let actual_block = end - start;

            let block = &data[start..start + actual_block];
            let off = (row * blocks_per_row + blk) * 210;

            let mut sub_max_abs = [0.0f32; 16];
            for sb in 0..16 {
                let sb_start = sb * 16;
                let sb_end = std::cmp::min(sb_start + 16, actual_block);
                if sb_start < actual_block {
                    let sb_slice = &block[sb_start..sb_end];
                    let mut mx = sb_slice[0].abs();
                    for &x in sb_slice {
                        mx = mx.max(x.abs());
                    }
                    sub_max_abs[sb] = mx;
                }
            }

            let max_val = sub_max_abs.iter().copied().fold(0.0f32, f32::max);
            let d = if max_val == 0.0 {
                1.0
            } else {
                max_val / (32.0 * 127.0)
            };

            let mut sc = [0u8; 16];
            for sb in 0..16 {
                sc[sb] = (sub_max_abs[sb] / (32.0 * d)).round().max(0.0).min(127.0) as u8;
            }

            for sb in 0..16 {
                out[off + 192 + sb] = sc[sb];
            }

            let d_bits = half::f16::from_f32(d).to_le_bytes();
            out[off + 208] = d_bits[0];
            out[off + 209] = d_bits[1];

            for half in 0..2 {
                let ql_off = half * 64;
                let qh_off = half * 32;
                let sc_off = half * 8;

                for l in 0..32 {
                    let is = l / 16;
                    let s1 = sc[sc_off + is ] as f32;
                    let s2 = sc[sc_off + is + 2] as f32;
                    let s3 = sc[sc_off + is + 4] as f32;
                    let s4 = sc[sc_off + is + 6] as f32;

                    let base = half * 128;
                    let idx1 = base + l;
                    let idx2 = base + l + 32;
                    let idx3 = base + l + 64;
                    let idx4 = base + l + 96;

                    let q1 = if idx1 < actual_block && s1 > 0.0 {
                        (block[idx1] / (d * s1)).round().max(-32.0).min(31.0) as i32
                    } else {
                        0
                    };
                    let q2 = if idx2 < actual_block && s2 > 0.0 {
                        (block[idx2] / (d * s2)).round().max(-32.0).min(31.0) as i32
                    } else {
                        0
                    };
                    let q3 = if idx3 < actual_block && s3 > 0.0 {
                        (block[idx3] / (d * s3)).round().max(-32.0).min(31.0) as i32
                    } else {
                        0
                    };
                    let q4 = if idx4 < actual_block && s4 > 0.0 {
                        (block[idx4] / (d * s4)).round().max(-32.0).min(31.0) as i32
                    } else {
                        0
                    };

                    let q1_u = (q1 + 32) as u8;
                    let q2_u = (q2 + 32) as u8;
                    let q3_u = (q3 + 32) as u8;
                    let q4_u = (q4 + 32) as u8;

                    out[off + ql_off + l] = (q1_u & 0x0F) | ((q3_u & 0x0F) << 4);

                    out[off + ql_off + l + 32] = (q2_u & 0x0F) | ((q4_u & 0x0F) << 4);

                    out[off + 128 + qh_off + l] =
                        (q1_u >> 4) | ((q2_u >> 4) << 2) | ((q3_u >> 4) << 4) | ((q4_u >> 4) << 6);
                }
            }
        }
    }
    out
}

// ─── f32 GGUF-order matmul (for asymmetric weights) ──────────────────

/// F32 matmul for weight data in GGUF [k, n] = [in, out] row-major order.
/// b is [k, n] = [in, out].  Computes C[M×N] = A[M×K] × B[K×N] by
/// reading b[i·n + j] (input-major).
fn matmul_f32_gguf(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if m > 1 {
        return matmul_f32_gguf_batched(a, b, m, n, k, c);
    }
    c.fill(0.0f32);
    for i in 0..k {
        let scale = a[i];
        if scale != 0.0 {
            for j in 0..n {
                c[j] = scale.mul_add(b[i * n + j], c[j]);
            }
        }
    }
}

fn matmul_f32_gguf_batched(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, c: &mut [f32]) {
    for i in 0..k {
        for row in 0..m {
            let scale = a[row * k + i];
            if scale != 0.0 {
                let c_row = &mut c[row * n..(row + 1) * n];
                for j in 0..n {
                    c_row[j] = scale.mul_add(b[i * n + j], c_row[j]);
                }
            }
        }
    }
}

// ─── f32 fallback (standard [n, k] order) ────────────────────────────

/// F32 matmul for weight data in [n, k] row-major order (standard).
/// Computes C[M×N] = A[M×K] × B[N×K]^T  (i.e. y = x @ W^T).
/// b is [n, k] = [out, in]; reads b[j * k + i].
pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if m > 1 {
        return matmul_f32_batched(a, b, m, n, k, c);
    }
    c.par_iter_mut().enumerate().for_each(|(idx, out)| {
        let col_b = idx % n;
        let b_row = &b[col_b * k..(col_b + 1) * k];
        let a_row = &a[..k];
        let mut acc = 0.0f32;
        for i in 0..k {
            acc += a_row[i] * b_row[i];
        }
        *out = acc;
    });
}

fn matmul_f32_batched(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, c: &mut [f32]) {
    let mut flat_results = vec![0.0f32; m * n];

    let tasks: Vec<&mut [f32]> = flat_results.chunks_mut(m).collect();
    tasks
        .into_par_iter()
        .enumerate()
        .for_each(|(col_b, results)| {
            let b_row = &b[col_b * k..(col_b + 1) * k];
            for i in 0..k {
                let w = b_row[i];
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

/// Add bias in-place: c[i] += bias[i % bias.len()]
pub fn add_bias(c: &mut [f32], bias: &[f32], n: usize) {
    for row in c.chunks_mut(n) {
        for (j, val) in row.iter_mut().enumerate() {
            *val += bias[j];
        }
    }
}
