#![allow(unsafe_code)]
use crate::quant::matmul::common::*;
use rayon::prelude::*;

use std::arch::aarch64::*;

#[inline(always)]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    vaddvq_f32(v)
}

#[inline(always)]
unsafe fn int8x16_to_f32x4_4(q: int8x16_t) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
    let low_s8 = vget_low_s8(q);
    let high_s8 = vget_high_s8(q);
    let low_s16 = vmovl_s8(low_s8);
    let high_s16 = vmovl_s8(high_s8);
    let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(low_s16)));
    let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(low_s16)));
    let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(high_s16)));
    let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(high_s16)));
    (f0, f1, f2, f3)
}

#[inline(always)]
unsafe fn my_vdotq_s32(mut acc: int32x4_t, b: int8x16_t, c: int8x16_t) -> int32x4_t {
    std::arch::asm!(
        "sdot {acc:v}.4s, {b:v}.16b, {c:v}.16b",
        acc = inout(vreg) acc,
        b = in(vreg) b,
        c = in(vreg) c,
        options(pure, nomem, nostack)
    );
    acc
}

#[inline(always)]
unsafe fn sum_i8x16(v: int8x16_t) -> i32 {
    let sum16 = vpaddlq_s8(v);
    let sum32 = vpaddlq_s16(sum16);
    vaddvq_s32(sum32)
}

#[inline(always)]
unsafe fn quantize_q8_0_block(src: *const f32, dst: *mut u8) {
    let v0 = vld1q_f32(src);
    let v1 = vld1q_f32(src.add(4));
    let v2 = vld1q_f32(src.add(8));
    let v3 = vld1q_f32(src.add(12));
    let v4 = vld1q_f32(src.add(16));
    let v5 = vld1q_f32(src.add(20));
    let v6 = vld1q_f32(src.add(24));
    let v7 = vld1q_f32(src.add(28));

    let abs0 = vabsq_f32(v0);
    let abs1 = vabsq_f32(v1);
    let abs2 = vabsq_f32(v2);
    let abs3 = vabsq_f32(v3);
    let abs4 = vabsq_f32(v4);
    let abs5 = vabsq_f32(v5);
    let abs6 = vabsq_f32(v6);
    let abs7 = vabsq_f32(v7);

    let max0 = vmaxq_f32(abs0, abs1);
    let max1 = vmaxq_f32(abs2, abs3);
    let max2 = vmaxq_f32(abs4, abs5);
    let max3 = vmaxq_f32(abs6, abs7);
    let max01 = vmaxq_f32(max0, max1);
    let max23 = vmaxq_f32(max2, max3);
    let max_all = vmaxq_f32(max01, max23);
    let max_t = vpmaxq_f32(max_all, max_all);
    let max_val = vgetq_lane_f32::<0>(vpmaxq_f32(max_t, max_t));

    let d = max_val / 127.0;
    let d_f16 = half::f16::from_f32(d);
    let d_bytes = d_f16.to_le_bytes();
    dst.write(d_bytes[0]);
    dst.add(1).write(d_bytes[1]);

    if d > 0.0 {
        let inv_d = vdupq_n_f32(1.0 / d);
        let q0 = vcvtnq_s32_f32(vmulq_f32(v0, inv_d));
        let q1 = vcvtnq_s32_f32(vmulq_f32(v1, inv_d));
        let q2 = vcvtnq_s32_f32(vmulq_f32(v2, inv_d));
        let q3 = vcvtnq_s32_f32(vmulq_f32(v3, inv_d));
        let q4 = vcvtnq_s32_f32(vmulq_f32(v4, inv_d));
        let q5 = vcvtnq_s32_f32(vmulq_f32(v5, inv_d));
        let q6 = vcvtnq_s32_f32(vmulq_f32(v6, inv_d));
        let q7 = vcvtnq_s32_f32(vmulq_f32(v7, inv_d));

        let q01_16 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        let q23_16 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        let q45_16 = vcombine_s16(vqmovn_s32(q4), vqmovn_s32(q5));
        let q67_16 = vcombine_s16(vqmovn_s32(q6), vqmovn_s32(q7));

        let q0123_8 = vcombine_s8(vqmovn_s16(q01_16), vqmovn_s16(q23_16));
        let q4567_8 = vcombine_s8(vqmovn_s16(q45_16), vqmovn_s16(q67_16));

        vst1q_s8(dst.add(2) as *mut i8, q0123_8);
        vst1q_s8(dst.add(18) as *mut i8, q4567_8);
    } else {
        std::ptr::write_bytes(dst.add(2), 0, 32);
    }
}

pub(crate) fn quantize_activations_q8_0(a: &[f32], k: usize) -> Vec<u8> {
    let blocks = k.div_ceil(32);
    let mut a_quant = vec![0u8; blocks * 34];
    // SAFETY: `a` has length `k`, so `a.as_ptr().add(i)` is valid for all
    // `i` where `i + 32 <= k`. The full and partial block loops guarantee
    // `(i / 32) * 34 + 34 <= blocks * 34 == a_quant.len()`, so the output
    // pointer is also in-bounds. `buf` is a 32-element stack array, passed
    // as a valid 32-element source to `quantize_q8_0_block`.
    unsafe {
        let mut i = 0;
        while i + 32 <= k {
            quantize_q8_0_block(a.as_ptr().add(i), a_quant.as_mut_ptr().add((i / 32) * 34));
            i += 32;
        }
        if i < k {
            let rem = k - i;
            let mut buf = [0.0f32; 32];
            std::ptr::copy_nonoverlapping(a.as_ptr().add(i), buf.as_mut_ptr(), rem);
            quantize_q8_0_block(buf.as_ptr(), a_quant.as_mut_ptr().add((i / 32) * 34));
        }
    }
    a_quant
}

pub(crate) fn quantize_activations_q8_0_batched(a: &[f32], m: usize, k: usize) -> Vec<u8> {
    let blocks = k.div_ceil(32);
    let mut a_quant = vec![0u8; m * blocks * 34];
    // SAFETY: `a` has length `m * k`, so `row_src = a.as_ptr().add(r * k)` is
    // valid for all `0 <= r < m`. `a_quant` was allocated with
    // `m * blocks * 34` bytes, so `row_dst = a_quant.as_mut_ptr().add(r * blocks * 34)`
    // is in-bounds. Inner pointer arithmetic follows `quantize_activations_q8_0`.
    unsafe {
        for r in 0..m {
            let row_src = a.as_ptr().add(r * k);
            let row_dst = a_quant.as_mut_ptr().add(r * blocks * 34);
            let mut i = 0;
            while i + 32 <= k {
                quantize_q8_0_block(row_src.add(i), row_dst.add((i / 32) * 34));
                i += 32;
            }
            if i < k {
                let rem = k - i;
                let mut buf = [0.0f32; 32];
                std::ptr::copy_nonoverlapping(row_src.add(i), buf.as_mut_ptr(), rem);
                quantize_q8_0_block(buf.as_ptr(), row_dst.add((i / 32) * 34));
            }
        }
    }
    a_quant
}

// ── Q4_K single-row (M=1) ────────────────────────────────────────

pub fn matmul_q4_k(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if m > 1 {
        return matmul_q4_k_batched(a, b_quant, m, n, k, c);
    }
    let blocks_per_row = k / Q4K_BLOCK_SIZE;

    c.par_iter_mut().enumerate().for_each(|(idx, out)| {
        let col_b = idx % n;
        let a_row = &a[..k];
        let mut acc = 0.0f32;
        let b_row_start = col_b * blocks_per_row * Q4K_BLOCK_BYTES;

        for block_idx in 0..blocks_per_row {
            let bo = b_row_start + block_idx * Q4K_BLOCK_BYTES;
            let d = decode_f16_scale(b_quant[bo], b_quant[bo + 1]);
            let dmin = decode_f16_scale(b_quant[bo + 2], b_quant[bo + 3]);
            let scales = &b_quant[bo + 4..bo + 16];
            let qs = &b_quant[bo + 16..bo + 144];
            let elem_base = block_idx * Q4K_BLOCK_SIZE;

            let mut qs_ptr = 0usize;
            let mut is = 0usize;

            for _j in (0..Q4K_BLOCK_SIZE).step_by(64) {
                let (sc0, mm0) = get_scale_min_k4(is, scales);
                let d1 = d * sc0 as f32;
                let m1 = dmin * mm0 as f32;
                let (sc1, mm1) = get_scale_min_k4(is + 1, scales);
                let d2 = d * sc1 as f32;
                let m2 = dmin * mm1 as f32;

                let base_lo = elem_base + is * 32;
                let base_hi = elem_base + (is + 1) * 32;

                // SAFETY: `qs` is a 128-byte slice (`Q4K_BLOCK_BYTES - 16`),
                // so `qs.as_ptr().add(qs_ptr + batch)` is valid for all
                // batches in `0..32` because `qs_ptr` advances by at most
                // 32 and the outer loop runs at most 4 times, consuming
                // the full 128 bytes. `a_row` has length `k` and
                // `base_lo + batch + 4 < k` because `base_lo` is bounded
                // by `elem_base + 32 < Q4K_BLOCK_SIZE` and `k` is a
                // multiple of `Q4K_BLOCK_SIZE`. NEON `vld1_s8` reads 8
                // bytes and `vld1q_f32` reads 16 bytes; alignment is
                // guaranteed by the slice implementation.
                unsafe {
                    let mut acc_aq_lo = vdupq_n_f32(0.0);
                    let mut acc_a_lo = vdupq_n_f32(0.0);
                    let mut acc_aq_hi = vdupq_n_f32(0.0);
                    let mut acc_a_hi = vdupq_n_f32(0.0);

                    for batch in (0..32).step_by(8) {
                        let bp = qs.as_ptr().add(qs_ptr + batch);
                        let ap_lo = a_row.as_ptr().add(base_lo + batch);
                        let ap_hi = a_row.as_ptr().add(base_hi + batch);

                        let bytes = vld1_s8(bp as *const i8);
                        let ubytes = vreinterpret_u8_s8(bytes);
                        let lo = vand_u8(ubytes, vdup_n_u8(0x0F));
                        let hi = vshr_n_u8(ubytes, 4);

                        let lo_u16 = vmovl_u8(lo);
                        let lo_lo_f = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_u16)));
                        let lo_hi_f = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_u16)));

                        let hi_u16 = vmovl_u8(hi);
                        let hi_lo_f = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_u16)));
                        let hi_hi_f = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_u16)));

                        let a_lo_0 = vld1q_f32(ap_lo);
                        let a_lo_1 = vld1q_f32(ap_lo.add(4));

                        acc_aq_lo = vmlaq_f32(acc_aq_lo, lo_lo_f, a_lo_0);
                        acc_aq_lo = vmlaq_f32(acc_aq_lo, lo_hi_f, a_lo_1);
                        acc_a_lo = vaddq_f32(acc_a_lo, a_lo_0);
                        acc_a_lo = vaddq_f32(acc_a_lo, a_lo_1);

                        let a_hi_0 = vld1q_f32(ap_hi);
                        let a_hi_1 = vld1q_f32(ap_hi.add(4));

                        acc_aq_hi = vmlaq_f32(acc_aq_hi, hi_lo_f, a_hi_0);
                        acc_aq_hi = vmlaq_f32(acc_aq_hi, hi_hi_f, a_hi_1);
                        acc_a_hi = vaddq_f32(acc_a_hi, a_hi_0);
                        acc_a_hi = vaddq_f32(acc_a_hi, a_hi_1);
                    }

                    qs_ptr += 32;
                    is += 2;

                    acc += d1 * hsum_f32x4(acc_aq_lo) - m1 * hsum_f32x4(acc_a_lo);
                    acc += d2 * hsum_f32x4(acc_aq_hi) - m2 * hsum_f32x4(acc_a_hi);
                }
            }
        }
        *out = acc;
    });
}

// ── Q4_K batched (M > 1) ──────────────────────────────────────────

fn matmul_q4_k_batched(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    let blocks_per_row = k / Q4K_BLOCK_SIZE;
    let mut flat_results = vec![0.0f32; m * n];

    let tasks: Vec<&mut [f32]> = flat_results.chunks_mut(m).collect();
    tasks
        .into_par_iter()
        .enumerate()
        .for_each(|(col_b, results)| {
            let b_row_start = col_b * blocks_per_row * Q4K_BLOCK_BYTES;

            for block_idx in 0..blocks_per_row {
                let bo = b_row_start + block_idx * Q4K_BLOCK_BYTES;
                let d = decode_f16_scale(b_quant[bo], b_quant[bo + 1]);
                let dmin = decode_f16_scale(b_quant[bo + 2], b_quant[bo + 3]);
                let scales = &b_quant[bo + 4..bo + 16];
                let qs = &b_quant[bo + 16..bo + 144];
                let elem_base = block_idx * Q4K_BLOCK_SIZE;

                let mut qs_ptr = 0usize;
                let mut is = 0usize;

                for _j in (0..Q4K_BLOCK_SIZE).step_by(64) {
                    let (sc0, mm0) = get_scale_min_k4(is, scales);
                    let d1 = d * sc0 as f32;
                    let m1 = dmin * mm0 as f32;
                    let (sc1, mm1) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * sc1 as f32;
                    let m2 = dmin * mm1 as f32;

                    let base_lo = elem_base + is * 32;
                    let base_hi = elem_base + (is + 1) * 32;

                    let mut lo_f32 = [0.0f32; 32];
                    let mut hi_f32 = [0.0f32; 32];
                    for l in 0..32 {
                        let byte = qs[qs_ptr + l];
                        lo_f32[l] = (byte & 0x0F) as f32;
                        hi_f32[l] = ((byte >> 4) & 0x0F) as f32;
                    }

                    for row_a in 0..m {
                        let act = &a[row_a * k..];
                        // SAFETY: `act` has at least `k` elements and
                        // `base_lo + batch + 4 < k` because the indices are
                        // bounded by `Q4K_BLOCK_SIZE`. `lo_f32` and `hi_f32`
                        // are 32-element stack arrays; `lo_f32.as_ptr().add(batch)`
                        // and `lo_f32.as_ptr().add(batch+4)` are valid for
                        // `vld1q_f32`. All NEON loads/stores use properly
                        // aligned pointers from Rust slices or stack arrays.
                        unsafe {
                            let mut acc_aq_lo = vdupq_n_f32(0.0);
                            let mut acc_a_lo = vdupq_n_f32(0.0);
                            let mut acc_aq_hi = vdupq_n_f32(0.0);
                            let mut acc_a_hi = vdupq_n_f32(0.0);

                            for batch in (0..32).step_by(8) {
                                let ap_lo = act.as_ptr().add(base_lo + batch);
                                let ap_hi = act.as_ptr().add(base_hi + batch);

                                let ql_0 = vld1q_f32(lo_f32.as_ptr().add(batch));
                                let ql_1 = vld1q_f32(lo_f32.as_ptr().add(batch + 4));
                                let qh_0 = vld1q_f32(hi_f32.as_ptr().add(batch));
                                let qh_1 = vld1q_f32(hi_f32.as_ptr().add(batch + 4));

                                let a_lo_0 = vld1q_f32(ap_lo);
                                let a_lo_1 = vld1q_f32(ap_lo.add(4));
                                let a_hi_0 = vld1q_f32(ap_hi);
                                let a_hi_1 = vld1q_f32(ap_hi.add(4));

                                acc_aq_lo = vmlaq_f32(acc_aq_lo, ql_0, a_lo_0);
                                acc_aq_lo = vmlaq_f32(acc_aq_lo, ql_1, a_lo_1);
                                acc_a_lo = vaddq_f32(acc_a_lo, a_lo_0);
                                acc_a_lo = vaddq_f32(acc_a_lo, a_lo_1);

                                acc_aq_hi = vmlaq_f32(acc_aq_hi, qh_0, a_hi_0);
                                acc_aq_hi = vmlaq_f32(acc_aq_hi, qh_1, a_hi_1);
                                acc_a_hi = vaddq_f32(acc_a_hi, a_hi_0);
                                acc_a_hi = vaddq_f32(acc_a_hi, a_hi_1);
                            }

                            results[row_a] +=
                                d1 * hsum_f32x4(acc_aq_lo) - m1 * hsum_f32x4(acc_a_lo);
                            results[row_a] +=
                                d2 * hsum_f32x4(acc_aq_hi) - m2 * hsum_f32x4(acc_a_hi);
                        }
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

// ── Q4_K dotprod kernels using vdotq_s32 ─────────────────────────

pub fn matmul_q4_k_dotprod(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if m > 1 {
        return matmul_q4_k_batched_dotprod(a, b_quant, m, n, k, c);
    }
    let blocks_per_row = k / Q4K_BLOCK_SIZE;
    let a_quant = quantize_activations_q8_0(a, k);

    c.par_iter_mut().enumerate().for_each(|(idx, out)| {
        let col_b = idx % n;
        let mut acc = 0.0f32;
        let b_row_start = col_b * blocks_per_row * Q4K_BLOCK_BYTES;

        for block_idx in 0..blocks_per_row {
            let bo = b_row_start + block_idx * Q4K_BLOCK_BYTES;
            let d = decode_f16_scale(b_quant[bo], b_quant[bo + 1]);
            let dmin = decode_f16_scale(b_quant[bo + 2], b_quant[bo + 3]);
            let scales = &b_quant[bo + 4..bo + 16];
            let qs = &b_quant[bo + 16..bo + 144];

            let mut qs_ptr = 0usize;
            let mut is = 0usize;

            for _j in (0..Q4K_BLOCK_SIZE).step_by(64) {
                let (sc0, mm0) = get_scale_min_k4(is, scales);
                let d1 = d * sc0 as f32;
                let m1 = dmin * mm0 as f32;
                let (sc1, mm1) = get_scale_min_k4(is + 1, scales);
                let d2 = d * sc1 as f32;
                let m2 = dmin * mm1 as f32;

                let b_lo = block_idx * 8 + is;
                let b_hi = block_idx * 8 + is + 1;

                let ao_lo = b_lo * Q8_BLOCK_BYTES;
                let ao_hi = b_hi * Q8_BLOCK_BYTES;

                let d_a1 = half::f16::from_le_bytes([a_quant[ao_lo], a_quant[ao_lo + 1]]).to_f32();
                let d_a2 = half::f16::from_le_bytes([a_quant[ao_hi], a_quant[ao_hi + 1]]).to_f32();

                // SAFETY: `qs.as_ptr().add(qs_ptr)` and `qs.as_ptr().add(qs_ptr + 16)`
                // point into the 128-byte Q4_K quant buffer; `qs_ptr` advances
                // by at most 32 per iteration and the outer loop runs at most
                // 4 times. `ao_lo` and `ao_hi` are byte offsets into `a_quant`,
                // which was allocated with `blocks_per_row * Q8_BLOCK_BYTES` bytes
                // (or `m * blocks_per_row * 8 * Q8_BLOCK_BYTES` for batched).
                // `ao_lo + 2 + 16` is within that allocation because the block
                // index is bounded by `block_idx < blocks_per_row`.
                // All `vld1q_s8` calls require 16-byte aligned pointers; the
                // slice-derived pointers satisfy this.
                unsafe {
                    let bp = qs.as_ptr().add(qs_ptr);
                    let w_bytes0 = vld1q_s8(bp as *const i8);
                    let w_bytes1 = vld1q_s8(bp.add(16) as *const i8);

                    let q_lo0 = vandq_s8(w_bytes0, vdupq_n_s8(0x0F));
                    let q_lo1 = vandq_s8(w_bytes1, vdupq_n_s8(0x0F));

                    let u_w0 = vreinterpretq_u8_s8(w_bytes0);
                    let u_w1 = vreinterpretq_u8_s8(w_bytes1);
                    let q_hi0 = vreinterpretq_s8_u8(vshrq_n_u8(u_w0, 4));
                    let q_hi1 = vreinterpretq_s8_u8(vshrq_n_u8(u_w1, 4));

                    let act_lo0 = vld1q_s8(a_quant.as_ptr().add(ao_lo + 2) as *const i8);
                    let act_lo1 = vld1q_s8(a_quant.as_ptr().add(ao_lo + 2 + 16) as *const i8);

                    let act_hi0 = vld1q_s8(a_quant.as_ptr().add(ao_hi + 2) as *const i8);
                    let act_hi1 = vld1q_s8(a_quant.as_ptr().add(ao_hi + 2 + 16) as *const i8);

                    let mut dot_lo = vdupq_n_s32(0);
                    dot_lo = my_vdotq_s32(dot_lo, act_lo0, q_lo0);
                    dot_lo = my_vdotq_s32(dot_lo, act_lo1, q_lo1);

                    let mut dot_hi = vdupq_n_s32(0);
                    dot_hi = my_vdotq_s32(dot_hi, act_hi0, q_hi0);
                    dot_hi = my_vdotq_s32(dot_hi, act_hi1, q_hi1);

                    let sum_lo = (sum_i8x16(act_lo0) + sum_i8x16(act_lo1)) as f32;
                    let sum_hi = (sum_i8x16(act_hi0) + sum_i8x16(act_hi1)) as f32;

                    acc += d_a1 * (d1 * (vaddvq_s32(dot_lo) as f32) - m1 * sum_lo);
                    acc += d_a2 * (d2 * (vaddvq_s32(dot_hi) as f32) - m2 * sum_hi);
                }

                qs_ptr += 32;
                is += 2;
            }
        }
        *out = acc;
    });
}

pub fn matmul_q4_k_batched_dotprod(
    a: &[f32],
    b_quant: &[u8],
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
) {
    let blocks_per_row = k / Q4K_BLOCK_SIZE;
    let a_quant = quantize_activations_q8_0_batched(a, m, k);
    let mut flat_results = vec![0.0f32; m * n];

    let tasks: Vec<&mut [f32]> = flat_results.chunks_mut(m).collect();
    tasks
        .into_par_iter()
        .enumerate()
        .for_each(|(col_b, results)| {
            let b_row_start = col_b * blocks_per_row * Q4K_BLOCK_BYTES;

            for block_idx in 0..blocks_per_row {
                let bo = b_row_start + block_idx * Q4K_BLOCK_BYTES;
                let d = decode_f16_scale(b_quant[bo], b_quant[bo + 1]);
                let dmin = decode_f16_scale(b_quant[bo + 2], b_quant[bo + 3]);
                let scales = &b_quant[bo + 4..bo + 16];
                let qs = &b_quant[bo + 16..bo + 144];

                let mut qs_ptr = 0usize;
                let mut is = 0usize;

                for _j in (0..Q4K_BLOCK_SIZE).step_by(64) {
                    let (sc0, mm0) = get_scale_min_k4(is, scales);
                    let d1 = d * sc0 as f32;
                    let m1 = dmin * mm0 as f32;
                    let (sc1, mm1) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * sc1 as f32;
                    let m2 = dmin * mm1 as f32;

                    let b_lo = block_idx * 8 + is;
                    let b_hi = block_idx * 8 + is + 1;

                    // SAFETY: Same weight pointer invariants as `matmul_q4_k_dotprod`.
                    // `a_quant` was allocated by `quantize_activations_q8_0_batched`
                    // with `m * blocks_per_row * 8 * Q8_BLOCK_BYTES` bytes.
                    // `ao_lo` and `ao_hi` for each `row_a` are computed from
                    // `(row_a * blocks_per_row * 8 + b_lo) * Q8_BLOCK_BYTES`,
                    // which never exceeds `m * blocks_per_row * 8 * Q8_BLOCK_BYTES`
                    // because `b_lo < blocks_per_row * 8`.
                    unsafe {
                        let bp = qs.as_ptr().add(qs_ptr);
                        let w_bytes0 = vld1q_s8(bp as *const i8);
                        let w_bytes1 = vld1q_s8(bp.add(16) as *const i8);

                        let q_lo0 = vandq_s8(w_bytes0, vdupq_n_s8(0x0F));
                        let q_lo1 = vandq_s8(w_bytes1, vdupq_n_s8(0x0F));

                        let u_w0 = vreinterpretq_u8_s8(w_bytes0);
                        let u_w1 = vreinterpretq_u8_s8(w_bytes1);
                        let q_hi0 = vreinterpretq_s8_u8(vshrq_n_u8(u_w0, 4));
                        let q_hi1 = vreinterpretq_s8_u8(vshrq_n_u8(u_w1, 4));

                        for row_a in 0..m {
                            let ao_lo = (row_a * blocks_per_row * 8 + b_lo) * Q8_BLOCK_BYTES;
                            let ao_hi = (row_a * blocks_per_row * 8 + b_hi) * Q8_BLOCK_BYTES;

                            let d_a1 =
                                half::f16::from_le_bytes([a_quant[ao_lo], a_quant[ao_lo + 1]])
                                    .to_f32();
                            let d_a2 =
                                half::f16::from_le_bytes([a_quant[ao_hi], a_quant[ao_hi + 1]])
                                    .to_f32();

                            let act_lo0 = vld1q_s8(a_quant.as_ptr().add(ao_lo + 2) as *const i8);
                            let act_lo1 =
                                vld1q_s8(a_quant.as_ptr().add(ao_lo + 2 + 16) as *const i8);

                            let act_hi0 = vld1q_s8(a_quant.as_ptr().add(ao_hi + 2) as *const i8);
                            let act_hi1 =
                                vld1q_s8(a_quant.as_ptr().add(ao_hi + 2 + 16) as *const i8);

                            let mut dot_lo = vdupq_n_s32(0);
                            dot_lo = my_vdotq_s32(dot_lo, act_lo0, q_lo0);
                            dot_lo = my_vdotq_s32(dot_lo, act_lo1, q_lo1);

                            let mut dot_hi = vdupq_n_s32(0);
                            dot_hi = my_vdotq_s32(dot_hi, act_hi0, q_hi0);
                            dot_hi = my_vdotq_s32(dot_hi, act_hi1, q_hi1);

                            let sum_lo = (sum_i8x16(act_lo0) + sum_i8x16(act_lo1)) as f32;
                            let sum_hi = (sum_i8x16(act_hi0) + sum_i8x16(act_hi1)) as f32;

                            results[row_a] +=
                                d_a1 * (d1 * (vaddvq_s32(dot_lo) as f32) - m1 * sum_lo);
                            results[row_a] +=
                                d_a2 * (d2 * (vaddvq_s32(dot_hi) as f32) - m2 * sum_hi);
                        }
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

// ── Q6_K single-row (M=1) ─────────────────────────────────────────

pub fn matmul_q6_k(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if m > 1 {
        return matmul_q6_k_batched(a, b_quant, m, n, k, c);
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

            // SAFETY: `ql` (128 bytes), `qh` (64 bytes), and `sc` (16 bytes)
            // are slices of the Q6_K super-block, validated by the bounds
            // check `bo + Q6K_BLOCK_BYTES <= b_quant.len()`. The loop indices
            // are bounded: `half < 2`, `l` steps by 16 from 0..32, and
            // `ql_off + l + 32 + 16 <= 128`, `qh_off + l + 16 <= 64`,
            // `sc_off + is + 6 < 16`. `a_row` has length `k` and
            // `hbase + 96 + 12 < k` because the offset never exceeds one
            // block of `Q6K_BLOCK_SIZE` and `k` is a multiple thereof.
            unsafe {
                let mut acc_v = vdupq_n_f32(0.0);

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

                        let ds1 = vdupq_n_f32(d * s1_val);
                        let ds2 = vdupq_n_f32(d * s2_val);
                        let ds3 = vdupq_n_f32(d * s3_val);
                        let ds4 = vdupq_n_f32(d * s4_val);

                        let v_ql_l = vld1q_s8(ql.as_ptr().add(ql_off + l) as *const i8);
                        let v_ql_l32 = vld1q_s8(ql.as_ptr().add(ql_off + l + 32) as *const i8);
                        let v_qh_l = vld1q_s8(qh.as_ptr().add(qh_off + l) as *const i8);

                        let q1_low = vandq_s8(v_ql_l, vdupq_n_s8(0x0F));
                        let q1_high = vshlq_n_s8(vandq_s8(v_qh_l, vdupq_n_s8(0x03)), 4);
                        let q1 = vsubq_s8(vorrq_s8(q1_low, q1_high), vdupq_n_s8(32));

                        let q2_low = vandq_s8(v_ql_l32, vdupq_n_s8(0x0F));
                        let q2_high = vshlq_n_s8(vandq_s8(v_qh_l, vdupq_n_s8(0x0C)), 2);
                        let q2 = vsubq_s8(vorrq_s8(q2_low, q2_high), vdupq_n_s8(32));

                        let u_ql_l = vreinterpretq_u8_s8(v_ql_l);
                        let q3_low = vreinterpretq_s8_u8(vshrq_n_u8(u_ql_l, 4));
                        let q3_high = vandq_s8(v_qh_l, vdupq_n_s8(0x30));
                        let q3 = vsubq_s8(vorrq_s8(q3_low, q3_high), vdupq_n_s8(32));

                        let u_ql_l32 = vreinterpretq_u8_s8(v_ql_l32);
                        let q4_low = vreinterpretq_s8_u8(vshrq_n_u8(u_ql_l32, 4));
                        let u_qh_l = vreinterpretq_u8_s8(v_qh_l);
                        let q4_high =
                            vreinterpretq_s8_u8(vshrq_n_u8(vandq_u8(u_qh_l, vdupq_n_u8(0xC0)), 2));
                        let q4 = vsubq_s8(vorrq_s8(q4_low, q4_high), vdupq_n_s8(32));

                        let (q1_0, q1_1, q1_2, q1_3) = int8x16_to_f32x4_4(q1);
                        let (q2_0, q2_1, q2_2, q2_3) = int8x16_to_f32x4_4(q2);
                        let (q3_0, q3_1, q3_2, q3_3) = int8x16_to_f32x4_4(q3);
                        let (q4_0, q4_1, q4_2, q4_3) = int8x16_to_f32x4_4(q4);

                        let hbase = elem_base + half * 128 + l;
                        let ap_stripe1 = a_row.as_ptr().add(hbase);
                        let ap_stripe2 = a_row.as_ptr().add(hbase + 32);
                        let ap_stripe3 = a_row.as_ptr().add(hbase + 64);
                        let ap_stripe4 = a_row.as_ptr().add(hbase + 96);

                        acc_v = vmlaq_f32(acc_v, vmulq_f32(q1_0, ds1), vld1q_f32(ap_stripe1));
                        acc_v =
                            vmlaq_f32(acc_v, vmulq_f32(q1_1, ds1), vld1q_f32(ap_stripe1.add(4)));
                        acc_v =
                            vmlaq_f32(acc_v, vmulq_f32(q1_2, ds1), vld1q_f32(ap_stripe1.add(8)));
                        acc_v =
                            vmlaq_f32(acc_v, vmulq_f32(q1_3, ds1), vld1q_f32(ap_stripe1.add(12)));

                        acc_v = vmlaq_f32(acc_v, vmulq_f32(q2_0, ds2), vld1q_f32(ap_stripe2));
                        acc_v =
                            vmlaq_f32(acc_v, vmulq_f32(q2_1, ds2), vld1q_f32(ap_stripe2.add(4)));
                        acc_v =
                            vmlaq_f32(acc_v, vmulq_f32(q2_2, ds2), vld1q_f32(ap_stripe2.add(8)));
                        acc_v =
                            vmlaq_f32(acc_v, vmulq_f32(q2_3, ds2), vld1q_f32(ap_stripe2.add(12)));

                        acc_v = vmlaq_f32(acc_v, vmulq_f32(q3_0, ds3), vld1q_f32(ap_stripe3));
                        acc_v =
                            vmlaq_f32(acc_v, vmulq_f32(q3_1, ds3), vld1q_f32(ap_stripe3.add(4)));
                        acc_v =
                            vmlaq_f32(acc_v, vmulq_f32(q3_2, ds3), vld1q_f32(ap_stripe3.add(8)));
                        acc_v =
                            vmlaq_f32(acc_v, vmulq_f32(q3_3, ds3), vld1q_f32(ap_stripe3.add(12)));

                        acc_v = vmlaq_f32(acc_v, vmulq_f32(q4_0, ds4), vld1q_f32(ap_stripe4));
                        acc_v =
                            vmlaq_f32(acc_v, vmulq_f32(q4_1, ds4), vld1q_f32(ap_stripe4.add(4)));
                        acc_v =
                            vmlaq_f32(acc_v, vmulq_f32(q4_2, ds4), vld1q_f32(ap_stripe4.add(8)));
                        acc_v =
                            vmlaq_f32(acc_v, vmulq_f32(q4_3, ds4), vld1q_f32(ap_stripe4.add(12)));
                    }
                }
                acc += vaddvq_f32(acc_v);
            }
        }
        *out = acc;
    });
}

pub fn matmul_q6_k_batched(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
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

                // SAFETY: Same Q6_K super-block slice guarantees as `matmul_q6_k`
                // for `ql`, `qh`, `sc`. `act = &a[row_a * k..]` has at least `k`
                // elements remaining. The pointer arithmetic with `ap_stripe1..4`
                // uses `hbase` offsets within `Q6K_BLOCK_SIZE`, which is known
                // to be `<= k` because the caller checks `k % Q6K_BLOCK_SIZE == 0`.
                unsafe {
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

                            let ds1 = vdupq_n_f32(d * s1_val);
                            let ds2 = vdupq_n_f32(d * s2_val);
                            let ds3 = vdupq_n_f32(d * s3_val);
                            let ds4 = vdupq_n_f32(d * s4_val);

                            let v_ql_l = vld1q_s8(ql.as_ptr().add(ql_off + l) as *const i8);
                            let v_ql_l32 = vld1q_s8(ql.as_ptr().add(ql_off + l + 32) as *const i8);
                            let v_qh_l = vld1q_s8(qh.as_ptr().add(qh_off + l) as *const i8);

                            let q1_low = vandq_s8(v_ql_l, vdupq_n_s8(0x0F));
                            let q1_high = vshlq_n_s8(vandq_s8(v_qh_l, vdupq_n_s8(0x03)), 4);
                            let q1 = vsubq_s8(vorrq_s8(q1_low, q1_high), vdupq_n_s8(32));

                            let q2_low = vandq_s8(v_ql_l32, vdupq_n_s8(0x0F));
                            let q2_high = vshlq_n_s8(vandq_s8(v_qh_l, vdupq_n_s8(0x0C)), 2);
                            let q2 = vsubq_s8(vorrq_s8(q2_low, q2_high), vdupq_n_s8(32));

                            let u_ql_l = vreinterpretq_u8_s8(v_ql_l);
                            let q3_low = vreinterpretq_s8_u8(vshrq_n_u8(u_ql_l, 4));
                            let q3_high = vandq_s8(v_qh_l, vdupq_n_s8(0x30));
                            let q3 = vsubq_s8(vorrq_s8(q3_low, q3_high), vdupq_n_s8(32));

                            let u_ql_l32 = vreinterpretq_u8_s8(v_ql_l32);
                            let q4_low = vreinterpretq_s8_u8(vshrq_n_u8(u_ql_l32, 4));
                            let u_qh_l = vreinterpretq_u8_s8(v_qh_l);
                            let q4_high = vreinterpretq_s8_u8(vshrq_n_u8(
                                vandq_u8(u_qh_l, vdupq_n_u8(0xC0)),
                                2,
                            ));
                            let q4 = vsubq_s8(vorrq_s8(q4_low, q4_high), vdupq_n_s8(32));

                            let (q1_0, q1_1, q1_2, q1_3) = int8x16_to_f32x4_4(q1);
                            let (q2_0, q2_1, q2_2, q2_3) = int8x16_to_f32x4_4(q2);
                            let (q3_0, q3_1, q3_2, q3_3) = int8x16_to_f32x4_4(q3);
                            let (q4_0, q4_1, q4_2, q4_3) = int8x16_to_f32x4_4(q4);

                            let w1_0 = vmulq_f32(q1_0, ds1);
                            let w1_1 = vmulq_f32(q1_1, ds1);
                            let w1_2 = vmulq_f32(q1_2, ds1);
                            let w1_3 = vmulq_f32(q1_3, ds1);

                            let w2_0 = vmulq_f32(q2_0, ds2);
                            let w2_1 = vmulq_f32(q2_1, ds2);
                            let w2_2 = vmulq_f32(q2_2, ds2);
                            let w2_3 = vmulq_f32(q2_3, ds2);

                            let w3_0 = vmulq_f32(q3_0, ds3);
                            let w3_1 = vmulq_f32(q3_1, ds3);
                            let w3_2 = vmulq_f32(q3_2, ds3);
                            let w3_3 = vmulq_f32(q3_3, ds3);

                            let w4_0 = vmulq_f32(q4_0, ds4);
                            let w4_1 = vmulq_f32(q4_1, ds4);
                            let w4_2 = vmulq_f32(q4_2, ds4);
                            let w4_3 = vmulq_f32(q4_3, ds4);

                            let hbase = elem_base + half * 128 + l;

                            for row_a in 0..m {
                                let act = &a[row_a * k..];
                                let ap_stripe1 = act.as_ptr().add(hbase);
                                let ap_stripe2 = act.as_ptr().add(hbase + 32);
                                let ap_stripe3 = act.as_ptr().add(hbase + 64);
                                let ap_stripe4 = act.as_ptr().add(hbase + 96);

                                let mut acc_v = vdupq_n_f32(0.0);

                                acc_v = vmlaq_f32(acc_v, w1_0, vld1q_f32(ap_stripe1));
                                acc_v = vmlaq_f32(acc_v, w1_1, vld1q_f32(ap_stripe1.add(4)));
                                acc_v = vmlaq_f32(acc_v, w1_2, vld1q_f32(ap_stripe1.add(8)));
                                acc_v = vmlaq_f32(acc_v, w1_3, vld1q_f32(ap_stripe1.add(12)));

                                acc_v = vmlaq_f32(acc_v, w2_0, vld1q_f32(ap_stripe2));
                                acc_v = vmlaq_f32(acc_v, w2_1, vld1q_f32(ap_stripe2.add(4)));
                                acc_v = vmlaq_f32(acc_v, w2_2, vld1q_f32(ap_stripe2.add(8)));
                                acc_v = vmlaq_f32(acc_v, w2_3, vld1q_f32(ap_stripe2.add(12)));

                                acc_v = vmlaq_f32(acc_v, w3_0, vld1q_f32(ap_stripe3));
                                acc_v = vmlaq_f32(acc_v, w3_1, vld1q_f32(ap_stripe3.add(4)));
                                acc_v = vmlaq_f32(acc_v, w3_2, vld1q_f32(ap_stripe3.add(8)));
                                acc_v = vmlaq_f32(acc_v, w3_3, vld1q_f32(ap_stripe3.add(12)));

                                acc_v = vmlaq_f32(acc_v, w4_0, vld1q_f32(ap_stripe4));
                                acc_v = vmlaq_f32(acc_v, w4_1, vld1q_f32(ap_stripe4.add(4)));
                                acc_v = vmlaq_f32(acc_v, w4_2, vld1q_f32(ap_stripe4.add(8)));
                                acc_v = vmlaq_f32(acc_v, w4_3, vld1q_f32(ap_stripe4.add(12)));

                                results[row_a] += hsum_f32x4(acc_v);
                            }
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

// ── Q6_K dotprod kernels using vdotq_s32 ─────────────────────────

pub fn matmul_q6_k_dotprod(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if m > 1 {
        return matmul_q6_k_batched_dotprod(a, b_quant, m, n, k, c);
    }
    let blocks_per_row = k / Q6K_BLOCK_SIZE;
    let a_quant = quantize_activations_q8_0(a, k);

    c.par_iter_mut().enumerate().for_each(|(idx, out)| {
        let col_b = idx % n;
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

            // SAFETY: Same Q6_K block slice guarantees for `ql`, `qh`, `sc`.
            // `a_quant` was allocated by `quantize_activations_q8_0` with
            // `blocks * 34` bytes where `blocks = k.div_ceil(32)`. The
            // offsets `ao1..ao4` are `b_base * Q8_BLOCK_BYTES` where
            // `b_base = block_idx * 8 + half * 4 < blocks_per_row * 8`,
            // so `ao4 + 2 + l + 16 < blocks * 34`.
            unsafe {
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

                        let v_ql_l = vld1q_s8(ql.as_ptr().add(ql_off + l) as *const i8);
                        let v_ql_l32 = vld1q_s8(ql.as_ptr().add(ql_off + l + 32) as *const i8);
                        let v_qh_l = vld1q_s8(qh.as_ptr().add(qh_off + l) as *const i8);

                        let q1_low = vandq_s8(v_ql_l, vdupq_n_s8(0x0F));
                        let q1_high = vshlq_n_s8(vandq_s8(v_qh_l, vdupq_n_s8(0x03)), 4);
                        let q1 = vsubq_s8(vorrq_s8(q1_low, q1_high), vdupq_n_s8(32));

                        let q2_low = vandq_s8(v_ql_l32, vdupq_n_s8(0x0F));
                        let q2_high = vshlq_n_s8(vandq_s8(v_qh_l, vdupq_n_s8(0x0C)), 2);
                        let q2 = vsubq_s8(vorrq_s8(q2_low, q2_high), vdupq_n_s8(32));

                        let u_ql_l = vreinterpretq_u8_s8(v_ql_l);
                        let q3_low = vreinterpretq_s8_u8(vshrq_n_u8(u_ql_l, 4));
                        let q3_high = vandq_s8(v_qh_l, vdupq_n_s8(0x30));
                        let q3 = vsubq_s8(vorrq_s8(q3_low, q3_high), vdupq_n_s8(32));

                        let u_ql_l32 = vreinterpretq_u8_s8(v_ql_l32);
                        let q4_low = vreinterpretq_s8_u8(vshrq_n_u8(u_ql_l32, 4));
                        let u_qh_l = vreinterpretq_u8_s8(v_qh_l);
                        let q4_high =
                            vreinterpretq_s8_u8(vshrq_n_u8(vandq_u8(u_qh_l, vdupq_n_u8(0xC0)), 2));
                        let q4 = vsubq_s8(vorrq_s8(q4_low, q4_high), vdupq_n_s8(32));

                        let b_base = block_idx * 8 + half * 4;

                        let ao1 = b_base * Q8_BLOCK_BYTES;
                        let ao2 = (b_base + 1) * Q8_BLOCK_BYTES;
                        let ao3 = (b_base + 2) * Q8_BLOCK_BYTES;
                        let ao4 = (b_base + 3) * Q8_BLOCK_BYTES;

                        let d_a1 =
                            half::f16::from_le_bytes([a_quant[ao1], a_quant[ao1 + 1]]).to_f32();
                        let d_a2 =
                            half::f16::from_le_bytes([a_quant[ao2], a_quant[ao2 + 1]]).to_f32();
                        let d_a3 =
                            half::f16::from_le_bytes([a_quant[ao3], a_quant[ao3 + 1]]).to_f32();
                        let d_a4 =
                            half::f16::from_le_bytes([a_quant[ao4], a_quant[ao4 + 1]]).to_f32();

                        let act1 = vld1q_s8(a_quant.as_ptr().add(ao1 + 2 + l) as *const i8);
                        let act2 = vld1q_s8(a_quant.as_ptr().add(ao2 + 2 + l) as *const i8);
                        let act3 = vld1q_s8(a_quant.as_ptr().add(ao3 + 2 + l) as *const i8);
                        let act4 = vld1q_s8(a_quant.as_ptr().add(ao4 + 2 + l) as *const i8);

                        let mut dot1 = vdupq_n_s32(0);
                        dot1 = my_vdotq_s32(dot1, act1, q1);
                        let mut dot2 = vdupq_n_s32(0);
                        dot2 = my_vdotq_s32(dot2, act2, q2);
                        let mut dot3 = vdupq_n_s32(0);
                        dot3 = my_vdotq_s32(dot3, act3, q3);
                        let mut dot4 = vdupq_n_s32(0);
                        dot4 = my_vdotq_s32(dot4, act4, q4);

                        let factor1 = d * s1_val * d_a1;
                        let factor2 = d * s2_val * d_a2;
                        let factor3 = d * s3_val * d_a3;
                        let factor4 = d * s4_val * d_a4;

                        acc += factor1 * (vaddvq_s32(dot1) as f32);
                        acc += factor2 * (vaddvq_s32(dot2) as f32);
                        acc += factor3 * (vaddvq_s32(dot3) as f32);
                        acc += factor4 * (vaddvq_s32(dot4) as f32);
                    }
                }
            }
        }
        *out = acc;
    });
}

pub fn matmul_q6_k_batched_dotprod(
    a: &[f32],
    b_quant: &[u8],
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
) {
    let blocks_per_row = k / Q6K_BLOCK_SIZE;
    let a_quant = quantize_activations_q8_0_batched(a, m, k);
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

                // SAFETY: Same Q6_K block and `a_quant` slice guarantees as
                // `matmul_q6_k_dotprod`. `a_quant` was allocated by
                // `quantize_activations_q8_0_batched` with
                // `m * blocks_per_row * 8 * Q8_BLOCK_BYTES` bytes. For each
                // `row_a`, `ao1..ao4` are computed from
                // `(row_a * blocks_per_row * 8 + b_base + k) * Q8_BLOCK_BYTES`,
                // which stays within that allocation because
                // `b_base + 3 < blocks_per_row * 8`.
                unsafe {
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

                            let v_ql_l = vld1q_s8(ql.as_ptr().add(ql_off + l) as *const i8);
                            let v_ql_l32 = vld1q_s8(ql.as_ptr().add(ql_off + l + 32) as *const i8);
                            let v_qh_l = vld1q_s8(qh.as_ptr().add(qh_off + l) as *const i8);

                            let q1_low = vandq_s8(v_ql_l, vdupq_n_s8(0x0F));
                            let q1_high = vshlq_n_s8(vandq_s8(v_qh_l, vdupq_n_s8(0x03)), 4);
                            let q1 = vsubq_s8(vorrq_s8(q1_low, q1_high), vdupq_n_s8(32));

                            let q2_low = vandq_s8(v_ql_l32, vdupq_n_s8(0x0F));
                            let q2_high = vshlq_n_s8(vandq_s8(v_qh_l, vdupq_n_s8(0x0C)), 2);
                            let q2 = vsubq_s8(vorrq_s8(q2_low, q2_high), vdupq_n_s8(32));

                            let u_ql_l = vreinterpretq_u8_s8(v_ql_l);
                            let q3_low = vreinterpretq_s8_u8(vshrq_n_u8(u_ql_l, 4));
                            let q3_high = vandq_s8(v_qh_l, vdupq_n_s8(0x30));
                            let q3 = vsubq_s8(vorrq_s8(q3_low, q3_high), vdupq_n_s8(32));

                            let u_ql_l32 = vreinterpretq_u8_s8(v_ql_l32);
                            let q4_low = vreinterpretq_s8_u8(vshrq_n_u8(u_ql_l32, 4));
                            let u_qh_l = vreinterpretq_u8_s8(v_qh_l);
                            let q4_high = vreinterpretq_s8_u8(vshrq_n_u8(
                                vandq_u8(u_qh_l, vdupq_n_u8(0xC0)),
                                2,
                            ));
                            let q4 = vsubq_s8(vorrq_s8(q4_low, q4_high), vdupq_n_s8(32));

                            let b_base = block_idx * 8 + half * 4;

                            for row_a in 0..m {
                                let ao1 = (row_a * blocks_per_row * 8 + b_base) * Q8_BLOCK_BYTES;
                                let ao2 =
                                    (row_a * blocks_per_row * 8 + b_base + 1) * Q8_BLOCK_BYTES;
                                let ao3 =
                                    (row_a * blocks_per_row * 8 + b_base + 2) * Q8_BLOCK_BYTES;
                                let ao4 =
                                    (row_a * blocks_per_row * 8 + b_base + 3) * Q8_BLOCK_BYTES;

                                let d_a1 =
                                    half::f16::from_le_bytes([a_quant[ao1], a_quant[ao1 + 1]])
                                        .to_f32();
                                let d_a2 =
                                    half::f16::from_le_bytes([a_quant[ao2], a_quant[ao2 + 1]])
                                        .to_f32();
                                let d_a3 =
                                    half::f16::from_le_bytes([a_quant[ao3], a_quant[ao3 + 1]])
                                        .to_f32();
                                let d_a4 =
                                    half::f16::from_le_bytes([a_quant[ao4], a_quant[ao4 + 1]])
                                        .to_f32();

                                let act1 = vld1q_s8(a_quant.as_ptr().add(ao1 + 2 + l) as *const i8);
                                let act2 = vld1q_s8(a_quant.as_ptr().add(ao2 + 2 + l) as *const i8);
                                let act3 = vld1q_s8(a_quant.as_ptr().add(ao3 + 2 + l) as *const i8);
                                let act4 = vld1q_s8(a_quant.as_ptr().add(ao4 + 2 + l) as *const i8);

                                let mut dot1 = vdupq_n_s32(0);
                                dot1 = my_vdotq_s32(dot1, act1, q1);
                                let mut dot2 = vdupq_n_s32(0);
                                dot2 = my_vdotq_s32(dot2, act2, q2);
                                let mut dot3 = vdupq_n_s32(0);
                                dot3 = my_vdotq_s32(dot3, act3, q3);
                                let mut dot4 = vdupq_n_s32(0);
                                dot4 = my_vdotq_s32(dot4, act4, q4);

                                let factor1 = d * s1_val * d_a1;
                                let factor2 = d * s2_val * d_a2;
                                let factor3 = d * s3_val * d_a3;
                                let factor4 = d * s4_val * d_a4;

                                results[row_a] += factor1 * (vaddvq_s32(dot1) as f32);
                                results[row_a] += factor2 * (vaddvq_s32(dot2) as f32);
                                results[row_a] += factor3 * (vaddvq_s32(dot3) as f32);
                                results[row_a] += factor4 * (vaddvq_s32(dot4) as f32);
                            }
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

// ── Q5_K single-row (M=1) f32 NEON ────────────────────────────

pub fn matmul_q5_k(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if m > 1 {
        return matmul_q5_k_batched(a, b_quant, m, n, k, c);
    }
    let blocks_per_row = k / Q5K_BLOCK_SIZE;

    c.par_iter_mut().enumerate().for_each(|(idx, out)| {
        let col_b = idx % n;
        let a_row = &a[..k];
        let mut acc = 0.0f32;
        let b_row_start = col_b * blocks_per_row * Q5K_BLOCK_BYTES;
        let bit_pos_arr: [i8; 8] = [0, -1, -2, -3, -4, -5, -6, -7];

        for block_idx in 0..blocks_per_row {
            let bo = b_row_start + block_idx * Q5K_BLOCK_BYTES;
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

                let base_lo = elem_base + is * 32;
                let base_hi = elem_base + (is + 1) * 32;

                // SAFETY: `qs` is a 128-byte slice (`Q5K_BLOCK_BYTES - 48`),
                // so `qs.as_ptr().add(qs_ptr + batch)` is valid for all
                // batches (max 8 bytes read). `a_row` has length `k` and
                // `base_lo + batch + 4 < k` because the indices are bounded
                // by `Q5K_BLOCK_SIZE`. `qh` is a 32-byte slice and
                // `qh[is * 4 + batch / 8]` is within bounds since
                // `is <= 7` and `batch / 8 < 4`.
                unsafe {
                    let mut acc_v = vdupq_n_f32(0.0);
                    let d1_v = vdupq_n_f32(d1);
                    let m1_v = vdupq_n_f32(m1);
                    let d2_v = vdupq_n_f32(d2);
                    let m2_v = vdupq_n_f32(m2);
                    let neg_m1_v = vnegq_f32(m1_v);
                    let neg_m2_v = vnegq_f32(m2_v);

                    for batch in (0..32).step_by(8) {
                        let bp = qs.as_ptr().add(qs_ptr + batch);
                        let ap_lo = a_row.as_ptr().add(base_lo + batch);
                        let ap_hi = a_row.as_ptr().add(base_hi + batch);

                        let bytes = vld1_s8(bp as *const i8);
                        let ubytes = vreinterpret_u8_s8(bytes);
                        let lo = vand_u8(ubytes, vdup_n_u8(0x0F));
                        let hi = vshr_n_u8(ubytes, 4);

                        let qh_byte_lo = qh[is * 4 + batch / 8];
                        let qh_byte_hi = qh[(is + 1) * 4 + batch / 8];
                        let qh_v_lo = vdup_n_u8(qh_byte_lo);
                        let qh_v_hi = vdup_n_u8(qh_byte_hi);
                        let bit_pos = vld1_s8(bit_pos_arr.as_ptr());
                        let qh_shr_lo = vshl_u8(qh_v_lo, bit_pos);
                        let qh_shr_hi = vshl_u8(qh_v_hi, bit_pos);
                        let qh_bits_lo = vand_u8(qh_shr_lo, vdup_n_u8(1));
                        let qh_bits_hi = vand_u8(qh_shr_hi, vdup_n_u8(1));

                        let lo_q5 = vand_u8(vorr_u8(lo, vshl_n_u8(qh_bits_lo, 4)), vdup_n_u8(0x1F));
                        let hi_q5 = vand_u8(vorr_u8(hi, vshl_n_u8(qh_bits_hi, 4)), vdup_n_u8(0x1F));

                        let lo_cent = vreinterpret_s8_u8(lo_q5);
                        let hi_cent = vreinterpret_s8_u8(hi_q5);

                        let lo_s16 = vmovl_s8(lo_cent);
                        let lo_0_f = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_s16)));
                        let lo_1_f = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_s16)));

                        let hi_s16 = vmovl_s8(hi_cent);
                        let hi_0_f = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_s16)));
                        let hi_1_f = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_s16)));

                        // scaled = d1 * q_centered - m1
                        let lo_0_s = vmlaq_f32(neg_m1_v, d1_v, lo_0_f);
                        let lo_1_s = vmlaq_f32(neg_m1_v, d1_v, lo_1_f);
                        let hi_0_s = vmlaq_f32(neg_m2_v, d2_v, hi_0_f);
                        let hi_1_s = vmlaq_f32(neg_m2_v, d2_v, hi_1_f);

                        let a_lo_0 = vld1q_f32(ap_lo);
                        let a_lo_1 = vld1q_f32(ap_lo.add(4));
                        let a_hi_0 = vld1q_f32(ap_hi);
                        let a_hi_1 = vld1q_f32(ap_hi.add(4));

                        acc_v = vmlaq_f32(acc_v, lo_0_s, a_lo_0);
                        acc_v = vmlaq_f32(acc_v, lo_1_s, a_lo_1);
                        acc_v = vmlaq_f32(acc_v, hi_0_s, a_hi_0);
                        acc_v = vmlaq_f32(acc_v, hi_1_s, a_hi_1);
                    }

                    qs_ptr += 32;
                    is += 2;

                    acc += vaddvq_f32(acc_v);
                }
            }
        }
        *out = acc;
    });
}

// ── Q5_K batched (M > 1) f32 NEON ─────────────────────────────

fn matmul_q5_k_batched(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    let blocks_per_row = k / Q5K_BLOCK_SIZE;
    let mut flat_results = vec![0.0f32; m * n];
    let bit_pos_arr: [i8; 8] = [0, -1, -2, -3, -4, -5, -6, -7];

    let tasks: Vec<&mut [f32]> = flat_results.chunks_mut(m).collect();
    tasks
        .into_par_iter()
        .enumerate()
        .for_each(|(col_b, results)| {
            let b_row_start = col_b * blocks_per_row * Q5K_BLOCK_BYTES;

            for block_idx in 0..blocks_per_row {
                let bo = b_row_start + block_idx * Q5K_BLOCK_BYTES;
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
                    let m1_val = dmin * mm0 as f32;
                    let (sc1, mm1) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * sc1 as f32;
                    let m2_val = dmin * mm1 as f32;

                    let _base_lo = elem_base + is * 32;
                    let _base_hi = elem_base + (is + 1) * 32;

                    let mut lo_cent_f32 = [0.0f32; 32];
                    let mut hi_cent_f32 = [0.0f32; 32];

                    // SAFETY: `qs` and `qh` are slices bounded by Q5_K block size.
                    unsafe {
                        for batch in (0..32).step_by(8) {
                            let bp = qs.as_ptr().add(qs_ptr + batch);
                            let bytes = vld1_s8(bp as *const i8);
                            let ubytes = vreinterpret_u8_s8(bytes);
                            let lo = vand_u8(ubytes, vdup_n_u8(0x0F));
                            let hi = vshr_n_u8(ubytes, 4);

                            let qh_byte_lo = qh[is * 4 + batch / 8];
                            let qh_byte_hi = qh[(is + 1) * 4 + batch / 8];
                            let qh_v_lo = vdup_n_u8(qh_byte_lo);
                            let qh_v_hi = vdup_n_u8(qh_byte_hi);
                            let bit_pos = vld1_s8(bit_pos_arr.as_ptr());
                            let qh_shr_lo = vshl_u8(qh_v_lo, bit_pos);
                            let qh_shr_hi = vshl_u8(qh_v_hi, bit_pos);
                            let qh_bits_lo = vand_u8(qh_shr_lo, vdup_n_u8(1));
                            let qh_bits_hi = vand_u8(qh_shr_hi, vdup_n_u8(1));

                            let lo_q5 =
                                vand_u8(vorr_u8(lo, vshl_n_u8(qh_bits_lo, 4)), vdup_n_u8(0x1F));
                            let hi_q5 =
                                vand_u8(vorr_u8(hi, vshl_n_u8(qh_bits_hi, 4)), vdup_n_u8(0x1F));

                            let lo_cent = vreinterpret_s8_u8(lo_q5);
                            let hi_cent = vreinterpret_s8_u8(hi_q5);

                            let lo_s16 = vmovl_s8(lo_cent);
                            let lo_0_f = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_s16)));
                            let lo_1_f = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_s16)));

                            let hi_s16 = vmovl_s8(hi_cent);
                            let hi_0_f = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_s16)));
                            let hi_1_f = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_s16)));

                            vst1q_f32(lo_cent_f32.as_mut_ptr().add(batch), lo_0_f);
                            vst1q_f32(lo_cent_f32.as_mut_ptr().add(batch + 4), lo_1_f);
                            vst1q_f32(hi_cent_f32.as_mut_ptr().add(batch), hi_0_f);
                            vst1q_f32(hi_cent_f32.as_mut_ptr().add(batch + 4), hi_1_f);
                        }

                        for row_a in 0..m {
                            let act = &a[row_a * k..];
                            let mut acc_v = vdupq_n_f32(0.0);
                            let d1_v = vdupq_n_f32(d1);
                            let m1_v = vnegq_f32(vdupq_n_f32(m1_val));
                            let d2_v = vdupq_n_f32(d2);
                            let m2_v = vnegq_f32(vdupq_n_f32(m2_val));
                            let base_lo = elem_base + is * 32;
                            let base_hi = elem_base + (is + 1) * 32;

                            for batch in (0..32).step_by(8) {
                                let ap_lo = act.as_ptr().add(base_lo + batch);
                                let ap_hi = act.as_ptr().add(base_hi + batch);

                                let ql_0 = vld1q_f32(lo_cent_f32.as_ptr().add(batch));
                                let ql_1 = vld1q_f32(lo_cent_f32.as_ptr().add(batch + 4));
                                let qh_0 = vld1q_f32(hi_cent_f32.as_ptr().add(batch));
                                let qh_1 = vld1q_f32(hi_cent_f32.as_ptr().add(batch + 4));

                                let lo_0_s = vmlaq_f32(m1_v, d1_v, ql_0);
                                let lo_1_s = vmlaq_f32(m1_v, d1_v, ql_1);
                                let hi_0_s = vmlaq_f32(m2_v, d2_v, qh_0);
                                let hi_1_s = vmlaq_f32(m2_v, d2_v, qh_1);

                                let a_lo_0 = vld1q_f32(ap_lo);
                                let a_lo_1 = vld1q_f32(ap_lo.add(4));
                                let a_hi_0 = vld1q_f32(ap_hi);
                                let a_hi_1 = vld1q_f32(ap_hi.add(4));

                                acc_v = vmlaq_f32(acc_v, lo_0_s, a_lo_0);
                                acc_v = vmlaq_f32(acc_v, lo_1_s, a_lo_1);
                                acc_v = vmlaq_f32(acc_v, hi_0_s, a_hi_0);
                                acc_v = vmlaq_f32(acc_v, hi_1_s, a_hi_1);
                            }

                            results[row_a] += vaddvq_f32(acc_v);
                        }
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

// ── Q5_K dotprod (batched, Q8_0 activations) ────────────────────

pub fn matmul_q5_k_dotprod(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    let blocks_per_row = k / Q5K_BLOCK_SIZE;
    let a_quant = quantize_activations_q8_0_batched(a, m, k);
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

                unsafe {
                    for is in (0..8).step_by(2) {
                        let (sc0, mm0) = get_scale_min_k4(is, scales);
                        let d1 = d * sc0 as f32;
                        let m1 = dmin * mm0 as f32;
                        let (sc1, mm1) = get_scale_min_k4(is + 1, scales);
                        let d2 = d * sc1 as f32;
                        let m2 = dmin * mm1 as f32;

                        for l in (0..32).step_by(16) {
                            let v_qs_lo = vld1q_s8(qs.as_ptr().add(is / 2 * 32 + l) as *const i8);
                            let qh_lo = *qh.as_ptr().add(is * 4 + l / 8);
                            let qh_hi = *qh.as_ptr().add(is * 4 + l / 8 + 1);
                            let qh_lo_odd = *qh.as_ptr().add((is + 1) * 4 + l / 8);
                            let qh_hi_odd = *qh.as_ptr().add((is + 1) * 4 + l / 8 + 1);

                            let ql_even = vandq_s8(v_qs_lo, vdupq_n_s8(0x0F));
                            let ql_odd =
                                vreinterpretq_s8_u8(vshrq_n_u8(vreinterpretq_u8_s8(v_qs_lo), 4));

                            let qh_bits_lo = vdupq_n_s8(qh_lo as i8);
                            let bit_mask = vdupq_n_u8(1);
                            let bit_pos_arr: [u8; 16] =
                                [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7];
                            let bit_pos = vld1q_u8(bit_pos_arr.as_ptr());
                            let shifted = vshlq_u8(
                                vreinterpretq_u8_s8(qh_bits_lo),
                                vnegq_s8(vreinterpretq_s8_u8(bit_pos)),
                            );
                            let qh_lo_bits = vandq_u8(shifted, bit_mask);
                            let qh_bits_hi = vdupq_n_s8(qh_hi as i8);
                            let shifted_hi = vshlq_u8(
                                vreinterpretq_u8_s8(qh_bits_hi),
                                vnegq_s8(vreinterpretq_s8_u8(bit_pos)),
                            );
                            let qh_hi_bits = vandq_u8(shifted_hi, bit_mask);

                            let qh_comb =
                                vcombine_u8(vget_low_u8(qh_lo_bits), vget_low_u8(qh_hi_bits));

                            let qh_bits_lo_odd = vdupq_n_s8(qh_lo_odd as i8);
                            let shifted_lo_odd = vshlq_u8(
                                vreinterpretq_u8_s8(qh_bits_lo_odd),
                                vnegq_s8(vreinterpretq_s8_u8(bit_pos)),
                            );
                            let qh_lo_bits_odd = vandq_u8(shifted_lo_odd, bit_mask);
                            let qh_bits_hi_odd = vdupq_n_s8(qh_hi_odd as i8);
                            let shifted_hi_odd = vshlq_u8(
                                vreinterpretq_u8_s8(qh_bits_hi_odd),
                                vnegq_s8(vreinterpretq_s8_u8(bit_pos)),
                            );
                            let qh_hi_bits_odd = vandq_u8(shifted_hi_odd, bit_mask);

                            let qh_comb_odd = vcombine_u8(
                                vget_low_u8(qh_lo_bits_odd),
                                vget_low_u8(qh_hi_bits_odd),
                            );

                            let qh_even = vandq_u8(
                                vorrq_u8(vreinterpretq_u8_s8(ql_even), vshlq_n_u8(qh_comb, 4)),
                                vdupq_n_u8(0x1F),
                            );
                            let qh_odd = vandq_u8(
                                vorrq_u8(vreinterpretq_u8_s8(ql_odd), vshlq_n_u8(qh_comb_odd, 4)),
                                vdupq_n_u8(0x1F),
                            );

                            let q_even_raw = vreinterpretq_s8_u8(qh_even);
                            let q_odd_raw = vreinterpretq_s8_u8(qh_odd);

                            for row_a in 0..m {
                                // Q8_0 activation blocks for sub-blocks is and is+1
                                let block_lo = is;
                                let block_hi = is + 1;
                                let base_lo =
                                    (row_a * blocks_per_row * 8 + block_lo) * Q8_BLOCK_BYTES;
                                let base_hi =
                                    (row_a * blocks_per_row * 8 + block_hi) * Q8_BLOCK_BYTES;

                                let act_lo =
                                    vld1q_s8(a_quant.as_ptr().add(base_lo + 2 + l) as *const i8);
                                let act_hi =
                                    vld1q_s8(a_quant.as_ptr().add(base_hi + 2 + l) as *const i8);

                                let d_a_lo = half::f16::from_le_bytes([
                                    a_quant[base_lo],
                                    a_quant[base_lo + 1],
                                ])
                                .to_f32();
                                let d_a_hi = half::f16::from_le_bytes([
                                    a_quant[base_hi],
                                    a_quant[base_hi + 1],
                                ])
                                .to_f32();

                                let mut dot_lo = vdupq_n_s32(0);
                                dot_lo = my_vdotq_s32(dot_lo, act_lo, q_even_raw);

                                let mut dot_hi = vdupq_n_s32(0);
                                dot_hi = my_vdotq_s32(dot_hi, act_hi, q_odd_raw);

                                let sum_act_lo = sum_i8x16(act_lo) as f32;
                                let sum_act_hi = sum_i8x16(act_hi) as f32;

                                results[row_a] +=
                                    d_a_lo * (d1 * (vaddvq_s32(dot_lo) as f32) - m1 * sum_act_lo);
                                results[row_a] +=
                                    d_a_hi * (d2 * (vaddvq_s32(dot_hi) as f32) - m2 * sum_act_hi);
                            }
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

// ── Q8_0 single-row (M=1) ─────────────────────────────────────────

pub fn matmul_q8_0(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if m > 1 {
        return matmul_q8_0_batched(a, b_quant, m, n, k, c);
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

            // SAFETY: `quant` is a slice of at most 32 bytes
            // (`elem_count <= Q8_BLOCK_SIZE`). The `while l + 8 <= elem_count`
            // loop guarantees `vld1_s8` reads 8 bytes within `quant`. `a_row`
            // has length `k` and `elem_base + l + 4 < k` because
            // `elem_base < k` and `l < Q8_BLOCK_SIZE`.
            unsafe {
                let mut acc_v = vdupq_n_f32(0.0);
                let mut l = 0usize;
                while l + 8 <= elem_count {
                    let q_bytes = vld1_s8(quant.as_ptr().add(l) as *const i8);
                    let q_16 = vmovl_s8(q_bytes);
                    let q_32_0 = vmovl_s16(vget_low_s16(q_16));
                    let q_32_1 = vmovl_s16(vget_high_s16(q_16));
                    let q_f32_0 = vcvtq_f32_s32(q_32_0);
                    let q_f32_1 = vcvtq_f32_s32(q_32_1);
                    let a_0 = vld1q_f32(a_row.as_ptr().add(elem_base + l));
                    let a_1 = vld1q_f32(a_row.as_ptr().add(elem_base + l + 4));
                    acc_v = vmlaq_f32(acc_v, q_f32_0, a_0);
                    acc_v = vmlaq_f32(acc_v, q_f32_1, a_1);
                    l += 8;
                }
                let mut sum = hsum_f32x4(acc_v);
                for i in l..elem_count {
                    sum += a_row[elem_base + i] * quant[i] as i8 as f32;
                }
                acc += d * sum;
            }
        }
        *out = acc;
    });
}

fn matmul_q8_0_batched(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
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
                    // SAFETY: Same `quant` and element bounds as `matmul_q8_0`.
                    // `act = &a[row_a * k..]` has at least `k` elements, and
                    // `elem_base + l + 4 < k` by the same block construction.
                    unsafe {
                        let mut acc_v = vdupq_n_f32(0.0);
                        let mut l = 0usize;
                        while l + 8 <= elem_count {
                            let q_bytes = vld1_s8(quant.as_ptr().add(l) as *const i8);
                            let q_16 = vmovl_s8(q_bytes);
                            let q_32_0 = vmovl_s16(vget_low_s16(q_16));
                            let q_32_1 = vmovl_s16(vget_high_s16(q_16));
                            let q_f32_0 = vcvtq_f32_s32(q_32_0);
                            let q_f32_1 = vcvtq_f32_s32(q_32_1);
                            let a_0 = vld1q_f32(act.as_ptr().add(elem_base + l));
                            let a_1 = vld1q_f32(act.as_ptr().add(elem_base + l + 4));
                            acc_v = vmlaq_f32(acc_v, q_f32_0, a_0);
                            acc_v = vmlaq_f32(acc_v, q_f32_1, a_1);
                            l += 8;
                        }
                        let mut sum = hsum_f32x4(acc_v);
                        for i in l..elem_count {
                            sum += act[elem_base + i] * quant[i] as i8 as f32;
                        }
                        results[row_a] += d * sum;
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

// ── Q8_0 dotprod kernels using vdotq_s32 ─────────────────────────

pub fn matmul_q8_0_dotprod(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if m > 1 {
        return matmul_q8_0_batched_dotprod(a, b_quant, m, n, k, c);
    }
    let blocks_per_row = k.div_ceil(Q8_BLOCK_SIZE);
    let a_quant = quantize_activations_q8_0(a, k);

    c.par_iter_mut().enumerate().for_each(|(idx, out)| {
        let col_b = idx % n;
        let mut acc = 0.0f32;
        let b_row_start = col_b * blocks_per_row * Q8_BLOCK_BYTES;

        for block_idx in 0..blocks_per_row {
            let bo = b_row_start + block_idx * Q8_BLOCK_BYTES;
            if bo + Q8_BLOCK_BYTES > b_quant.len() {
                break;
            }
            let d_b = half::f16::from_le_bytes([b_quant[bo], b_quant[bo + 1]]).to_f32();
            let ao = block_idx * Q8_BLOCK_BYTES;
            let d_a = half::f16::from_le_bytes([a_quant[ao], a_quant[ao + 1]]).to_f32();
            let d_ab = d_a * d_b;

            // SAFETY: `b_quant` has been bounds-checked so `bo + Q8_BLOCK_BYTES`
            // does not exceed its length; `bo + 2 + 16` reads 16 bytes from
            // the 32-byte quant within each Q8_0 block. `a_quant` was allocated
            // by `quantize_activations_q8_0` with `blocks * 34` bytes, and
            // `ao` is `block_idx * Q8_BLOCK_BYTES` which stays within bounds.
            // All `vld1q_s8` calls require 16-byte aligned pointers, which
            // Rust slices guarantee.
            unsafe {
                let w0 = vld1q_s8(b_quant.as_ptr().add(bo + 2) as *const i8);
                let w1 = vld1q_s8(b_quant.as_ptr().add(bo + 2 + 16) as *const i8);
                let act0 = vld1q_s8(a_quant.as_ptr().add(ao + 2) as *const i8);
                let act1 = vld1q_s8(a_quant.as_ptr().add(ao + 2 + 16) as *const i8);

                let mut acc_i32 = vdupq_n_s32(0);
                acc_i32 = my_vdotq_s32(acc_i32, w0, act0);
                acc_i32 = my_vdotq_s32(acc_i32, w1, act1);

                let block_sum = vaddvq_s32(acc_i32);
                acc += (block_sum as f32) * d_ab;
            }
        }
        *out = acc;
    });
}

pub fn matmul_q8_0_batched_dotprod(
    a: &[f32],
    b_quant: &[u8],
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
) {
    let blocks_per_row = k.div_ceil(Q8_BLOCK_SIZE);
    let a_quant = quantize_activations_q8_0_batched(a, m, k);
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
                let d_b = half::f16::from_le_bytes([b_quant[bo], b_quant[bo + 1]]).to_f32();
                // SAFETY: Weight pointer invariants same as `matmul_q8_0_dotprod`.
                // `a_quant` was allocated by `quantize_activations_q8_0_batched`
                // with `m * blocks_per_row * Q8_BLOCK_BYTES` bytes. `ao` is
                // `(row_a * blocks_per_row + block_idx) * Q8_BLOCK_BYTES`,
                // which never exceeds `m * blocks_per_row * Q8_BLOCK_BYTES`.
                unsafe {
                    let w0 = vld1q_s8(b_quant.as_ptr().add(bo + 2) as *const i8);
                    let w1 = vld1q_s8(b_quant.as_ptr().add(bo + 2 + 16) as *const i8);

                    for row_a in 0..m {
                        let ao = (row_a * blocks_per_row + block_idx) * Q8_BLOCK_BYTES;
                        let d_a = half::f16::from_le_bytes([a_quant[ao], a_quant[ao + 1]]).to_f32();
                        let act0 = vld1q_s8(a_quant.as_ptr().add(ao + 2) as *const i8);
                        let act1 = vld1q_s8(a_quant.as_ptr().add(ao + 2 + 16) as *const i8);

                        let mut acc_i32 = vdupq_n_s32(0);
                        acc_i32 = my_vdotq_s32(acc_i32, w0, act0);
                        acc_i32 = my_vdotq_s32(acc_i32, w1, act1);

                        let block_sum = vaddvq_s32(acc_i32);
                        results[row_a] += (block_sum as f32) * (d_a * d_b);
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

// ── F16 NEON kernel ──────────────────────────────────────────────────
//
// Since stdarch_neon_f16 (vld1q_f16) is unstable on stable Rust, we
// load raw u16 bytes, convert via half::f16, then use NEON f32 ops.
// This still avoids materializing the full f32 weight matrix.

pub fn matmul_f16_neon(a: &[f32], b_bytes: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if m > 1 {
        return matmul_f16_neon_batched(a, b_bytes, m, n, k, c);
    }
    let a_row = &a[..k];
    c.par_iter_mut().enumerate().for_each(|(idx, out)| {
        let col_b = idx % n;
        let b_row_start = col_b * k * 2;
        let acc;
        // SAFETY: `b_bytes` has length `k * 2 * n`, and `b_row_start` is
        // `col_b * k * 2` where `col_b < n`, so `bo + 15 <= k * 2 * n - 1`.
        // Stack arrays `w01`, `w23`, `a01`, `a23` of 4 f32s are 16-byte
        // aligned on aarch64, required by `vld1q_f32`. `a_row` has length `k`
        // and `i + 7 < k` in the main loop, with the remainder handling the
        // trailing elements.
        unsafe {
            let mut acc_v = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 8 <= k {
                let bo = b_row_start + i * 2;
                let b0 = half::f16::from_bits(u16::from_le_bytes([b_bytes[bo], b_bytes[bo + 1]]))
                    .to_f32();
                let b1 =
                    half::f16::from_bits(u16::from_le_bytes([b_bytes[bo + 2], b_bytes[bo + 3]]))
                        .to_f32();
                let b2 =
                    half::f16::from_bits(u16::from_le_bytes([b_bytes[bo + 4], b_bytes[bo + 5]]))
                        .to_f32();
                let b3 =
                    half::f16::from_bits(u16::from_le_bytes([b_bytes[bo + 6], b_bytes[bo + 7]]))
                        .to_f32();
                let b4 =
                    half::f16::from_bits(u16::from_le_bytes([b_bytes[bo + 8], b_bytes[bo + 9]]))
                        .to_f32();
                let b5 =
                    half::f16::from_bits(u16::from_le_bytes([b_bytes[bo + 10], b_bytes[bo + 11]]))
                        .to_f32();
                let b6 =
                    half::f16::from_bits(u16::from_le_bytes([b_bytes[bo + 12], b_bytes[bo + 13]]))
                        .to_f32();
                let b7 =
                    half::f16::from_bits(u16::from_le_bytes([b_bytes[bo + 14], b_bytes[bo + 15]]))
                        .to_f32();
                let w01 = [b0, b1, b2, b3];
                let w23 = [b4, b5, b6, b7];
                let v_w01 = vld1q_f32(w01.as_ptr());
                let v_w23 = vld1q_f32(w23.as_ptr());
                let a01 = [a_row[i], a_row[i + 1], a_row[i + 2], a_row[i + 3]];
                let a23 = [a_row[i + 4], a_row[i + 5], a_row[i + 6], a_row[i + 7]];
                let v_a01 = vld1q_f32(a01.as_ptr());
                let v_a23 = vld1q_f32(a23.as_ptr());
                acc_v = vfmaq_f32(acc_v, v_a01, v_w01);
                acc_v = vfmaq_f32(acc_v, v_a23, v_w23);
                i += 8;
            }
            for j in i..k {
                let bo = b_row_start + j * 2;
                let w = half::f16::from_bits(u16::from_le_bytes([b_bytes[bo], b_bytes[bo + 1]]))
                    .to_f32();
                acc_v = vfmaq_n_f32(acc_v, vdupq_n_f32(a_row[j]), w);
            }
            acc = vaddvq_f32(acc_v);
        }
        *out = acc;
    });
}

fn matmul_f16_neon_batched(a: &[f32], b_bytes: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    let mut flat_results = vec![0.0f32; m * n];
    let tasks: Vec<&mut [f32]> = flat_results.chunks_mut(m).collect();
    tasks
        .into_par_iter()
        .enumerate()
        .for_each(|(col_b, results)| {
            let b_row_start = col_b * k * 2;
            let mut i = 0;
            while i + 8 <= k {
                let bo = b_row_start + i * 2;
                let b0 = half::f16::from_bits(u16::from_le_bytes([b_bytes[bo], b_bytes[bo + 1]]))
                    .to_f32();
                let b1 =
                    half::f16::from_bits(u16::from_le_bytes([b_bytes[bo + 2], b_bytes[bo + 3]]))
                        .to_f32();
                let b2 =
                    half::f16::from_bits(u16::from_le_bytes([b_bytes[bo + 4], b_bytes[bo + 5]]))
                        .to_f32();
                let b3 =
                    half::f16::from_bits(u16::from_le_bytes([b_bytes[bo + 6], b_bytes[bo + 7]]))
                        .to_f32();
                let b4 =
                    half::f16::from_bits(u16::from_le_bytes([b_bytes[bo + 8], b_bytes[bo + 9]]))
                        .to_f32();
                let b5 =
                    half::f16::from_bits(u16::from_le_bytes([b_bytes[bo + 10], b_bytes[bo + 11]]))
                        .to_f32();
                let b6 =
                    half::f16::from_bits(u16::from_le_bytes([b_bytes[bo + 12], b_bytes[bo + 13]]))
                        .to_f32();
                let b7 =
                    half::f16::from_bits(u16::from_le_bytes([b_bytes[bo + 14], b_bytes[bo + 15]]))
                        .to_f32();
                let w01 = [b0, b1, b2, b3];
                let w23 = [b4, b5, b6, b7];
                // SAFETY: Stack arrays `w01`, `w23` (4 f32s each) are 16-byte
                // aligned for `vld1q_f32`. `a` has length `m * k`, so
                // `a_base + 7 = row_a * k + i + 7 < m * k` for all iterations
                // in the `while i + 8 <= k` loop. Stack arrays `a01`, `a23`
                // are also 4-element f32 buffers with proper alignment.
                unsafe {
                    let v_w01 = vld1q_f32(w01.as_ptr());
                    let v_w23 = vld1q_f32(w23.as_ptr());
                    for row_a in 0..m {
                        let a_base = row_a * k + i;
                        let a01 = [a[a_base], a[a_base + 1], a[a_base + 2], a[a_base + 3]];
                        let a23 = [a[a_base + 4], a[a_base + 5], a[a_base + 6], a[a_base + 7]];
                        let v_a01 = vld1q_f32(a01.as_ptr());
                        let v_a23 = vld1q_f32(a23.as_ptr());
                        results[row_a] += vaddvq_f32(vmulq_f32(v_a01, v_w01))
                            + vaddvq_f32(vmulq_f32(v_a23, v_w23));
                    }
                }
                i += 8;
            }
            for j in i..k {
                let bo = b_row_start + j * 2;
                let w = half::f16::from_bits(u16::from_le_bytes([b_bytes[bo], b_bytes[bo + 1]]))
                    .to_f32();
                for row_a in 0..m {
                    results[row_a] += a[row_a * k + j] * w;
                }
            }
        });
    for col_b in 0..n {
        for row_a in 0..m {
            c[row_a * n + col_b] = flat_results[col_b * m + row_a];
        }
    }
}
