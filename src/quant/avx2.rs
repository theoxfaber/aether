use super::dequant::{dequant_q4_k_block, dequant_q5_k_block, dequant_q6_k_block};

/// Horizontal sum of 8 f32 values in an AVX2 register
#[inline(always)]
unsafe fn hsum_f32x8(v: std::arch::x86_64::__m256) -> f32 {
    let hi = std::arch::x86_64::_mm256_extractf128_ps(v, 1);
    let lo = std::arch::x86_64::_mm256_castps256_ps128(v);
    let sum = std::arch::x86_64::_mm_add_ps(lo, hi);
    let sum = std::arch::x86_64::_mm_hadd_ps(sum, sum);
    let sum = std::arch::x86_64::_mm_hadd_ps(sum, sum);
    std::arch::x86_64::_mm_cvtss_f32(sum)
}

/// Q4_K matmul using AVX2 (f32 pipeline, no dotprod)
pub unsafe fn matmul_q4_k(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    let block_size = super::Q4K_BLOCK_SIZE; // 256
    let block_bytes = super::Q4K_BLOCK_BYTES; // 144
    let num_blocks = k / block_size;

    for row in 0..m {
        for col in 0..n {
            let a_base = row * k;
            let mut sum = 0.0f32;

            for blk in 0..num_blocks {
                let b_off = (col * num_blocks + blk) * block_bytes;
                let b_data = &b_quant[b_off..b_off + block_bytes];
                let a_off = a_base + blk * block_size;
                let a_data = &a[a_off..a_off + block_size];

                let deq = dequant_q4_k_block(b_data);
                let mut block_sum = 0.0f32;
                let mut i = 0;
                while i + 8 <= block_size {
                    let avec = std::arch::x86_64::_mm256_loadu_ps(&a_data[i] as *const f32);
                    let dvec = std::arch::x86_64::_mm256_loadu_ps(&deq[i] as *const f32);
                    let prod = std::arch::x86_64::_mm256_mul_ps(avec, dvec);
                    block_sum += hsum_f32x8(prod);
                    i += 8;
                }
                for j in i..block_size {
                    block_sum += a_data[j] * deq[j];
                }
                sum += block_sum;
            }
            c[row * n + col] = sum;
        }
    }
}

/// Q6_K matmul using AVX2 (f32 pipeline)
pub unsafe fn matmul_q6_k(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    let block_size = super::Q6K_BLOCK_SIZE; // 256
    let block_bytes = super::Q6K_BLOCK_BYTES; // 210
    let num_blocks = k / block_size;

    for row in 0..m {
        for col in 0..n {
            let a_base = row * k;
            let mut sum = 0.0f32;

            for blk in 0..num_blocks {
                let b_off = (col * num_blocks + blk) * block_bytes;
                let b_data = &b_quant[b_off..b_off + block_bytes];
                let a_off = a_base + blk * block_size;
                let a_data = &a[a_off..a_off + block_size];

                let deq = dequant_q6_k_block(b_data);
                let mut block_sum = 0.0f32;
                let mut i = 0;
                while i + 8 <= block_size {
                    let avec = std::arch::x86_64::_mm256_loadu_ps(&a_data[i] as *const f32);
                    let dvec = std::arch::x86_64::_mm256_loadu_ps(&deq[i] as *const f32);
                    let prod = std::arch::x86_64::_mm256_mul_ps(avec, dvec);
                    block_sum += hsum_f32x8(prod);
                    i += 8;
                }
                for j in i..block_size {
                    block_sum += a_data[j] * deq[j];
                }
                sum += block_sum;
            }
            c[row * n + col] = sum;
        }
    }
}

/// Q5_K matmul using AVX2 (f32 pipeline)
pub unsafe fn matmul_q5_k(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    let block_size = super::Q5K_BLOCK_SIZE; // 256
    let block_bytes = super::Q5K_BLOCK_BYTES; // 176
    let num_blocks = k / block_size;

    for row in 0..m {
        for col in 0..n {
            let a_base = row * k;
            let mut sum = 0.0f32;

            for blk in 0..num_blocks {
                let b_off = (col * num_blocks + blk) * block_bytes;
                let b_data = &b_quant[b_off..b_off + block_bytes];
                let a_off = a_base + blk * block_size;
                let a_data = &a[a_off..a_off + block_size];

                let deq = dequant_q5_k_block(b_data);
                let mut block_sum = 0.0f32;
                let mut i = 0;
                while i + 8 <= block_size {
                    let avec = std::arch::x86_64::_mm256_loadu_ps(&a_data[i] as *const f32);
                    let dvec = std::arch::x86_64::_mm256_loadu_ps(&deq[i] as *const f32);
                    let prod = std::arch::x86_64::_mm256_mul_ps(avec, dvec);
                    block_sum += hsum_f32x8(prod);
                    i += 8;
                }
                for j in i..block_size {
                    block_sum += a_data[j] * deq[j];
                }
                sum += block_sum;
            }
            c[row * n + col] = sum;
        }
    }
}

/// Q8_0 matmul using AVX2 (f32 pipeline)
pub unsafe fn matmul_q8_0(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let num_blocks = k / BLOCK_SIZE;

    for row in 0..m {
        for col in 0..n {
            let a_base = row * k;
            let mut sum = 0.0f32;

            for blk in 0..num_blocks {
                let b_off = (col * num_blocks + blk) * BLOCK_BYTES;
                let d_raw = &b_quant[b_off..b_off + 2];
                let d = half::f16::from_bits(u16::from_le_bytes([d_raw[0], d_raw[1]])).to_f32();

                let q_offset = b_off + 2;
                let mut block_sum = 0.0f32;
                let mut i = 0;

                while i + 16 <= BLOCK_SIZE {
                    let avec = std::arch::x86_64::_mm256_loadu_ps(
                        &a[a_base + blk * BLOCK_SIZE + i] as *const f32,
                    );
                    let ptr =
                        &b_quant[q_offset + i] as *const u8 as *const std::arch::x86_64::__m128i;
                    let qi8 = std::arch::x86_64::_mm_loadl_epi64(ptr);
                    let qi16 = std::arch::x86_64::_mm_cvtepi8_epi16(qi8);
                    let qi16_hi = std::arch::x86_64::_mm_cvtepi8_epi16(
                        std::arch::x86_64::_mm_loadl_epi64(ptr.add(1)),
                    );
                    let qi32_0 = std::arch::x86_64::_mm_cvtepi16_epi32(qi16);
                    let qi32_1 = std::arch::x86_64::_mm_cvtepi16_epi32(
                        std::arch::x86_64::_mm_shuffle_epi32(qi16, 0xEE),
                    );
                    let qi32_2 = std::arch::x86_64::_mm_cvtepi16_epi32(qi16_hi);
                    let qi32_3 = std::arch::x86_64::_mm_cvtepi16_epi32(
                        std::arch::x86_64::_mm_shuffle_epi32(qi16_hi, 0xEE),
                    );

                    let qf = std::arch::x86_64::_mm256_set_m128(
                        std::arch::x86_64::_mm_cvtepi32_ps(qi32_1),
                        std::arch::x86_64::_mm_cvtepi32_ps(qi32_0),
                    );
                    let qf2 = std::arch::x86_64::_mm256_set_m128(
                        std::arch::x86_64::_mm_cvtepi32_ps(qi32_3),
                        std::arch::x86_64::_mm_cvtepi32_ps(qi32_2),
                    );

                    let prod = std::arch::x86_64::_mm256_fmadd_ps(
                        qf,
                        avec,
                        std::arch::x86_64::_mm256_setzero_ps(),
                    );
                    let prod2 = std::arch::x86_64::_mm256_fmadd_ps(
                        qf2,
                        std::arch::x86_64::_mm256_loadu_ps(
                            &a[a_base + blk * BLOCK_SIZE + i + 8] as *const f32,
                        ),
                        std::arch::x86_64::_mm256_setzero_ps(),
                    );
                    block_sum += hsum_f32x8(prod) + hsum_f32x8(prod2);
                    i += 16;
                }
                for j in i..BLOCK_SIZE {
                    let q_val = b_quant[q_offset + j] as i8 as f32;
                    block_sum += a[a_base + blk * BLOCK_SIZE + j] * q_val;
                }
                sum += block_sum * d;
            }
            c[row * n + col] = sum;
        }
    }
}

/// F16 matmul using AVX2 with F16C conversion
pub unsafe fn matmul_f16(a: &[f32], b_bytes: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    for row in 0..m {
        for col in 0..n {
            let a_base = row * k;
            let b_base = col * k * 2; // f16 = 2 bytes per element
            let mut sum = 0.0f32;
            let mut i = 0;

            while i + 8 <= k {
                let avec = std::arch::x86_64::_mm256_loadu_ps(&a[a_base + i] as *const f32);
                let f16_vals = std::arch::x86_64::_mm_loadu_si128(
                    &b_bytes[b_base + i * 2] as *const u8 as *const std::arch::x86_64::__m128i,
                );
                let bvec = std::arch::x86_64::_mm256_cvtph_ps(f16_vals);
                let prod = std::arch::x86_64::_mm256_mul_ps(avec, bvec);
                sum += hsum_f32x8(prod);
                i += 8;
            }
            for j in i..k {
                let f16_val = half::f16::from_bits(u16::from_le_bytes([
                    b_bytes[b_base + j * 2],
                    b_bytes[b_base + j * 2 + 1],
                ]));
                sum += a[a_base + j] * f16_val.to_f32();
            }
            c[row * n + col] = sum;
        }
    }
}

/// Q2_K matmul using AVX2 (f32 pipeline)
pub unsafe fn matmul_q2_k(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    const BLOCK_BYTES: usize = 84;
    const BLOCK_SIZE: usize = 256;
    let num_blocks = k / BLOCK_SIZE;
    for row in 0..m {
        for col in 0..n {
            let a_base = row * k;
            let mut sum = 0.0f32;
            for blk in 0..num_blocks {
                let bo = (col * num_blocks + blk) * BLOCK_BYTES;
                if bo + BLOCK_BYTES > b_quant.len() {
                    break;
                }
                let d = half::f16::from_bits(u16::from_le_bytes([b_quant[bo], b_quant[bo + 1]]))
                    .to_f32();
                let dmin =
                    half::f16::from_bits(u16::from_le_bytes([b_quant[bo + 2], b_quant[bo + 3]]))
                        .to_f32();
                let scales = &b_quant[bo + 4..bo + 20];
                let qs = &b_quant[bo + 20..bo + 84];
                let a_off = a_base + blk * BLOCK_SIZE;
                for s in 0..16usize {
                    let half = s / 8;
                    let q_off = half * 32;
                    let shift = (s as i32 / 2 % 4) * 2;
                    let sc_val = scales[s];
                    let dl = d * (sc_val & 0xF) as f32;
                    let ml = dmin * (sc_val >> 4) as f32;
                    let byte_base = (s % 2) * 16;
                    let mut block_sum = 0.0f32;
                    let mut i = 0;
                    while i + 8 <= 16 {
                        let a_idx = a_off + s * 16 + i;
                        let avec = std::arch::x86_64::_mm256_loadu_ps(&a[a_idx] as *const f32);
                        let dvec = std::arch::x86_64::_mm256_set1_ps(dl);
                        let mvec = std::arch::x86_64::_mm256_set1_ps(ml);
                        let mut qval = [0i32; 8];
                        for j in 0..8 {
                            qval[j] = ((qs[q_off + byte_base + i + j] >> shift) & 3) as i32;
                        }
                        let qf = std::arch::x86_64::_mm256_cvtepi32_ps(
                            std::arch::x86_64::_mm256_loadu_si256(
                                qval.as_ptr() as *const std::arch::x86_64::__m256i
                            ),
                        );
                        let scaled = std::arch::x86_64::_mm256_fmadd_ps(
                            dvec,
                            qf,
                            std::arch::x86_64::_mm256_sub_ps(
                                std::arch::x86_64::_mm256_setzero_ps(),
                                mvec,
                            ),
                        );
                        let prod = std::arch::x86_64::_mm256_mul_ps(scaled, avec);
                        block_sum += hsum_f32x8(prod);
                        i += 8;
                    }
                    for j in i..16 {
                        let q = ((qs[q_off + byte_base + j] >> shift) & 3) as i8;
                        let a_val = a[a_off + s * 16 + j];
                        block_sum += a_val * (dl * q as f32 - ml);
                    }
                    sum += block_sum;
                }
            }
            c[row * n + col] = sum;
        }
    }
}

/// Q3_K matmul using AVX2 (f32 pipeline)
pub unsafe fn matmul_q3_k(a: &[f32], b_quant: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    const BLOCK_BYTES: usize = 110;
    const BLOCK_SIZE: usize = 256;
    let num_blocks = k / BLOCK_SIZE;
    for row in 0..m {
        for col in 0..n {
            let a_base = row * k;
            let mut sum = 0.0f32;
            for blk in 0..num_blocks {
                let bo = (col * num_blocks + blk) * BLOCK_BYTES;
                if bo + BLOCK_BYTES > b_quant.len() {
                    break;
                }
                let hmask = &b_quant[bo..bo + 32];
                let qs = &b_quant[bo + 32..bo + 96];
                let sr = &b_quant[bo + 96..bo + 108];
                let d = half::f16::from_bits(u16::from_le_bytes([
                    b_quant[bo + 108],
                    b_quant[bo + 109],
                ]))
                .to_f32();
                let mut sc = [0i16; 16];
                for i in 0..8 {
                    sc[i] = (sr[i] & 0x0F) as i16;
                    sc[i + 8] = (sr[i] >> 4) as i16;
                }
                for i in 0..4 {
                    let b = sr[i + 8];
                    sc[i] |= ((b >> 0) as i16 & 0x03) << 4;
                    sc[i + 4] |= ((b >> 2) as i16 & 0x03) << 4;
                    sc[i + 8] |= ((b >> 4) as i16 & 0x03) << 4;
                    sc[i + 12] |= ((b >> 6) as i16 & 0x03) << 4;
                }
                for s in &mut sc {
                    *s -= 32;
                }
                let a_off = a_base + blk * BLOCK_SIZE;
                for s in 0..16usize {
                    let half = s / 8;
                    let q_off = half * 32;
                    let shift = (s as i32 / 2 % 4) * 2;
                    let dl = d * sc[s] as f32;
                    let byte_base = (s % 2) * 16;
                    let hm_bit = s as u8 / 2;
                    let mut block_sum = 0.0f32;
                    let mut i = 0;
                    while i + 8 <= 16 {
                        let avec = std::arch::x86_64::_mm256_loadu_ps(
                            &a[a_off + s * 16 + i] as *const f32,
                        );
                        let dvec = std::arch::x86_64::_mm256_set1_ps(dl);
                        let mut qval = [0f32; 8];
                        for j in 0..8 {
                            let ql = (qs[q_off + byte_base + i + j] >> shift) & 3;
                            let qh = ((hmask[byte_base + i + j] >> hm_bit) & 1) ^ 1;
                            let q = (ql as i8) - ((qh as i8) << 2);
                            qval[j] = q as f32;
                        }
                        let qf = std::arch::x86_64::_mm256_loadu_ps(qval.as_ptr());
                        let prod = std::arch::x86_64::_mm256_mul_ps(
                            std::arch::x86_64::_mm256_mul_ps(dvec, qf),
                            avec,
                        );
                        block_sum += hsum_f32x8(prod);
                        i += 8;
                    }
                    for j in i..16 {
                        let ql = (qs[q_off + byte_base + j] >> shift) & 3;
                        let qh = ((hmask[byte_base + j] >> hm_bit) & 1) ^ 1;
                        let q = (ql as i8) - ((qh as i8) << 2);
                        block_sum += a[a_off + s * 16 + j] * dl * q as f32;
                    }
                    sum += block_sum;
                }
            }
            c[row * n + col] = sum;
        }
    }
}
