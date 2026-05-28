use super::*;

    fn generate_activations(m: usize, k: usize) -> Vec<f32> {
        let mut a = vec![0.0f32; m * k];
        for i in 0..a.len() {
            a[i] = ((i * 17) % 100) as f32 / 50.0 - 1.0;
        }
        a
    }

    fn generate_random_bytes(len: usize) -> Vec<u8> {
        let mut b = vec![0u8; len];
        for i in 0..len {
            b[i] = ((i * 31) % 256) as u8;
        }
        b
    }

    fn f32_to_f16_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for &v in values {
            let f = half::f16::from_f32(v);
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        bytes
    }

    fn matmul_q8_0_dotprod_reference(
        a_quant: &[u8],
        b_quant: &[u8],
        m: usize,
        n: usize,
        k: usize,
        c: &mut [f32],
    ) {
        let blocks_per_row = k.div_ceil(Q8_BLOCK_SIZE);
        for row_a in 0..m {
            for col_b in 0..n {
                let mut acc = 0.0f32;
                let b_row_start = col_b * blocks_per_row * Q8_BLOCK_BYTES;
                for block_idx in 0..blocks_per_row {
                    let bo = b_row_start + block_idx * Q8_BLOCK_BYTES;
                    let ao = (row_a * blocks_per_row + block_idx) * Q8_BLOCK_BYTES;

                    let d_b = half::f16::from_le_bytes([b_quant[bo], b_quant[bo + 1]]).to_f32();
                    let d_a = half::f16::from_le_bytes([a_quant[ao], a_quant[ao + 1]]).to_f32();
                    let d_ab = d_a * d_b;

                    let mut sum = 0i32;
                    for i in 0..32 {
                        let q_a = a_quant[ao + 2 + i] as i8 as i32;
                        let q_b = b_quant[bo + 2 + i] as i8 as i32;
                        sum += q_a * q_b;
                    }
                    acc += (sum as f32) * d_ab;
                }
                c[row_a * n + col_b] = acc;
            }
        }
    }

    fn matmul_q4_k_dotprod_reference(
        a_quant: &[u8],
        b_quant: &[u8],
        m: usize,
        n: usize,
        k: usize,
        c: &mut [f32],
    ) {
        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        for row_a in 0..m {
            for col_b in 0..n {
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

                        let start_lo = (row_a * blocks_per_row * 8 + b_lo) * Q8_BLOCK_BYTES;
                        let start_hi = (row_a * blocks_per_row * 8 + b_hi) * Q8_BLOCK_BYTES;

                        let d_a1 =
                            half::f16::from_le_bytes([a_quant[start_lo], a_quant[start_lo + 1]])
                                .to_f32();
                        let d_a2 =
                            half::f16::from_le_bytes([a_quant[start_hi], a_quant[start_hi + 1]])
                                .to_f32();

                        let mut dot_lo = 0i32;
                        let mut sum_lo = 0i32;
                        let mut dot_hi = 0i32;
                        let mut sum_hi = 0i32;

                        for l in 0..32 {
                            let byte = qs[qs_ptr + l];
                            let q_lo = (byte & 0x0F) as i8 as i32;
                            let q_hi = ((byte >> 4) & 0x0F) as i8 as i32;

                            let act_lo = a_quant[start_lo + 2 + l] as i8 as i32;
                            let act_hi = a_quant[start_hi + 2 + l] as i8 as i32;

                            dot_lo += q_lo * act_lo;
                            sum_lo += act_lo;

                            dot_hi += q_hi * act_hi;
                            sum_hi += act_hi;
                        }

                        acc += d_a1 * (d1 * dot_lo as f32 - m1 * sum_lo as f32);
                        acc += d_a2 * (d2 * dot_hi as f32 - m2 * sum_hi as f32);

                        qs_ptr += 32;
                        is += 2;
                    }
                }
                c[row_a * n + col_b] = acc;
            }
        }
    }

    fn matmul_q6_k_dotprod_reference(
        a_quant: &[u8],
        b_quant: &[u8],
        m: usize,
        n: usize,
        k: usize,
        c: &mut [f32],
    ) {
        let blocks_per_row = k / Q6K_BLOCK_SIZE;
        for row_a in 0..m {
            for col_b in 0..n {
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

                    for half_idx in 0..2 {
                        let ql_off = half_idx * 64;
                        let qh_off = half_idx * 32;
                        let sc_off = half_idx * 8;

                        for l in (0..32).step_by(16) {
                            let is = l / 16;
                            let s1_val = sc[sc_off + is + 0] as i8 as f32;
                            let s2_val = sc[sc_off + is + 2] as i8 as f32;
                            let s3_val = sc[sc_off + is + 4] as i8 as f32;
                            let s4_val = sc[sc_off + is + 6] as i8 as f32;

                            let mut q1 = [0i8; 16];
                            let mut q2 = [0i8; 16];
                            let mut q3 = [0i8; 16];
                            let mut q4 = [0i8; 16];

                            for idx in 0..16 {
                                let ql_val = ql[ql_off + l + idx];
                                let ql_val32 = ql[ql_off + l + 32 + idx];
                                let qh_val = qh[qh_off + l + idx];

                                let q1_low = ql_val & 0x0F;
                                let q1_high = (qh_val & 0x03) << 4;
                                q1[idx] = ((q1_low | q1_high) as i8).wrapping_sub(32);

                                let q2_low = ql_val32 & 0x0F;
                                let q2_high = (qh_val & 0x0C) << 2;
                                q2[idx] = ((q2_low | q2_high) as i8).wrapping_sub(32);

                                let q3_low = ql_val >> 4;
                                let q3_high = qh_val & 0x30;
                                q3[idx] = ((q3_low | q3_high) as i8).wrapping_sub(32);

                                let q4_low = ql_val32 >> 4;
                                let q4_high = (qh_val & 0xC0) >> 2;
                                q4[idx] = ((q4_low | q4_high) as i8).wrapping_sub(32);
                            }

                            let b_base = block_idx * 8 + half_idx * 4;

                            let start_ao1 = (row_a * blocks_per_row * 8 + b_base) * Q8_BLOCK_BYTES;
                            let start_ao2 =
                                (row_a * blocks_per_row * 8 + b_base + 1) * Q8_BLOCK_BYTES;
                            let start_ao3 =
                                (row_a * blocks_per_row * 8 + b_base + 2) * Q8_BLOCK_BYTES;
                            let start_ao4 =
                                (row_a * blocks_per_row * 8 + b_base + 3) * Q8_BLOCK_BYTES;

                            let d_a1 = half::f16::from_le_bytes([
                                a_quant[start_ao1],
                                a_quant[start_ao1 + 1],
                            ])
                            .to_f32();
                            let d_a2 = half::f16::from_le_bytes([
                                a_quant[start_ao2],
                                a_quant[start_ao2 + 1],
                            ])
                            .to_f32();
                            let d_a3 = half::f16::from_le_bytes([
                                a_quant[start_ao3],
                                a_quant[start_ao3 + 1],
                            ])
                            .to_f32();
                            let d_a4 = half::f16::from_le_bytes([
                                a_quant[start_ao4],
                                a_quant[start_ao4 + 1],
                            ])
                            .to_f32();

                            let mut dot1 = 0i32;
                            let mut dot2 = 0i32;
                            let mut dot3 = 0i32;
                            let mut dot4 = 0i32;

                            for idx in 0..16 {
                                let act1 = a_quant[start_ao1 + 2 + l + idx] as i8 as i32;
                                let act2 = a_quant[start_ao2 + 2 + l + idx] as i8 as i32;
                                let act3 = a_quant[start_ao3 + 2 + l + idx] as i8 as i32;
                                let act4 = a_quant[start_ao4 + 2 + l + idx] as i8 as i32;

                                dot1 += act1 * q1[idx] as i32;
                                dot2 += act2 * q2[idx] as i32;
                                dot3 += act3 * q3[idx] as i32;
                                dot4 += act4 * q4[idx] as i32;
                            }

                            let factor1 = d * s1_val * d_a1;
                            let factor2 = d * s2_val * d_a2;
                            let factor3 = d * s3_val * d_a3;
                            let factor4 = d * s4_val * d_a4;

                            acc += factor1 * dot1 as f32;
                            acc += factor2 * dot2 as f32;
                            acc += factor3 * dot3 as f32;
                            acc += factor4 * dot4 as f32;
                        }
                    }
                }
                c[row_a * n + col_b] = acc;
            }
        }
    }

    fn matmul_q5_k_dotprod_reference(
        a_quant: &[u8],
        b_quant: &[u8],
        m: usize,
        n: usize,
        k: usize,
        c: &mut [f32],
    ) {
        let blocks_per_row = k / Q5K_BLOCK_SIZE;
        for row_a in 0..m {
            for col_b in 0..n {
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

                    let mut qs_ptr = 0usize;
                    let mut is = 0usize;

                    for _j in (0..Q5K_BLOCK_SIZE).step_by(64) {
                        let (sc0, mm0) = get_scale_min_k4(is, scales);
                        let d1 = d * sc0 as f32;
                        let m1 = dmin * mm0 as f32;
                        let (sc1, mm1) = get_scale_min_k4(is + 1, scales);
                        let d2 = d * sc1 as f32;
                        let m2_val = dmin * mm1 as f32;

                        let mut dot_lo = 0i32;
                        let mut sum_lo = 0i32;
                        let mut dot_hi = 0i32;
                        let mut sum_hi = 0i32;

                        for l in 0..32 {
                            let byte_l = qs[qs_ptr + l];
                            let ql_lo = (byte_l & 0x0F) as i32;
                            let ql_hi = ((byte_l >> 4) & 0x0F) as i32;
                            let qh_byte_lo = qh[(is * 32 + l) / 8];
                            let qh_bit_lo = ((qh_byte_lo >> ((is * 32 + l) % 8)) & 1) as i32;
                            let qh_byte_hi = qh[((is + 1) * 32 + l) / 8];
                            let qh_bit_hi = ((qh_byte_hi >> (((is + 1) * 32 + l) % 8)) & 1) as i32;

                            let q_lo = ql_lo + qh_bit_lo * 16;
                            let q_hi = ql_hi + qh_bit_hi * 16;

                            let start_lo = (row_a * blocks_per_row * 8 + is) * Q8_BLOCK_BYTES;
                            let start_hi = (row_a * blocks_per_row * 8 + is + 1) * Q8_BLOCK_BYTES;
                            let act_lo = a_quant[start_lo + 2 + l] as i8 as i32;
                            let act_hi = a_quant[start_hi + 2 + l] as i8 as i32;

                            dot_lo += q_lo * act_lo;
                            sum_lo += act_lo;
                            dot_hi += q_hi * act_hi;
                            sum_hi += act_hi;
                        }

                        let d_a_lo = half::f16::from_le_bytes([
                            a_quant[(row_a * blocks_per_row * 8 + is) * Q8_BLOCK_BYTES],
                            a_quant[(row_a * blocks_per_row * 8 + is) * Q8_BLOCK_BYTES + 1],
                        ])
                        .to_f32();
                        let d_a_hi = half::f16::from_le_bytes([
                            a_quant[(row_a * blocks_per_row * 8 + is + 1) * Q8_BLOCK_BYTES],
                            a_quant[(row_a * blocks_per_row * 8 + is + 1) * Q8_BLOCK_BYTES + 1],
                        ])
                        .to_f32();

                        acc += d_a_lo * (d1 * dot_lo as f32 - m1 * sum_lo as f32);
                        acc += d_a_hi * (d2 * dot_hi as f32 - m2_val * sum_hi as f32);

                        qs_ptr += 32;
                        is += 2;
                    }
                }
                c[row_a * n + col_b] = acc;
            }
        }
    }

    #[test]
    fn test_q8_0_correctness() {
        let m_cases = vec![1, 3];
        let k = 256;
        let n = 8;
        let b_bytes_len = n * (k / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES;
        let b_quant = generate_random_bytes(b_bytes_len);

        for m in m_cases {
            let a = generate_activations(m, k);
            let mut c_dotprod = vec![0.0f32; m * n];
            let mut c_ref = vec![0.0f32; m * n];

            let a_quant = if m > 1 {
                neon::quantize_activations_q8_0_batched(&a, m, k)
            } else {
                neon::quantize_activations_q8_0(&a, k)
            };

            matmul_q8_0_dotprod_reference(&a_quant, &b_quant, m, n, k, &mut c_ref);

            #[cfg(target_arch = "aarch64")]
            {
                neon::matmul_q8_0_dotprod(&a, &b_quant, m, n, k, &mut c_dotprod);

                for i in 0..c_ref.len() {
                    let diff = (c_ref[i] - c_dotprod[i]).abs();
                    assert!(
                        diff < 1e-3 || (c_ref[i].is_nan() && c_dotprod[i].is_nan()),
                        "Q8_0 dotprod mismatch at index {} for m={}: ref={}, dotprod={}, diff={}",
                        i,
                        m,
                        c_ref[i],
                        c_dotprod[i],
                        diff
                    );
                }
            }
        }
    }

    #[test]
    fn test_q4_k_correctness() {
        let m_cases = vec![1, 3];
        let k = 256;
        let n = 8;
        let b_bytes_len = n * (k / Q4K_BLOCK_SIZE) * Q4K_BLOCK_BYTES;
        let b_quant = generate_random_bytes(b_bytes_len);

        for m in m_cases {
            let a = generate_activations(m, k);
            let mut c_scalar = vec![0.0f32; m * n];
            let mut c_simd = vec![0.0f32; m * n];

            matmul_q4_k_scalar(&a, &b_quant, m, n, k, &mut c_scalar);

            #[cfg(target_arch = "aarch64")]
            {
                neon::matmul_q4_k(&a, &b_quant, m, n, k, &mut c_simd);
                for i in 0..c_scalar.len() {
                    let diff = (c_scalar[i] - c_simd[i]).abs();
                    let relative_diff = diff / c_scalar[i].abs().max(1.0);
                    assert!(diff < 1e-2 || relative_diff < 1e-4, "Q4_K NEON standard mismatch at index {} for m={}: scalar={}, simd={}, diff={}, rel={}", i, m, c_scalar[i], c_simd[i], diff, relative_diff);
                }

                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    let mut c_dotprod = vec![0.0f32; m * n];
                    let mut c_ref = vec![0.0f32; m * n];
                    let a_quant = if m > 1 {
                        neon::quantize_activations_q8_0_batched(&a, m, k)
                    } else {
                        neon::quantize_activations_q8_0(&a, k)
                    };
                    matmul_q4_k_dotprod_reference(&a_quant, &b_quant, m, n, k, &mut c_ref);
                    neon::matmul_q4_k_dotprod(&a, &b_quant, m, n, k, &mut c_dotprod);
                    for i in 0..c_ref.len() {
                        let diff = (c_ref[i] - c_dotprod[i]).abs();
                        assert!(diff < 1e-3, "Q4_K NEON dotprod mismatch at index {} for m={}: ref={}, dotprod={}, diff={}", i, m, c_ref[i], c_dotprod[i], diff);
                    }
                }
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                matmul_q4_k(&a, &b_quant, m, n, k, &mut c_simd);
                for i in 0..c_scalar.len() {
                    let diff = (c_scalar[i] - c_simd[i]).abs();
                    let relative_diff = diff / c_scalar[i].abs().max(1.0);
                    assert!(
                        diff < 1e-2 || relative_diff < 1e-4,
                        "Q4_K mismatch at index {} for m={}: scalar={}, simd={}, diff={}, rel={}",
                        i,
                        m,
                        c_scalar[i],
                        c_simd[i],
                        diff,
                        relative_diff
                    );
                }
            }
        }
    }

    #[test]
    fn test_q6_k_correctness() {
        let m_cases = vec![1, 3];
        let k = 256;
        let n = 8;
        let b_bytes_len = n * (k / Q6K_BLOCK_SIZE) * Q6K_BLOCK_BYTES;
        let b_quant = generate_random_bytes(b_bytes_len);

        for m in m_cases {
            let a = generate_activations(m, k);
            let mut c_scalar = vec![0.0f32; m * n];
            let mut c_simd = vec![0.0f32; m * n];

            matmul_q6_k_scalar(&a, &b_quant, m, n, k, &mut c_scalar);

            #[cfg(target_arch = "aarch64")]
            {
                neon::matmul_q6_k(&a, &b_quant, m, n, k, &mut c_simd);
                for i in 0..c_scalar.len() {
                    let diff = (c_scalar[i] - c_simd[i]).abs();
                    let relative_diff = diff / c_scalar[i].abs().max(1.0);
                    assert!(diff < 1e-2 || relative_diff < 1e-4, "Q6_K NEON standard mismatch at index {} for m={}: scalar={}, simd={}, diff={}, rel={}", i, m, c_scalar[i], c_simd[i], diff, relative_diff);
                }

                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    let mut c_dotprod = vec![0.0f32; m * n];
                    let mut c_ref = vec![0.0f32; m * n];
                    let a_quant = if m > 1 {
                        neon::quantize_activations_q8_0_batched(&a, m, k)
                    } else {
                        neon::quantize_activations_q8_0(&a, k)
                    };
                    matmul_q6_k_dotprod_reference(&a_quant, &b_quant, m, n, k, &mut c_ref);
                    neon::matmul_q6_k_dotprod(&a, &b_quant, m, n, k, &mut c_dotprod);
                    for i in 0..c_ref.len() {
                        let diff = (c_ref[i] - c_dotprod[i]).abs();
                        assert!(diff < 1e-3, "Q6_K NEON dotprod mismatch at index {} for m={}: ref={}, dotprod={}, diff={}", i, m, c_ref[i], c_dotprod[i], diff);
                    }
                }
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                matmul_q6_k(&a, &b_quant, m, n, k, &mut c_simd);
                for i in 0..c_scalar.len() {
                    let diff = (c_scalar[i] - c_simd[i]).abs();
                    let relative_diff = diff / c_scalar[i].abs().max(1.0);
                    assert!(
                        diff < 1e-2 || relative_diff < 1e-4,
                        "Q6_K mismatch at index {} for m={}: scalar={}, simd={}, diff={}, rel={}",
                        i,
                        m,
                        c_scalar[i],
                        c_simd[i],
                        diff,
                        relative_diff
                    );
                }
            }
        }
    }
    #[test]
    fn test_q5_k_correctness() {
        let m_cases = vec![1, 3];
        let k = 256;
        let n = 8;
        let b_bytes_len = n * (k / Q5K_BLOCK_SIZE) * Q5K_BLOCK_BYTES;
        let b_quant = generate_random_bytes(b_bytes_len);

        for m in m_cases {
            let a = generate_activations(m, k);
            let mut c_scalar = vec![0.0f32; m * n];
            let mut c_simd = vec![0.0f32; m * n];

            matmul_q5_k_scalar(&a, &b_quant, m, n, k, &mut c_scalar);

            #[cfg(target_arch = "aarch64")]
            {
                neon::matmul_q5_k(&a, &b_quant, m, n, k, &mut c_simd);
                for i in 0..c_scalar.len() {
                    let diff = (c_scalar[i] - c_simd[i]).abs();
                    let relative_diff = diff / c_scalar[i].abs().max(1.0);
                    assert!(diff < 1e-2 || relative_diff < 1e-4, "Q5_K NEON standard mismatch at index {} for m={}: scalar={}, simd={}, diff={}, rel={}", i, m, c_scalar[i], c_simd[i], diff, relative_diff);
                }

                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    let mut c_dotprod = vec![0.0f32; m * n];
                    let mut c_ref = vec![0.0f32; m * n];
                    let a_quant = if m > 1 {
                        neon::quantize_activations_q8_0_batched(&a, m, k)
                    } else {
                        neon::quantize_activations_q8_0(&a, k)
                    };
                    matmul_q5_k_dotprod_reference(&a_quant, &b_quant, m, n, k, &mut c_ref);
                    neon::matmul_q5_k_dotprod(&a, &b_quant, m, n, k, &mut c_dotprod);
                    for i in 0..c_ref.len() {
                        let diff = (c_ref[i] - c_dotprod[i]).abs();
                        let relative_diff = diff / c_ref[i].abs().max(1.0);
                        assert!(diff < 1e-2 || relative_diff < 1e-4,
                            "Q5_K NEON dotprod mismatch at index {} for m={}: ref={}, dotprod={}, diff={}, rel={}",
                            i, m, c_ref[i], c_dotprod[i], diff, relative_diff);
                    }
                }
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                matmul_q5_k(&a, &b_quant, m, n, k, &mut c_simd);
                for i in 0..c_scalar.len() {
                    let diff = (c_scalar[i] - c_simd[i]).abs();
                    let relative_diff = diff / c_scalar[i].abs().max(1.0);
                    assert!(
                        diff < 1e-2 || relative_diff < 1e-4,
                        "Q5_K mismatch at index {} for m={}: scalar={}, simd={}, diff={}, rel={}",
                        i,
                        m,
                        c_scalar[i],
                        c_simd[i],
                        diff,
                        relative_diff
                    );
                }
            }
        }
    }
    fn generate_f16_weight_bytes(n: usize, k: usize) -> Vec<u8> {
        let mut w = vec![0.0f32; n * k];
        for i in 0..w.len() {
            w[i] = ((i * 17 + 5) % 200) as f32 / 50.0 - 2.0;
        }
        f32_to_f16_bytes(&w)
    }

    #[test]
    fn test_f16_correctness() {
        let m_cases = vec![1, 3];
        let ks = vec![32, 128, 256];
        let n = 8;

        for &k in &ks {
            let b_bytes = generate_f16_weight_bytes(n, k);

            let b_f32 = crate::loader::dequant::dequantize(
                &b_bytes,
                crate::loader::gguf::GGUFDtype::F16,
                &[n, k],
            );

            for &m in &m_cases {
                let a = generate_activations(m, k);
                let mut c_ref = vec![0.0f32; m * n];
                let mut c_fused = vec![0.0f32; m * n];

                matmul_f32(&a, &b_f32, m, n, k, &mut c_ref);
                matmul_f16_scalar(&a, &b_bytes, m, n, k, &mut c_fused);

                for i in 0..c_ref.len() {
                    let diff = (c_ref[i] - c_fused[i]).abs();
                    let relative_diff = diff / c_ref[i].abs().max(1.0);
                    assert!(diff < 1e-4 || relative_diff < 1e-5,
                        "F16 scalar mismatch at index {} for m={}, k={}: ref={}, fused={}, diff={}, rel={}",
                        i, m, k, c_ref[i], c_fused[i], diff, relative_diff);
                }

                #[cfg(target_arch = "aarch64")]
                {
                    let mut c_neon = vec![0.0f32; m * n];
                    neon::matmul_f16_neon(&a, &b_bytes, m, n, k, &mut c_neon);
                    for i in 0..c_ref.len() {
                        let diff = (c_ref[i] - c_neon[i]).abs();
                        let relative_diff = diff / c_ref[i].abs().max(1.0);
                        assert!(diff < 1e-3 || relative_diff < 1e-4,
                            "F16 NEON mismatch at index {} for m={}, k={}: ref={}, neon={}, diff={}, rel={}",
                            i, m, k, c_ref[i], c_neon[i], diff, relative_diff);
                    }
                }

                let mut c_dispatch = vec![0.0f32; m * n];
                matmul_f16_impl(&a, &b_bytes, m, n, k, &mut c_dispatch);
                for i in 0..c_ref.len() {
                    let diff = (c_ref[i] - c_dispatch[i]).abs();
                    let relative_diff = diff / c_ref[i].abs().max(1.0);
                    assert!(diff < 1e-4 || relative_diff < 1e-5,
                        "F16 dispatch mismatch at index {} for m={}, k={}: ref={}, dispatch={}, diff={}, rel={}",
                        i, m, k, c_ref[i], c_dispatch[i], diff, relative_diff);
                }
            }
        }
    }

    #[test]
    fn test_q2_k_correctness() {
        let m_cases = vec![1, 3];
        let k = 256;
        let n = 8;
        let b_bytes_len = n * (k / Q2K_BLOCK_SIZE) * Q2K_BLOCK_BYTES;
        let b_quant = generate_random_bytes(b_bytes_len);

        // Reference: dequantize + f32 matmul
        let b_shape = [n, k];
        let b_f32 = crate::loader::dequant::dequantize(
            &b_quant,
            crate::loader::gguf::GGUFDtype::Q2_K,
            &b_shape,
        );

        for m in m_cases {
            let a = generate_activations(m, k);
            let mut c_ref = vec![0.0f32; m * n];
            let mut c_scalar = vec![0.0f32; m * n];

            matmul_f32(&a, &b_f32, m, n, k, &mut c_ref);
            matmul_q2_k_scalar(&a, &b_quant, m, n, k, &mut c_scalar);

            for i in 0..c_ref.len() {
                let diff = (c_ref[i] - c_scalar[i]).abs();
                let relative_diff = diff / c_ref[i].abs().max(1.0);
                assert!(
                    diff < 1e-2 || relative_diff < 1e-4,
                    "Q2_K scalar mismatch at index {} for m={}: ref={}, scalar={}, diff={}, rel={}",
                    i,
                    m,
                    c_ref[i],
                    c_scalar[i],
                    diff,
                    relative_diff
                );
            }

            let mut c_dispatch = vec![0.0f32; m * n];
            matmul_q2_k(&a, &b_quant, m, n, k, &mut c_dispatch);
            for i in 0..c_scalar.len() {
                let diff = (c_scalar[i] - c_dispatch[i]).abs();
                let relative_diff = diff / c_scalar[i].abs().max(1.0);
                assert!(diff < 1e-2 || relative_diff < 1e-4,
                    "Q2_K dispatch mismatch at index {} for m={}: scalar={}, dispatch={}, diff={}, rel={}",
                    i, m, c_scalar[i], c_dispatch[i], diff, relative_diff);
            }

            // Test batched path too
            if m > 1 {
                let mut c_batched = vec![0.0f32; m * n];
                matmul_q2_k_batched_scalar(&a, &b_quant, m, n, k, &mut c_batched);
                for i in 0..c_scalar.len() {
                    let diff = (c_scalar[i] - c_batched[i]).abs();
                    assert!(diff < 1e-5, "Q2_K batched mismatch at index {} for m={}: scalar={}, batched={}, diff={}",
                        i, m, c_scalar[i], c_batched[i], diff);
                }
            }
        }
    }

    #[test]
    fn test_asymmetric_gguf_order_f16() {
        // Prove that quantized_matmul correctly handles asymmetric weights
        // stored in GGUF [k, n] row-major order (k = in_features, n = out_features, n != k).
        // F16 has no block structure, so dequantize is unambiguous — we can compare
        // the [k, n] → matmul_f32_gguf path against a transposed reference.
        let k = 256; // in_features
        let n = 32; // out_features

        // Generate f16 weight data (flat, shape-agnostic)
        let b_bytes = generate_f16_weight_bytes(k, n);

        // Dequantize in [k, n] order → f32 in GGUF row-major order
        let b_f32_gguf = crate::loader::dequant::dequantize(
            &b_bytes,
            crate::loader::gguf::GGUFDtype::F16,
            &[k, n],
        );

        // Transpose from [k, n] to [n, k] order for the reference
        let mut b_f32_ref = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                b_f32_ref[j * k + i] = b_f32_gguf[i * n + j];
            }
        }

        for m in [1, 3] {
            let a = generate_activations(m, k);
            let mut c_ref = vec![0.0f32; m * n];
            let mut c_dut = vec![0.0f32; m * n];

            // Reference: matmul_f32 with data in [n, k] order
            matmul_f32(&a, &b_f32_ref, m, n, k, &mut c_ref);

            // Device-under-test: quantized_matmul with shape [n, k] (asymmetric → n != k)
            // Internally dequantizes with [k, n] and calls matmul_f32_gguf
            quantized_matmul_impl(
                &a,
                m,
                &b_bytes,
                &[n, k],
                crate::loader::gguf::GGUFDtype::F16,
                &mut c_dut,
                None,
            );

            for i in 0..c_ref.len() {
                let diff = (c_ref[i] - c_dut[i]).abs();
                let relative_diff = diff / c_ref[i].abs().max(1.0);
                assert!(diff < 1e-4 || relative_diff < 1e-5,
                    "F16 asymmetric GGUF order mismatch at index {} for m={}: ref={}, dut={}, diff={}, rel={}",
                    i, m, c_ref[i], c_dut[i], diff, relative_diff);
            }
        }
    }

    #[test]
    fn test_q3_k_correctness() {
        let m_cases = vec![1, 3];
        let k = 256;
        let n = 8;
        let b_bytes_len = n * (k / Q3K_BLOCK_SIZE) * Q3K_BLOCK_BYTES;
        let b_quant = generate_random_bytes(b_bytes_len);

        let b_shape = [n, k];
        let b_f32 = crate::loader::dequant::dequantize(
            &b_quant,
            crate::loader::gguf::GGUFDtype::Q3_K,
            &b_shape,
        );

        for m in m_cases {
            let a = generate_activations(m, k);
            let mut c_ref = vec![0.0f32; m * n];
            let mut c_scalar = vec![0.0f32; m * n];

            matmul_f32(&a, &b_f32, m, n, k, &mut c_ref);
            matmul_q3_k_scalar(&a, &b_quant, m, n, k, &mut c_scalar);

            for i in 0..c_ref.len() {
                let diff = (c_ref[i] - c_scalar[i]).abs();
                let relative_diff = diff / c_ref[i].abs().max(1.0);
                assert!(
                    diff < 1e-2 || relative_diff < 1e-4,
                    "Q3_K scalar mismatch at index {} for m={}: ref={}, scalar={}, diff={}, rel={}",
                    i,
                    m,
                    c_ref[i],
                    c_scalar[i],
                    diff,
                    relative_diff
                );
            }

            let mut c_dispatch = vec![0.0f32; m * n];
            matmul_q3_k(&a, &b_quant, m, n, k, &mut c_dispatch);
            for i in 0..c_scalar.len() {
                let diff = (c_scalar[i] - c_dispatch[i]).abs();
                let relative_diff = diff / c_scalar[i].abs().max(1.0);
                assert!(diff < 1e-2 || relative_diff < 1e-4,
                    "Q3_K dispatch mismatch at index {} for m={}: scalar={}, dispatch={}, diff={}, rel={}",
                    i, m, c_scalar[i], c_dispatch[i], diff, relative_diff);
            }

            if m > 1 {
                let mut c_batched = vec![0.0f32; m * n];
                matmul_q3_k_batched_scalar(&a, &b_quant, m, n, k, &mut c_batched);
                for i in 0..c_scalar.len() {
                    let diff = (c_scalar[i] - c_batched[i]).abs();
                    assert!(diff < 1e-5, "Q3_K batched mismatch at index {} for m={}: scalar={}, batched={}, diff={}",
                        i, m, c_scalar[i], c_batched[i], diff);
                }
            }
        }
    }

