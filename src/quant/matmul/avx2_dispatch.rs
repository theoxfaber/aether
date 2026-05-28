// SAFETY: This module contains unsafe dispatch to AVX2/NEON SIMD matmul kernels.
// Each unsafe call is gated behind runtime CPU feature detection.
#![allow(unsafe_code)]
#[cfg(target_arch = "aarch64")]
use crate::quant::matmul::neon;

#[cfg(target_arch = "aarch64")]
fn matmul_q4_k_impl(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        neon::matmul_q4_k_dotprod(a, b, m, n, k, c);
    } else {
        neon::matmul_q4_k(a, b, m, n, k, c);
    }
}
#[cfg(not(target_arch = "aarch64"))]
fn matmul_q4_k_impl(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return unsafe { super::super::avx2::matmul_q4_k(a, b, m, n, k, c) };
    }
    crate::quant::matmul::q4_k::matmul_q4_k_scalar(a, b, m, n, k, c);
}

#[cfg(target_arch = "aarch64")]
fn matmul_q8_0_impl(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        neon::matmul_q8_0_dotprod(a, b, m, n, k, c);
    } else {
        neon::matmul_q8_0(a, b, m, n, k, c);
    }
}
#[cfg(not(target_arch = "aarch64"))]
fn matmul_q8_0_impl(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return unsafe { super::super::avx2::matmul_q8_0(a, b, m, n, k, c) };
    }
    crate::quant::matmul::q8_0::matmul_q8_0_scalar(a, b, m, n, k, c);
}

#[cfg(target_arch = "aarch64")]
fn matmul_q6_k_impl(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        neon::matmul_q6_k_dotprod(a, b, m, n, k, c);
    } else {
        neon::matmul_q6_k(a, b, m, n, k, c);
    }
}
#[cfg(not(target_arch = "aarch64"))]
fn matmul_q6_k_impl(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return unsafe { super::super::avx2::matmul_q6_k(a, b, m, n, k, c) };
    }
    crate::quant::matmul::q6_k::matmul_q6_k_scalar(a, b, m, n, k, c);
}

pub fn matmul_q4_k(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    matmul_q4_k_impl(a, b, m, n, k, c);
}

pub fn matmul_q8_0(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    matmul_q8_0_impl(a, b, m, n, k, c);
}

pub fn matmul_q6_k(a: &[f32], b: &[u8], m: usize, n: usize, k: usize, c: &mut [f32]) {
    matmul_q6_k_impl(a, b, m, n, k, c);
}
