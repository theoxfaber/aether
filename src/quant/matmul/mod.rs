#![allow(unused_imports)]

mod common;
mod f16;
mod q2_k;
mod q3_k;
mod q4_k;
mod q5_k;
mod q6_k;
mod q8_0;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

mod avx2_dispatch;
mod dispatch;
pub(crate) mod registry;

#[cfg(test)]
mod tests;

pub use avx2_dispatch::{matmul_q4_k, matmul_q6_k, matmul_q8_0};

pub use dispatch::{
    add_bias, matmul_f16, matmul_f32, matmul_q2_k, matmul_q3_k, matmul_q5_k, quantized_matmul_impl,
    requantize,
};

#[cfg(test)]
pub(crate) use common::{
    decode_f16_scale, get_scale_min_k4, Q2K_BLOCK_BYTES, Q2K_BLOCK_SIZE, Q3K_BLOCK_BYTES,
    Q3K_BLOCK_SIZE, Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE, Q5K_BLOCK_BYTES, Q5K_BLOCK_SIZE,
    Q6K_BLOCK_BYTES, Q6K_BLOCK_SIZE, Q8_BLOCK_BYTES, Q8_BLOCK_SIZE,
};

#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use dispatch::matmul_f16_impl;

#[cfg(test)]
pub(crate) use q2_k::{matmul_q2_k_batched_scalar, matmul_q2_k_scalar};

#[cfg(test)]
pub(crate) use q3_k::{matmul_q3_k_batched_scalar, matmul_q3_k_scalar};

#[cfg(test)]
pub(crate) use q4_k::matmul_q4_k_scalar;

#[cfg(test)]
pub(crate) use q5_k::{matmul_q5_k_batched_scalar, matmul_q5_k_scalar};

#[cfg(test)]
pub(crate) use q6_k::matmul_q6_k_scalar;

#[cfg(test)]
pub(crate) use q8_0::matmul_q8_0_scalar;

#[cfg(test)]
pub(crate) use f16::matmul_f16_scalar;
