#[cfg(target_arch = "x86_64")]
pub mod avx2;
pub mod matmul;
pub use crate::loader::dequant;
pub use matmul::{
    add_bias, matmul_f16, matmul_f32, matmul_q2_k, matmul_q3_k, matmul_q4_k, matmul_q5_k,
    matmul_q6_k, matmul_q8_0, quantized_matmul_impl, requantize,
};
