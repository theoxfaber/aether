use crate::tensor::Tensor;
/// BLAS-accelerated matrix multiplication using Apple's Accelerate framework.
///
/// When the `accelerate` feature is enabled on macOS, this replaces the
/// ndarray-based matmul with a direct `cblas_sgemm` call.
/// The Accelerate BLAS is configured for multithreaded dispatch.
use crate::Error;

/// Initialize multithreaded BLAS. Call once at startup.
pub fn init() {
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        // Let Accelerate use all available CPU cores
        // 0 = use default thread count (all cores)
        std::env::set_var("VECLIB_MAXIMUM_THREADS", "0");
        // Enable Apple AMX coprocessor for matrix ops
        std::env::set_var("VECLIB_ENABLE_AMX", "1");
    }
}

#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub mod accelerate_blas {
    use crate::tensor::{Shape, Tensor};
    use crate::Error;

    // LP64 BLAS interface (standard on macOS without ACCELERATE_NEW_LAPACK)
    extern "C" {
        fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANS: i32 = 111;
    const CBLAS_TRANS: i32 = 112;

    /// C = alpha * op(A) * op(B) + beta * C
    /// op(A) = A (no transpose) or A^T (transpose)
    /// op(B) = B (no transpose) or B^T (transpose)
    ///
    /// Dimensions:
    ///   op(A): m × k
    ///   op(B): k × n
    ///   C:     m × n
    ///
    /// # Safety
    ///
    /// - `a` must point to a valid f32 array of size `lda * k` (NoTrans) or `lda * m` (Trans)
    /// - `b` must point to a valid f32 array of size `ldb * n` (NoTrans) or `ldb * k` (Trans)
    /// - `c` must point to a valid f32 array of size `ldc * n`
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemm(
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: *const f32,
        lda: usize,
        b: *const f32,
        ldb: usize,
        beta: f32,
        c: *mut f32,
        ldc: usize,
    ) {
        let ta = if trans_a { CBLAS_TRANS } else { CBLAS_NO_TRANS };
        let tb = if trans_b { CBLAS_TRANS } else { CBLAS_NO_TRANS };
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            ta,
            tb,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            lda as i32,
            b,
            ldb as i32,
            beta,
            c,
            ldc as i32,
        );
    }

    pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, Error> {
        let lhs_shape = lhs.shape().dims();
        let rhs_shape = rhs.shape().dims();

        if lhs_shape.len() != 2 || rhs_shape.len() != 2 || lhs_shape[1] != rhs_shape[0] {
            return Err(Error::ShapeMismatch(format!(
                "MatMul shape mismatch: {:?} and {:?}",
                lhs.shape(),
                rhs.shape()
            )));
        }

        let m = lhs_shape[0] as i32;
        let k = lhs_shape[1] as i32;
        let n = rhs_shape[1] as i32;

        let mut out_data = vec![0.0f32; (m * n) as usize];

        // SAFETY: `lhs.data()` is a `&[f32]` of length `m * k`, `rhs.data()` is
        // a `&[f32]` of length `k * n`, and `out_data` is a freshly allocated
        // `Vec<f32>` of length `m * n`. The strides `lda=k`, `ldb=n`, `ldc=n`
        // match the row-major layout of these buffers. `cblas_sgemm` follows the
        // C ABI and reads/writes through these pointers with the given dimensions.
        unsafe {
            sgemm(
                false,
                false,
                m as usize,
                n as usize,
                k as usize,
                1.0,
                lhs.data().as_ptr(),
                k as usize,
                rhs.data().as_ptr(),
                n as usize,
                0.0,
                out_data.as_mut_ptr(),
                n as usize,
            );
        }

        Ok(Tensor::new(
            out_data,
            Shape::new(vec![m as usize, n as usize]),
        ))
    }
}

#[cfg(not(all(feature = "accelerate", target_os = "macos")))]
pub mod fallback {
    use crate::tensor::{Shape, Tensor};
    use crate::Error;

    pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, Error> {
        let lhs_shape = lhs.shape().dims();
        let rhs_shape = rhs.shape().dims();

        if lhs_shape.len() != 2 || rhs_shape.len() != 2 || lhs_shape[1] != rhs_shape[0] {
            return Err(Error::ShapeMismatch(format!(
                "MatMul shape mismatch: {:?} and {:?}",
                lhs.shape(),
                rhs.shape()
            )));
        }

        let lhs_view = ndarray::ArrayView2::from_shape((lhs_shape[0], lhs_shape[1]), lhs.data())
            .map_err(|e| Error::ExecutionError(format!("ndarray shape error: {}", e)))?;
        let rhs_view = ndarray::ArrayView2::from_shape((rhs_shape[0], rhs_shape[1]), rhs.data())
            .map_err(|e| Error::ExecutionError(format!("ndarray shape error: {}", e)))?;

        let out_arr = lhs_view.dot(&rhs_view);
        let out_data = out_arr.iter().cloned().collect::<Vec<f32>>();
        let out_shape = Shape::new(vec![lhs_shape[0], rhs_shape[1]]);

        Ok(Tensor::new(out_data, out_shape))
    }

    /// Fallback: same as matmul (no transpose optimization without Accelerate)
    pub unsafe fn sgemm(
        _trans_a: bool,
        _trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        _alpha: f32,
        a: *const f32,
        _lda: usize,
        b: *const f32,
        _ldb: usize,
        _beta: f32,
        c: *mut f32,
        ldc: usize,
    ) {
        // Naive fallback for non-Accelerate platforms
        let a_slice = std::slice::from_raw_parts(a, m * k);
        let b_slice = std::slice::from_raw_parts(b, k * n);
        let c_slice = std::slice::from_raw_parts_mut(c, m * n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..k {
                    sum += a_slice[i * k + kk] * b_slice[kk * n + j];
                }
                c_slice[i * ldc + j] = sum;
            }
        }
    }
}

/// Dispatch to the best available matmul implementation.
pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, Error> {
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        accelerate_blas::matmul(lhs, rhs)
    }
    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
    {
        fallback::matmul(lhs, rhs)
    }
}

/// BLAS SGEMM: C = alpha * op(A) * op(B) + beta * C
///
/// Safe wrapper that dispatches to Accelerate sgemm on macOS,
/// or a naive fallback elsewhere.
///
/// # Safety
///
/// - a must point to a valid f32 array of size lda * k (NoTrans) or lda * m (Trans)
/// - b must point to a valid f32 array of size ldb * n (NoTrans) or ldb * k (Trans)
/// - c must point to a valid f32 array of size ldc * n
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        accelerate_blas::sgemm(
            trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
    {
        fallback::sgemm(
            trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}
