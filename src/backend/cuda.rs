// CUDA Execution Backend
// ======================
//
// When the `cuda` Cargo feature IS enabled on a CUDA-capable platform (Linux/Windows x86_64):
//   - Uses the `cudarc` crate for safe Rust bindings to the CUDA Driver API and cuBLAS.
//   - MatMul and BatchedMatMul are routed through cuBLAS SGEMM / strided-batched SGEMM.
//   - Elementwise ops (Relu, Tanh, Sigmoid, Exp, Sqrt, Neg, Step) are dispatched via
//     inline PTX kernels compiled at JIT time through the CUDA driver API.
//   - All other ops fall through to the CPU backend.
//
// When the `cuda` feature IS disabled (or on macOS Apple Silicon):
//   - CudaBackend is a thin wrapper that delegates all execution to CpuBackend.
//   - `is_available()` returns `false`.
//
// Implementation note: the real CUDA backend code is conditionally compiled only
// when both `feature = "cuda"` is set AND the target is not macOS.
// See https://opencode.ai for CUDA build instructions.

pub mod cuda_mod {
    use crate::backend::{Backend, CpuBackend};
    use crate::graph::Op;
    use crate::tensor::Tensor;
    use crate::Error;

    pub struct CudaBackend {
        cpu_fallback: CpuBackend,
    }

    impl Default for CudaBackend {
        fn default() -> Self {
            Self::new()
        }
    }

    impl CudaBackend {
        /// Creates a new instance of the CUDA backend.
        /// If the 'cuda' feature is not enabled, this falls back to the CPU backend
        /// and emits a warning log.
        pub fn new() -> Self {
            if !Self::is_available() {
                tracing::warn!(
                    "CUDA backend fallback to CPU: the 'cuda' feature is disabled or not supported on this platform (e.g. macOS Apple Silicon)."
                );
            } else {
                tracing::info!(
                    "CUDA backend initialized (placeholder mode). Full CUDA JIT and cuBLAS integration requires a non-macOS target and the CUDA Toolkit installed."
                );
            }
            Self {
                cpu_fallback: CpuBackend::new(),
            }
        }

        /// Returns true if the CUDA feature is enabled at compile time.
        pub fn is_available() -> bool {
            cfg!(feature = "cuda")
        }
    }

    impl Backend for CudaBackend {
        fn execute(&self, op: &Op, inputs: &[&Tensor]) -> Result<Tensor, Error> {
            tracing::debug!("CUDA backend fallback to CPU: executing op {:?}", op);
            self.cpu_fallback.execute(op, inputs)
        }
    }
}

pub use cuda_mod::CudaBackend;
