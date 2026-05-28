use crate::loader::gguf::GGUFDtype;
use std::collections::HashMap;
use std::sync::RwLock;

/// A quantized matmul kernel for a specific [`GGUFDtype`].
#[allow(dead_code)]
pub trait QuantKernel: Send + Sync {
    /// Compute C = A @ B where A is f32 activations and B is quantized weights.
    ///
    /// `a` has shape `[m, k]` (row-major), `b_raw` has quantized bytes in
    /// GGUF block layout, `c` has shape `[m, n]` (row-major output).
    fn matmul(&self, a: &[f32], m: usize, b_raw: &[u8], n: usize, k: usize, c: &mut [f32]);

    /// The GGUF dtype this kernel handles.
    fn dtype(&self) -> GGUFDtype;

    /// Human-readable name (e.g. "Q4_K").
    fn name(&self) -> &'static str;
}

// ── Global kernel registry ────────────────────────────────────────────────

use std::sync::LazyLock;

static REGISTRY: LazyLock<RwLock<HashMap<GGUFDtype, Box<dyn QuantKernel>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Register a quantized matmul kernel for a specific dtype.
///
/// Called once at startup (e.g. in a `ctor` or `lazy_static` initialiser).
/// Panics if a kernel for the same dtype is already registered.
#[allow(dead_code)]
pub fn register_kernel(kernel: Box<dyn QuantKernel>) {
    let dtype = kernel.dtype();
    let mut reg = REGISTRY
        .write()
        .expect("QuantKernel registry lock poisoned");
    if reg.contains_key(&dtype) {
        panic!("QuantKernel for {:?} already registered", dtype);
    }
    reg.insert(dtype, kernel);
}

/// Dispatch a quantized matmul using registered kernels.
///
/// Falls back to dequantize+f32 matmul if no kernel is registered for `dtype`.
pub fn dispatch_matmul(
    a: &[f32],
    m: usize,
    b_raw: &[u8],
    n: usize,
    k: usize,
    dtype: GGUFDtype,
    c: &mut [f32],
) -> bool {
    let reg = REGISTRY.read().expect("QuantKernel registry lock poisoned");
    if let Some(kernel) = reg.get(&dtype) {
        kernel.matmul(a, m, b_raw, n, k, c);
        true
    } else {
        false // caller should fall back to dequant
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestQ8Kernel;
    impl QuantKernel for TestQ8Kernel {
        fn matmul(&self, a: &[f32], _m: usize, b_raw: &[u8], n: usize, k: usize, c: &mut [f32]) {
            let b = crate::loader::dequant::dequantize(b_raw, GGUFDtype::Q8_0, &[k, n]);
            for i in 0.. {
                if i >= a.len() || i >= c.len() {
                    break;
                }
                c[i] = a[i] + b[i];
            }
        }
        fn dtype(&self) -> GGUFDtype {
            GGUFDtype::Q8_0
        }
        fn name(&self) -> &'static str {
            "test_q8"
        }
    }

    #[test]
    fn test_register_and_dispatch() {
        let k = Box::new(TestQ8Kernel);
        register_kernel(k);
        let a = vec![1.0f32; 64];
        let b = crate::quant::requantize(&a, GGUFDtype::Q8_0, &[2, 32]).unwrap();
        let mut c = vec![0.0f32; 64];
        let dispatched = dispatch_matmul(&a, 2, &b, 32, 32, GGUFDtype::Q8_0, &mut c);
        assert!(dispatched, "expected Q8_0 kernel to be dispatched");
    }
}
