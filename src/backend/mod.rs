pub mod backend_mod {
    use crate::graph::Op;
    use crate::tensor::{Shape, Tensor};
    use crate::Error;

    /// Common trait implemented by all computation backends (CPU, CUDA, WGPU).
    ///
    /// Every op from the [`Op`] enum has a dedicated method with a default
    /// implementation that falls back to [`Self::execute`].  Backends that have
    /// a native implementation for a given op override the method; backends
    /// that don't get the generic dispatch for free.
    pub trait Backend {
        // ── Generic dispatch ──────────────────────────────────────────────

        /// Execute a single op given reference inputs.  The catch-all fallback
        /// for every op-specific method below.
        fn execute(&self, op: &Op, inputs: &[&Tensor]) -> Result<Tensor, Error>;

        // ── Element-wise unary ────────────────────────────────────────────

        fn relu(&self, x: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::Relu, &[x])
        }
        fn tanh(&self, x: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::Tanh, &[x])
        }
        fn sigmoid(&self, x: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::Sigmoid, &[x])
        }
        fn exp(&self, x: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::Exp, &[x])
        }
        fn sqrt(&self, x: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::Sqrt, &[x])
        }
        fn neg(&self, x: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::Neg, &[x])
        }

        // ── Element-wise binary ───────────────────────────────────────────

        fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::Add, &[a, b])
        }
        fn sub(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::Sub, &[a, b])
        }
        fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::Mul, &[a, b])
        }
        fn div(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::Div, &[a, b])
        }

        // ── Broadcasting element-wise ─────────────────────────────────────

        fn broadcast_add(&self, a: &Tensor, b: &Tensor, input_shapes: Vec<Shape>) -> Result<Tensor, Error> {
            self.execute(&Op::BroadcastAdd { input_shapes }, &[a, b])
        }
        fn broadcast_mul(&self, a: &Tensor, b: &Tensor, input_shapes: Vec<Shape>) -> Result<Tensor, Error> {
            self.execute(&Op::BroadcastMul { input_shapes }, &[a, b])
        }
        fn broadcast_sub(&self, a: &Tensor, b: &Tensor, input_shapes: Vec<Shape>) -> Result<Tensor, Error> {
            self.execute(&Op::BroadcastSub { input_shapes }, &[a, b])
        }
        fn broadcast_div(&self, a: &Tensor, b: &Tensor, input_shapes: Vec<Shape>) -> Result<Tensor, Error> {
            self.execute(&Op::BroadcastDiv { input_shapes }, &[a, b])
        }

        // ── Linear algebra ────────────────────────────────────────────────

        fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::MatMul, &[a, b])
        }
        fn transpose(&self, x: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::Transpose, &[x])
        }
        fn batched_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::BatchedMatMul, &[a, b])
        }
        fn batched_transpose(&self, x: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::BatchedTranspose, &[x])
        }

        // ── Reduction ─────────────────────────────────────────────────────

        fn sum_all(&self, x: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::SumAll, &[x])
        }
        fn sum_dim(&self, x: &Tensor, axis: usize) -> Result<Tensor, Error> {
            self.execute(&Op::SumDim { axis }, &[x])
        }

        // ── Shape manipulation ────────────────────────────────────────────

        fn reshape(&self, x: &Tensor, shape: Shape) -> Result<Tensor, Error> {
            self.execute(&Op::Reshape { shape }, &[x])
        }
        fn concat(&self, inputs: &[&Tensor], axis: usize) -> Result<Tensor, Error> {
            // use execute with multiple inputs (concrete backend handles the axis)
            self.execute(&Op::Concat { axis }, inputs)
        }
        fn slice(&self, x: &Tensor, axis: usize, start: usize, end: usize) -> Result<Tensor, Error> {
            self.execute(&Op::Slice { axis, start, end }, &[x])
        }

        // ── Activation / normalisation ────────────────────────────────────

        fn softmax(&self, x: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::Softmax, &[x])
        }
        fn step(&self, x: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::Step, &[x])
        }
        fn layer_norm(&self, x: &Tensor, weight: &Tensor, bias: &Tensor, epsilon: f32) -> Result<Tensor, Error> {
            self.execute(&Op::LayerNorm { epsilon }, &[x, weight, bias])
        }
        fn rms_norm(&self, x: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor, Error> {
            self.execute(&Op::RmsNorm { epsilon }, &[x, weight])
        }

        // ── Convolution / pooling ─────────────────────────────────────────

        fn conv2d(&self, x: &Tensor, weight: &Tensor, stride: usize, padding: usize) -> Result<Tensor, Error> {
            self.execute(&Op::Conv2d { stride, padding }, &[x, weight])
        }
        fn max_pool2d(&self, x: &Tensor, pool_size: usize, stride: usize, padding: usize) -> Result<Tensor, Error> {
            self.execute(&Op::MaxPool2d { pool_size, stride, padding }, &[x])
        }
        fn avg_pool2d(&self, x: &Tensor, pool_size: usize, stride: usize, padding: usize) -> Result<Tensor, Error> {
            self.execute(&Op::AvgPool2d { pool_size, stride, padding }, &[x])
        }

        // ── Attention ─────────────────────────────────────────────────────

        fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor, Error> {
            self.execute(&Op::Attention { scale }, &[q, k, v])
        }
        fn causal_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, num_heads: usize) -> Result<Tensor, Error> {
            self.execute(&Op::CausalAttention { scale, num_heads }, &[q, k, v])
        }
        fn multi_head_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, num_heads: usize) -> Result<Tensor, Error> {
            self.execute(&Op::MultiHeadAttention { scale, num_heads }, &[q, k, v])
        }
        fn flash_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor, Error> {
            self.execute(&Op::FlashAttention { scale, causal }, &[q, k, v])
        }

        // ── Type casting ──────────────────────────────────────────────────

        fn cast_f32_to_f16(&self, x: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::CastF32ToF16, &[x])
        }
        fn cast_f16_to_f32(&self, x: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::CastF16ToF32, &[x])
        }
        fn cast_f32_to_bf16(&self, x: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::CastF32ToBF16, &[x])
        }
        fn cast_bf16_to_f32(&self, x: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::CastBF16ToF32, &[x])
        }

        // ── Gradient ops (backward pass) ──────────────────────────────────

        fn softmax_grad(&self, dy: &Tensor, y: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::SoftmaxGrad, &[dy, y])
        }
        fn layer_norm_grad_x(&self, dy: &Tensor, x: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor, Error> {
            self.execute(&Op::LayerNormGradX { epsilon }, &[dy, x, weight])
        }
        fn layer_norm_grad_w(&self, dy: &Tensor, x: &Tensor, epsilon: f32) -> Result<Tensor, Error> {
            self.execute(&Op::LayerNormGradW { epsilon }, &[dy, x])
        }
        fn layer_norm_grad_b(&self, dy: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::LayerNormGradB, &[dy])
        }
        fn conv2d_grad_x(&self, dy: &Tensor, weight: &Tensor, stride: usize, padding: usize) -> Result<Tensor, Error> {
            self.execute(&Op::Conv2dGradX { stride, padding }, &[dy, weight])
        }
        fn conv2d_grad_w(&self, dy: &Tensor, x: &Tensor, stride: usize, padding: usize) -> Result<Tensor, Error> {
            self.execute(&Op::Conv2dGradW { stride, padding }, &[dy, x])
        }
        fn conv2d_grad_b(&self, dy: &Tensor) -> Result<Tensor, Error> {
            self.execute(&Op::Conv2dGradB, &[dy])
        }
        fn max_pool2d_grad(&self, dy: &Tensor, x: &Tensor, pool_size: usize, stride: usize, padding: usize) -> Result<Tensor, Error> {
            self.execute(&Op::MaxPool2dGrad { pool_size, stride, padding }, &[dy, x])
        }
        fn avg_pool2d_grad(&self, dy: &Tensor, x: &Tensor, pool_size: usize, stride: usize, padding: usize) -> Result<Tensor, Error> {
            self.execute(&Op::AvgPool2dGrad { pool_size, stride, padding }, &[dy, x])
        }
        fn slice_grad(&self, dy: &Tensor, axis: usize, start: usize, end: usize) -> Result<Tensor, Error> {
            self.execute(&Op::SliceGrad { axis, start, end }, &[dy])
        }
        fn attention_grad_q(&self, dy: &Tensor, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor, Error> {
            self.execute(&Op::AttentionGradQ { scale }, &[dy, q, k, v])
        }
        fn attention_grad_k(&self, dy: &Tensor, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor, Error> {
            self.execute(&Op::AttentionGradK { scale }, &[dy, q, k, v])
        }
        fn attention_grad_v(&self, dy: &Tensor, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor, Error> {
            self.execute(&Op::AttentionGradV { scale }, &[dy, q, k, v])
        }

        // ── Fused ops ─────────────────────────────────────────────────────

        fn execute_matmul_relu(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
            let temp = self.execute(&Op::MatMul, &[a, b])?;
            self.execute(&Op::Relu, &[&temp])
        }

        fn execute_matmul_add(
            &self,
            a: &Tensor,
            b: &Tensor,
            bias: &Tensor,
        ) -> Result<Tensor, Error> {
            let temp = self.execute(&Op::MatMul, &[a, b])?;
            self.execute(&Op::Add, &[&temp, bias])
        }

        fn execute_matmul_add_relu(
            &self,
            a: &Tensor,
            b: &Tensor,
            bias: &Tensor,
        ) -> Result<Tensor, Error> {
            let temp = self.execute(&Op::MatMul, &[a, b])?;
            let temp2 = self.execute(&Op::Add, &[&temp, bias])?;
            self.execute(&Op::Relu, &[&temp2])
        }
    }
}
pub mod cpu;
pub mod cuda;
pub mod wgpu_backend;

pub use backend_mod::Backend;
pub use cpu::CpuBackend;
pub use cuda::CudaBackend;
pub use wgpu_backend::WgpuBackend;
