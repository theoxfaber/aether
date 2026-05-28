use crate::{Graph, GraphTensor, Shape};

/// Linear layer (Fully Connected / Dense).
pub struct Linear {
    pub weight: GraphTensor,
    pub bias: GraphTensor,
}

impl Linear {
    /// Create a new Linear layer with Xavier-uniform initialization.
    pub fn new(graph: &Graph, in_features: usize, out_features: usize) -> Self {
        let bound = 1.0 / (in_features as f32).sqrt();
        let mut data = Vec::with_capacity(in_features * out_features);
        let mut state = 42u32;
        for _ in 0..(in_features * out_features) {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let r = (state as f32) / (u32::MAX as f32);
            data.push(r * 2.0 * bound - bound);
        }
        let weight = graph.tensor(data, Shape::new(vec![in_features, out_features]));
        let bias = graph.tensor(vec![0.0; out_features], Shape::new(vec![1, out_features]));
        Self { weight, bias }
    }

    /// Forward pass: x @ W + b. Handles 2D and 3D (batched sequence) inputs.
    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let shape = x.shape();
        if shape.ndim() == 3 {
            let b = shape.dims()[0];
            let s = shape.dims()[1];
            let d = shape.dims()[2];
            let out_features = self.weight.shape().dims()[1];
            let x_flat = x.reshape(Shape::new(vec![b * s, d]));
            let out_flat = x_flat.matmul(self.weight.clone()).add(self.bias.clone());
            out_flat.reshape(Shape::new(vec![b, s, out_features]))
        } else {
            x.matmul(self.weight.clone()).add(self.bias.clone())
        }
    }

    /// Forward pass: x @ W. No bias term added.
    pub fn forward_no_bias(&self, x: GraphTensor) -> GraphTensor {
        let shape = x.shape();
        if shape.ndim() == 3 {
            let b = shape.dims()[0];
            let s = shape.dims()[1];
            let d = shape.dims()[2];
            let out_features = self.weight.shape().dims()[1];
            let x_flat = x.reshape(Shape::new(vec![b * s, d]));
            let out_flat = x_flat.matmul(self.weight.clone());
            out_flat.reshape(Shape::new(vec![b, s, out_features]))
        } else {
            x.matmul(self.weight.clone())
        }
    }

    /// Returns the trainable parameters of this layer.
    pub fn parameters(&self) -> Vec<GraphTensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

/// Root Mean Square Normalization (RMSNorm).
pub struct RMSNorm {
    pub weight: GraphTensor,
    pub epsilon: f32,
}

impl RMSNorm {
    pub fn new(graph: &Graph, normalized_shape: Vec<usize>, epsilon: f32) -> Self {
        let size = normalized_shape.iter().product();
        let weight = graph.tensor(vec![1.0; size], Shape::new(normalized_shape));
        Self { weight, epsilon }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        x.rmsnorm(self.weight.clone(), self.epsilon)
    }

    pub fn parameters(&self) -> Vec<GraphTensor> {
        vec![self.weight.clone()]
    }
}

/// Layer Normalization layer.
pub struct LayerNorm {
    pub weight: GraphTensor,
    pub bias: GraphTensor,
    pub epsilon: f32,
}

impl LayerNorm {
    pub fn new(graph: &Graph, normalized_shape: Vec<usize>, epsilon: f32) -> Self {
        let size = normalized_shape.iter().product();
        let weight = graph.tensor(vec![1.0; size], Shape::new(normalized_shape.clone()));
        let bias = graph.tensor(vec![0.0; size], Shape::new(normalized_shape));
        Self {
            weight,
            bias,
            epsilon,
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        x.layernorm(self.weight.clone(), self.bias.clone(), self.epsilon)
    }

    pub fn parameters(&self) -> Vec<GraphTensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

/// KV Cache for autoregressive generation.
pub struct KVCache {
    pub k_cache: Option<GraphTensor>,
    pub v_cache: Option<GraphTensor>,
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            k_cache: None,
            v_cache: None,
        }
    }

    pub fn update(
        &mut self,
        k: GraphTensor,
        v: GraphTensor,
        _graph: &Graph,
    ) -> (GraphTensor, GraphTensor) {
        match (&self.k_cache, &self.v_cache) {
            (Some(kc), Some(vc)) => {
                let k_new = GraphTensor::concat(&[kc.clone(), k], 1);
                let v_new = GraphTensor::concat(&[vc.clone(), v], 1);
                self.k_cache = Some(k_new.clone());
                self.v_cache = Some(v_new.clone());
                (k_new, v_new)
            }
            _ => {
                self.k_cache = Some(k.clone());
                self.v_cache = Some(v.clone());
                (k, v)
            }
        }
    }

    pub fn reset(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
    }
}

/// Rotary Position Embedding (RoPE) applied to Q and K.
/// Uses pre-computed sin/cos values on the host side.
pub fn apply_rope(
    q: GraphTensor,
    k: GraphTensor,
    sin_vals: GraphTensor,
    cos_vals: GraphTensor,
    d_model: usize,
) -> (GraphTensor, GraphTensor) {
    let half_d = d_model / 2;
    let q_shape = q.shape();
    let k_shape = k.shape();
    let seq_len = q_shape.dims()[1];

    let q_flat = q.reshape(Shape::new(vec![seq_len, d_model]));
    let k_flat = k.reshape(Shape::new(vec![seq_len, d_model]));

    let mut q_parts = Vec::new();
    let mut k_parts = Vec::new();
    for i in 0..half_d {
        let q1 = q_flat.slice(1, i, i + 1);
        let q2 = q_flat.slice(1, half_d + i, half_d + i + 1);

        let cos_i = cos_vals.slice(1, i, i + 1);
        let sin_i = sin_vals.slice(1, i, i + 1);

        let q1_rot = q1.mul(cos_i.clone()).sub(q2.clone().mul(sin_i.clone()));
        let q2_rot = q1.mul(sin_i).add(q2.mul(cos_i));

        q_parts.push(q1_rot);
        q_parts.push(q2_rot);
    }
    let q_rope = GraphTensor::concat(&q_parts, 1).reshape(q_shape);

    for i in 0..half_d {
        let k1 = k_flat.slice(1, i, i + 1);
        let k2 = k_flat.slice(1, half_d + i, half_d + i + 1);

        let cos_i = cos_vals.slice(1, i, i + 1);
        let sin_i = sin_vals.slice(1, i, i + 1);

        let k1_rot = k1.mul(cos_i.clone()).sub(k2.clone().mul(sin_i.clone()));
        let k2_rot = k1.mul(sin_i).add(k2.mul(cos_i));

        k_parts.push(k1_rot);
        k_parts.push(k2_rot);
    }
    let k_rope = GraphTensor::concat(&k_parts, 1).reshape(k_shape);

    (q_rope, k_rope)
}

/// Pre-compute RoPE sin/cos values for a given sequence length.
pub fn precompute_rope(
    length: usize,
    d_model: usize,
    base: f32,
    graph: &Graph,
) -> (GraphTensor, GraphTensor) {
    let half_d = d_model / 2;
    let mut sin_vals = Vec::with_capacity(length * half_d);
    let mut cos_vals = Vec::with_capacity(length * half_d);
    for pos in 0..length {
        for i in 0..half_d {
            let theta = pos as f32 * (base.powf(-2.0 * (i as f32) / (d_model as f32)));
            sin_vals.push(theta.sin());
            cos_vals.push(theta.cos());
        }
    }
    let sin_t = graph.tensor(sin_vals, Shape::new(vec![length, half_d]));
    let cos_t = graph.tensor(cos_vals, Shape::new(vec![length, half_d]));
    (sin_t, cos_t)
}

/// A single Llama-style decoder layer with RMSNorm, RoPE, GQA, and SiLU-MLP.
pub struct LlamaDecoderLayer {
    pub input_layernorm: RMSNorm,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub post_attention_layernorm: RMSNorm,
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
    pub d_model: usize,
    pub num_heads: usize,
    pub rope_base: f32,
}

impl LlamaDecoderLayer {
    pub fn new(
        graph: &Graph,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        rope_base: f32,
    ) -> Self {
        Self {
            input_layernorm: RMSNorm::new(graph, vec![d_model], 1e-5),
            q_proj: Linear::new(graph, d_model, d_model),
            k_proj: Linear::new(graph, d_model, d_model),
            v_proj: Linear::new(graph, d_model, d_model),
            o_proj: Linear::new(graph, d_model, d_model),
            post_attention_layernorm: RMSNorm::new(graph, vec![d_model], 1e-5),
            gate_proj: Linear::new(graph, d_model, d_ff),
            up_proj: Linear::new(graph, d_model, d_ff),
            down_proj: Linear::new(graph, d_ff, d_model),
            d_model,
            num_heads,
            rope_base,
        }
    }

    pub fn forward(
        &self,
        x: GraphTensor,
        sin: GraphTensor,
        cos: GraphTensor,
        kv_cache: &mut KVCache,
    ) -> GraphTensor {
        // Pre-RMSNorm attention
        let norm_x = self.input_layernorm.forward(x.clone());

        // QKV projections
        let q = self.q_proj.forward_no_bias(norm_x.clone());
        let k = self.k_proj.forward_no_bias(norm_x.clone());
        let v = self.v_proj.forward_no_bias(norm_x);

        // Apply RoPE to Q and K
        let (q_rope, k_rope) = apply_rope(q, k, sin, cos, self.d_model);

        // Update KV cache
        let (k_full, v_full) = kv_cache.update(k_rope, v.clone(), &x.graph());

        // Scaled dot-product attention
        let scale = 1.0 / ((self.d_model / self.num_heads) as f32).sqrt();
        let attn_out = q_rope.flash_attention(k_full, v_full, scale, true);
        let attn_proj = self.o_proj.forward(attn_out);

        // Residual
        let x_attn = x.add(attn_proj);

        // Post-RMSNorm FFN with SiLU gating
        let norm_x_attn = self.post_attention_layernorm.forward(x_attn.clone());
        let gate = norm_x_attn.mul(norm_x_attn.sigmoid()); // SiLU approximation: x * sigmoid(x)
        let gate_proj = self.gate_proj.forward_no_bias(gate);
        let up_proj = self.up_proj.forward_no_bias(norm_x_attn);
        let mlp_out = self.down_proj.forward_no_bias(gate_proj.mul(up_proj));

        x_attn.add(mlp_out)
    }

    pub fn parameters(&self) -> Vec<GraphTensor> {
        let mut params = Vec::new();
        params.extend(self.input_layernorm.parameters());
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.o_proj.parameters());
        params.extend(self.post_attention_layernorm.parameters());
        params.extend(self.gate_proj.parameters());
        params.extend(self.up_proj.parameters());
        params.extend(self.down_proj.parameters());
        params
    }
}

/// A standard Transformer Block (Pre-LN style).
pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub ln2: LayerNorm,
    pub mlp1: Linear,
    pub mlp2: Linear,
    pub d_model: usize,
}

impl TransformerBlock {
    pub fn new(graph: &Graph, d_model: usize, d_ff: usize) -> Self {
        Self {
            ln1: LayerNorm::new(graph, vec![d_model], 1e-5),
            q_proj: Linear::new(graph, d_model, d_model),
            k_proj: Linear::new(graph, d_model, d_model),
            v_proj: Linear::new(graph, d_model, d_model),
            out_proj: Linear::new(graph, d_model, d_model),
            ln2: LayerNorm::new(graph, vec![d_model], 1e-5),
            mlp1: Linear::new(graph, d_model, d_ff),
            mlp2: Linear::new(graph, d_ff, d_model),
            d_model,
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        self.forward_with_flash(x, false)
    }

    pub fn forward_with_flash(&self, x: GraphTensor, use_flash: bool) -> GraphTensor {
        let shape = x.shape();
        let is_3d = shape.ndim() == 3;
        let x_3d = if !is_3d {
            let s = shape.dims()[0];
            let d = shape.dims()[1];
            x.reshape(Shape::new(vec![1, s, d]))
        } else {
            x.clone()
        };
        let norm_x = self.ln1.forward(x_3d.clone());
        let q = self.q_proj.forward(norm_x.clone());
        let k = self.k_proj.forward(norm_x.clone());
        let v = self.v_proj.forward(norm_x);
        let scale = 1.0 / (self.d_model as f32).sqrt();
        let attn_out = if use_flash {
            q.flash_attention(k, v, scale, false)
        } else {
            q.attention(k, v, scale)
        };
        let attn_proj = self.out_proj.forward(attn_out);
        let x_attn = x_3d.add(attn_proj);
        let norm_x_attn = self.ln2.forward(x_attn.clone());
        let mlp_h = self.mlp1.forward(norm_x_attn).relu();
        let mlp_out = self.mlp2.forward(mlp_h);
        let out_3d = x_attn.add(mlp_out);
        if !is_3d {
            let s = shape.dims()[0];
            let d = shape.dims()[1];
            out_3d.reshape(Shape::new(vec![s, d]))
        } else {
            out_3d
        }
    }

    pub fn parameters(&self) -> Vec<GraphTensor> {
        let mut params = Vec::new();
        params.extend(self.ln1.parameters());
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params.extend(self.ln2.parameters());
        params.extend(self.mlp1.parameters());
        params.extend(self.mlp2.parameters());
        params
    }
}
