pub mod wgpu_backend_mod {
    use crate::backend::Backend;
    use crate::graph::Op;
    use crate::tensor::{Shape, Tensor};
    use crate::Error;
    use std::sync::{Arc, OnceLock};
    use wgpu::util::DeviceExt;

    include!("wgsl_shaders.rs");

    fn attention_shader_src(max_seq: u32) -> String {
        format!(
            r#"
        struct AttentionParams {{
            n_tokens: u32,
            n_heads: u32,
            n_kv_heads: u32,
            head_dim: u32,
            seq_len: u32,
            kv_groups: u32,
            scale: f32,
            cache_offset: u32,
        }}

        @group(0) @binding(0) var<storage, read> q: array<f32>;
        @group(0) @binding(1) var<storage, read> k_cache: array<f32>;
        @group(0) @binding(2) var<storage, read> v_cache: array<f32>;
        @group(0) @binding(3) var<storage, read_write> output: array<f32>;
        @group(0) @binding(4) var<uniform> params: AttentionParams;

        var<workgroup> shared_q: array<f32, 128>;
        var<workgroup> shared_partial: array<f32, 128>;
        var<workgroup> shared_scores: array<f32, {}>;

        @compute @workgroup_size(128)
        fn main(
            @builtin(local_invocation_id) lid: vec3<u32>,
            @builtin(workgroup_id) wgid: vec3<u32>,
        ) {{
            let token_idx = wgid.x;
            let head_idx = wgid.y;

            if (token_idx >= params.n_tokens || head_idx >= params.n_heads) {{
                return;
            }}

            let kv_head_idx = head_idx / params.kv_groups;
            let dim = lid.x;
            let hd = params.head_dim;
            let seq = params.seq_len;
            let pos_stride = params.n_kv_heads * hd;
            let cache_base = params.cache_offset;

            // 1. Load Q vector into shared memory
            let q_base = token_idx * params.n_heads * hd + head_idx * hd;
            if (dim < hd) {{
                shared_q[dim] = q[q_base + dim];
            }}
            workgroupBarrier();

            // Causal position: cache_offset is the absolute sequence position
            // of the first token in the current batch. For decode (1 token),
            // cache_offset = pos, token_idx = 0 → causal_pos = pos, attending
            // to all seq positions. For prefill batch, each token at batch
            // index t attends to positions 0..=(cache_offset + t).
            let causal_pos = cache_base + token_idx;

            // 2. Compute Q·K scores for each KV position
            var i = 0u;
            while (i < seq && i <= causal_pos) {{
                let k_idx = i * pos_stride + kv_head_idx * hd + dim;
                if (dim < hd) {{
                    shared_partial[dim] = shared_q[dim] * k_cache[k_idx];
                }} else {{
                    shared_partial[dim] = 0.0;
                }}
                workgroupBarrier();

                // Tree reduction: sum partial products (128 → 64 → ... → 1)
                var offset = 64u;
                while (offset > 0u) {{
                    if (dim < offset) {{
                        shared_partial[dim] += shared_partial[dim + offset];
                    }}
                    workgroupBarrier();
                    offset /= 2u;
                }}

                if (dim == 0u) {{
                    shared_scores[i] = shared_partial[0] * params.scale;
                }}
                workgroupBarrier();

                i++;
            }}

            // 3. Softmax: distributed across all 128 threads via tree reduction
            // Each thread processes a strided chunk of shared_scores
            var local_max: f32 = -3.402823e+38;
            var j = dim;
            while (j < seq && j <= causal_pos) {{
                let s = shared_scores[j];
                if (s > local_max) {{ local_max = s; }}
                j += 128u;
            }}
            shared_partial[dim] = local_max;
            workgroupBarrier();

            // Tree reduction: find global max
            var offset = 64u;
            while (offset > 0u) {{
                if (dim < offset) {{
                    let other = shared_partial[dim + offset];
                    if (other > shared_partial[dim]) {{
                        shared_partial[dim] = other;
                    }}
                }}
                workgroupBarrier();
                offset /= 2u;
            }}
            let max_val = shared_partial[0];
            workgroupBarrier();

            // Compute exp and partial sum
            var local_sum = 0.0;
            j = dim;
            while (j < seq && j <= causal_pos) {{
                let e = exp(shared_scores[j] - max_val);
                shared_scores[j] = e;
                local_sum += e;
                j += 128u;
            }}
            shared_partial[dim] = local_sum;
            workgroupBarrier();

            // Tree reduction: sum
            offset = 64u;
            while (offset > 0u) {{
                if (dim < offset) {{
                    shared_partial[dim] += shared_partial[dim + offset];
                }}
                workgroupBarrier();
                offset /= 2u;
            }}
            let sum_val = shared_partial[0];
            let inv_sum = select(1.0 / sum_val, 0.0, sum_val == 0.0);
            workgroupBarrier();

            // Normalize
            j = dim;
            while (j < seq && j <= causal_pos) {{
                shared_scores[j] = shared_scores[j] * inv_sum;
                j += 128u;
            }}
            workgroupBarrier();

            // 4. Weighted sum of V values
            var output_val = 0.0;
            for (var p = 0u; p < seq && p <= causal_pos; p++) {{
                let v_idx = p * pos_stride + kv_head_idx * hd + dim;
                if (dim < hd) {{
                    output_val += shared_scores[p] * v_cache[v_idx];
                }}
            }}

            // 5. Write output
            if (dim < hd) {{
                let out_base = token_idx * params.n_heads * hd + head_idx * hd;
                output[out_base + dim] = output_val;
            }}
        }}
    "#,
            max_seq
        )
    }

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct MatmulDims {
        m: u32,
        n: u32,
        k: u32,
        padding: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct BatchedMatmulDims {
        b: u32,
        m: u32,
        n: u32,
        k: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct BatchedTransposeDims {
        b: u32,
        m: u32,
        n: u32,
        padding: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct PoolDims {
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        out_h: u32,
        out_w: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        padding_h: u32,
        padding_w: u32,
    }

    /// Uniform layout for pooling-gradient shaders (identical fields to PoolDims).
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct PoolGradDims {
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        out_h: u32,
        out_w: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        padding_h: u32,
        padding_w: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct RoPEParams {
        n_tokens: u32,
        n_heads: u32,
        head_dim: u32,
        start_pos: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct AttentionParams {
        n_tokens: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        kv_groups: u32,
        scale: f32,
        cache_offset: u32,
    }

    struct WgpuBackendInner {
        device: wgpu::Device,
        queue: wgpu::Queue,
        relu_pipeline: wgpu::ComputePipeline,
        matmul_pipeline: wgpu::ComputePipeline,
        tiny_matmul_pipeline: wgpu::ComputePipeline,
        add_pipeline: wgpu::ComputePipeline,
        matmul_relu_pipeline: wgpu::ComputePipeline,
        matmul_add_pipeline: wgpu::ComputePipeline,
        matmul_add_relu_pipeline: wgpu::ComputePipeline,
        batched_matmul_pipeline: wgpu::ComputePipeline,
        batched_transpose_pipeline: wgpu::ComputePipeline,
        max_pool_pipeline: wgpu::ComputePipeline,
        avg_pool_pipeline: wgpu::ComputePipeline,
        max_pool_grad_pipeline: wgpu::ComputePipeline,
        avg_pool_grad_pipeline: wgpu::ComputePipeline,
        matmul_q8_0_pipeline: wgpu::ComputePipeline,
        matmul_q4_k_pipeline: wgpu::ComputePipeline,
        dequant_q4_k_pipeline: wgpu::ComputePipeline,
        dequant_q5_k_pipeline: wgpu::ComputePipeline,
        dequant_q6_k_pipeline: wgpu::ComputePipeline,
        dequant_q8_0_pipeline: wgpu::ComputePipeline,
        memory_bandwidth_gbps: f64,
        compute_flops_gflops: f64,
        /// Pre-allocated 4-byte dummy bias used whenever Conv2d has no bias tensor.
        /// Eliminates one device.create_buffer() call per Conv2d invocation.
        dummy_bias: wgpu::Buffer,
        /// 256-byte scratch uniform buffer reused across all view-dispatch calls.
        uniform_scratch: wgpu::Buffer,
    }

    #[derive(Clone)]
    pub struct WgpuBackend {
        inner: Arc<WgpuBackendInner>,
    }

    static WGPU_BACKEND: OnceLock<Result<WgpuBackend, String>> = OnceLock::new();

    impl WgpuBackend {
        /// Run a closure within a WGPU error scope, returning any errors
        /// instead of panicking.  Used during init to gracefully fall back to
        /// CPU when the GPU path is unavailable.
        pub fn try_init_with<T>(f: impl FnOnce(&Self) -> Result<T, Error>) -> Result<T, Error> {
            let backend = Self::get_or_init()?;
            backend
                .inner
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation);
            backend
                .inner
                .device
                .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let result = f(&backend);
            let oom = futures::executor::block_on(backend.inner.device.pop_error_scope());
            let val = futures::executor::block_on(backend.inner.device.pop_error_scope());
            if let Some(e) = oom.or(val) {
                return Err(Error::ExecutionError(format!("WGPU init error: {e:?}")));
            }
            result
        }

        pub fn get_or_init() -> Result<Self, Error> {
            let res =
                WGPU_BACKEND.get_or_init(|| Self::new_inner().map_err(|e| format!("{:?}", e)));
            match res {
                Ok(backend) => Ok(backend.clone()),
                Err(err) => Err(Error::ExecutionError(err.clone())),
            }
        }

        pub fn execute_silu_mul_buffers(
            &self,
            gate_buf: &wgpu::Buffer,
            up_buf: &wgpu::Buffer,
            num_rows: u32,
            d_ff: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("SiLU Mul Encoder"),
                    });
            let result = self.execute_silu_mul_buffers_with_encoder(
                &mut encoder,
                gate_buf,
                up_buf,
                num_rows,
                d_ff,
            )?;
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(result)
        }

        pub fn execute_silu_mul_buffers_with_encoder(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            gate_buf: &wgpu::Buffer,
            up_buf: &wgpu::Buffer,
            num_rows: u32,
            d_ff: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let size_bytes = (num_rows * d_ff) as usize * std::mem::size_of::<f32>();
            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SiLU Mul Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let silu_mul_wgsl = r#"
                @group(0) @binding(0) var<storage, read> gate: array<f32>;
                @group(0) @binding(1) var<storage, read> up: array<f32>;
                @group(0) @binding(2) var<storage, read_write> output: array<f32>;

                fn sigmoid_f32(x: f32) -> f32 {
                    return 1.0 / (1.0 + exp(-x));
                }

                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let idx = id.x;
                    if (idx < arrayLength(&gate)) {
                        let g = gate[idx];
                        output[idx] = g * sigmoid_f32(g) * up[idx];
                    }
                }
            "#;

            let pipeline = crate::codegen::cache::PipelineCache::global().get_or_compile(
                "silu_mul_pipeline",
                silu_mul_wgsl,
                &self.inner.device,
            );

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SiLU Mul Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: gate_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: up_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: output_buf.as_entire_binding(),
                        },
                    ],
                });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SiLU Mul Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let total_elements = num_rows * d_ff;
                compute_pass.dispatch_workgroups(total_elements.div_ceil(256), 1, 1);
            }
            Ok(output_buf)
        }

        fn new_inner() -> Result<Self, Error> {
            let instance = wgpu::Instance::default();

            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
                .ok_or_else(|| {
                    Error::ExecutionError("Failed to find a compatible WGPU adapter".to_string())
                })?;

            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Aether WGPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            ))
            .map_err(|e| Error::ExecutionError(format!("Failed to create WGPU device: {:?}", e)))?;

            let relu_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ReLU Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(RELU_SHADER_SRC)),
            });

            let add_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Add Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(ADD_SHADER_SRC)),
            });

            let matmul_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MatMul Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MATMUL_SHADER_SRC)),
            });

            let tiny_matmul_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Tiny MatMul Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    TINY_MATMUL_SHADER_SRC,
                )),
            });

            let matmul_relu_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MatMul ReLU Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    MATMUL_RELU_SHADER_SRC,
                )),
            });

            let matmul_add_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MatMul Add Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MATMUL_ADD_SHADER_SRC)),
            });

            let matmul_add_relu_module =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("MatMul Add ReLU Shader Module"),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                        MATMUL_ADD_RELU_SHADER_SRC,
                    )),
                });

            let batched_matmul_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("BatchedMatMul Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    BATCHED_MATMUL_SHADER_SRC,
                )),
            });

            let batched_transpose_module =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("BatchedTranspose Shader Module"),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                        BATCHED_TRANSPOSE_SHADER_SRC,
                    )),
                });

            let max_pool_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MaxPool2d Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MAX_POOL_SHADER_SRC)),
            });

            let avg_pool_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("AvgPool2d Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(AVG_POOL_SHADER_SRC)),
            });

            let relu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ReLU Compute Pipeline"),
                layout: None,
                module: &relu_module,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

            let add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Add Compute Pipeline"),
                layout: None,
                module: &add_module,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

            let matmul_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MatMul Compute Pipeline"),
                    layout: None,
                    module: &matmul_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

            let tiny_matmul_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Tiny MatMul Compute Pipeline"),
                    layout: None,
                    module: &tiny_matmul_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

            let matmul_q8_0_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MatMul Q8_0 Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    MATMUL_Q8_0_SHADER_SRC,
                )),
            });
            let matmul_q4_k_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MatMul Q4_K Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    MATMUL_Q4_K_SHADER_SRC,
                )),
            });

            let matmul_q8_0_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MatMul Q8_0 Compute Pipeline"),
                    layout: None,
                    module: &matmul_q8_0_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });
            let matmul_q4_k_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MatMul Q4_K Compute Pipeline"),
                    layout: None,
                    module: &matmul_q4_k_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

            let matmul_relu_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MatMul ReLU Compute Pipeline"),
                    layout: None,
                    module: &matmul_relu_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

            let matmul_add_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MatMul Add Compute Pipeline"),
                    layout: None,
                    module: &matmul_add_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

            let matmul_add_relu_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MatMul Add ReLU Compute Pipeline"),
                    layout: None,
                    module: &matmul_add_relu_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

            let batched_matmul_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("BatchedMatMul Compute Pipeline"),
                    layout: None,
                    module: &batched_matmul_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

            let batched_transpose_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("BatchedTranspose Compute Pipeline"),
                    layout: None,
                    module: &batched_transpose_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

            let max_pool_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MaxPool2d Compute Pipeline"),
                    layout: None,
                    module: &max_pool_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

            let avg_pool_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("AvgPool2d Compute Pipeline"),
                    layout: None,
                    module: &avg_pool_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

            // ── Pooling backward pipelines ─────────────────────────────────────────
            let max_pool_grad_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MaxPool2dGrad Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    MAX_POOL_GRAD_SHADER_SRC,
                )),
            });
            let avg_pool_grad_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("AvgPool2dGrad Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    AVG_POOL_GRAD_SHADER_SRC,
                )),
            });
            let dequant_q4_k_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Dequant Q4_K Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    DEQUANT_Q4_K_SHADER_SRC,
                )),
            });
            let dequant_q5_k_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Dequant Q5_K Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    DEQUANT_Q5_K_SHADER_SRC,
                )),
            });
            let dequant_q6_k_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Dequant Q6_K Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    DEQUANT_Q6_K_SHADER_SRC,
                )),
            });
            let dequant_q8_0_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Dequant Q8_0 Shader Module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    DEQUANT_Q8_0_SHADER_SRC,
                )),
            });

            let dequant_q4_k_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Dequant Q4_K Compute Pipeline"),
                    layout: None,
                    module: &dequant_q4_k_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });
            let dequant_q5_k_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Dequant Q5_K Compute Pipeline"),
                    layout: None,
                    module: &dequant_q5_k_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });
            let dequant_q6_k_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Dequant Q6_K Compute Pipeline"),
                    layout: None,
                    module: &dequant_q6_k_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });
            let dequant_q8_0_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Dequant Q8_0 Compute Pipeline"),
                    layout: None,
                    module: &dequant_q8_0_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

            let max_pool_grad_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MaxPool2dGrad Compute Pipeline"),
                    layout: None,
                    module: &max_pool_grad_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });
            let avg_pool_grad_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("AvgPool2dGrad Compute Pipeline"),
                    layout: None,
                    module: &avg_pool_grad_module,
                    entry_point: "main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

            let memory_bandwidth_gbps =
                Self::calibrate_memory_bandwidth(&device, &queue).unwrap_or(100.0);
            let compute_flops_gflops =
                Self::calibrate_compute_flops(&device, &queue, &matmul_pipeline).unwrap_or(3600.0);

            let dummy_bias = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Conv2d Dummy Bias (static)"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let uniform_scratch = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Uniform Scratch Buffer (static 256B)"),
                size: 256,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            Ok(Self {
                inner: Arc::new(WgpuBackendInner {
                    device,
                    queue,
                    relu_pipeline,
                    matmul_pipeline,
                    tiny_matmul_pipeline,
                    matmul_q8_0_pipeline,
                    matmul_q4_k_pipeline,
                    add_pipeline,
                    matmul_relu_pipeline,
                    matmul_add_pipeline,
                    matmul_add_relu_pipeline,
                    batched_matmul_pipeline,
                    batched_transpose_pipeline,
                    max_pool_pipeline,
                    avg_pool_pipeline,
                    max_pool_grad_pipeline,
                    avg_pool_grad_pipeline,
                    dequant_q4_k_pipeline,
                    dequant_q5_k_pipeline,
                    dequant_q6_k_pipeline,
                    dequant_q8_0_pipeline,
                    memory_bandwidth_gbps,
                    compute_flops_gflops,
                    dummy_bias,
                    uniform_scratch,
                }),
            })
        }

        fn calibrate_memory_bandwidth(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
        ) -> Result<f64, Error> {
            let size = 8 * 1024 * 1024; // 8 MB
            let data = vec![0u8; size];
            let src_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Calibration Src"),
                contents: &data,
                usage: wgpu::BufferUsages::COPY_SRC,
            });
            let dst_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Calibration Dst"),
                size: size as u64,
                usage: wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Warmup
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(&src_buf, 0, &dst_buf, 0, size as u64);
            queue.submit(std::iter::once(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);

            let start = std::time::Instant::now();
            let iterations = 5;
            for _ in 0..iterations {
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                encoder.copy_buffer_to_buffer(&src_buf, 0, &dst_buf, 0, size as u64);
                queue.submit(std::iter::once(encoder.finish()));
            }
            device.poll(wgpu::Maintain::Wait);
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed <= 0.0 {
                return Err(Error::ExecutionError(
                    "Calibration elapsed time is zero".to_string(),
                ));
            }

            let total_bytes = (size * iterations) as f64;
            let gbps = (total_bytes / elapsed) / 1e9;
            Ok(gbps)
        }

        fn calibrate_compute_flops(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            matmul_pipeline: &wgpu::ComputePipeline,
        ) -> Result<f64, Error> {
            let size = 256;
            let data = vec![1.0f32; size * size];
            let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Calib A"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Calib B"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let c_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Calib C"),
                size: (size * size * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            let dims = MatmulDims {
                m: size as u32,
                n: size as u32,
                k: size as u32,
                padding: 0,
            };
            let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Calib Dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bind_group_layout = matmul_pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Calib BG"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: c_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dims_buf.as_entire_binding(),
                    },
                ],
            });

            // Warmup
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(matmul_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let workgroups = (size as u32).div_ceil(64);
                compute_pass.dispatch_workgroups(workgroups, workgroups, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);

            let start = std::time::Instant::now();
            let iterations = 5;
            for _ in 0..iterations {
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(matmul_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    let workgroups = (size as u32).div_ceil(64);
                    compute_pass.dispatch_workgroups(workgroups, workgroups, 1);
                }
                queue.submit(std::iter::once(encoder.finish()));
            }
            device.poll(wgpu::Maintain::Wait);
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed <= 0.0 {
                return Err(Error::ExecutionError(
                    "Calibration elapsed time is zero".to_string(),
                ));
            }

            let total_flops = 2.0 * (size as f64).powi(3) * iterations as f64;
            let gflops = (total_flops / elapsed) / 1e9;
            Ok(gflops)
        }

        pub fn create_buffer_with_data(
            &self,
            data: &[f32],
            usage: wgpu::BufferUsages,
        ) -> wgpu::Buffer {
            self.inner
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Data Buffer"),
                    contents: bytemuck::cast_slice(data),
                    usage,
                })
        }

        pub fn create_quant_buffer(&self, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
            self.inner
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Quantized Weight Buffer"),
                    contents: data,
                    usage,
                })
        }

        pub fn create_device_buffer(
            &self,
            size: u64,
            usage: wgpu::BufferUsages,
            label: &str,
        ) -> wgpu::Buffer {
            self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        }

        pub fn copy_buffers(
            &self,
            src: &wgpu::Buffer,
            dst: &wgpu::Buffer,
            src_offset: u64,
            dst_offset: u64,
            size: u64,
        ) {
            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Buffer Copy"),
                    });
            self.copy_buffers_with_encoder(&mut encoder, src, dst, src_offset, dst_offset, size);
            self.inner.queue.submit(Some(encoder.finish()));
        }

        pub fn copy_buffers_with_encoder(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            src: &wgpu::Buffer,
            dst: &wgpu::Buffer,
            src_offset: u64,
            dst_offset: u64,
            size: u64,
        ) {
            encoder.copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size);
        }

        pub fn create_encoder(&self, label: &str) -> wgpu::CommandEncoder {
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
        }

        pub fn submit_encoder(&self, encoder: wgpu::CommandEncoder) {
            self.inner.queue.submit(Some(encoder.finish()));
        }

        pub fn read_buffer(
            &self,
            buffer: &wgpu::Buffer,
            size_bytes: usize,
        ) -> Result<Vec<f32>, Error> {
            let staging = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Read Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Readback Encoder"),
                    });
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size_bytes as u64);
            self.inner.queue.submit(Some(encoder.finish()));

            let staging_slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            staging_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            self.inner.device.poll(wgpu::Maintain::Wait);
            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    return Err(Error::ExecutionError(format!(
                        "Buffer mapping failed: {:?}",
                        e
                    )))
                }
                Err(_) => {
                    return Err(Error::ExecutionError(
                        "Channel disconnected during map_async".into(),
                    ));
                }
            }

            let data_view = staging_slice.get_mapped_range();
            let data = bytemuck::cast_slice::<u8, f32>(&data_view).to_vec();
            drop(data_view);
            staging.unmap();

            Ok(data)
        }

        pub fn execute_rope_buffers(
            &self,
            buf: &wgpu::Buffer,
            n_tokens: u32,
            n_heads: u32,
            head_dim: u32,
            start_pos: u32,
            sin_buf: &wgpu::Buffer,
            cos_buf: &wgpu::Buffer,
        ) -> Result<(), Error> {
            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("RoPE Encoder"),
                    });
            self.execute_rope_buffers_with_encoder(
                &mut encoder,
                buf,
                n_tokens,
                n_heads,
                head_dim,
                start_pos,
                sin_buf,
                cos_buf,
            )?;
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(())
        }

        pub fn execute_rope_buffers_with_encoder(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            buf: &wgpu::Buffer,
            n_tokens: u32,
            n_heads: u32,
            head_dim: u32,
            start_pos: u32,
            sin_buf: &wgpu::Buffer,
            cos_buf: &wgpu::Buffer,
        ) -> Result<(), Error> {
            let params = RoPEParams {
                n_tokens,
                n_heads,
                head_dim,
                start_pos,
            };
            let params_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("RoPE Params Uniform"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let pipeline = crate::codegen::cache::PipelineCache::global().get_or_compile(
                "rope_pipeline",
                ROPE_SHADER_SRC,
                &self.inner.device,
            );

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RoPE Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: sin_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: cos_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("RoPE Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let total = n_tokens * n_heads * (head_dim / 2);
                compute_pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
            }
            Ok(())
        }

        pub fn execute_attention_buffers(
            &self,
            q_buf: &wgpu::Buffer,
            k_cache: &wgpu::Buffer,
            v_cache: &wgpu::Buffer,
            output_buf: &wgpu::Buffer,
            n_tokens: u32,
            n_heads: u32,
            n_kv_heads: u32,
            head_dim: u32,
            seq_len: u32,
            max_seq: u32,
            cache_offset: u32,
        ) -> Result<(), Error> {
            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Attention Encoder"),
                    });
            self.execute_attention_buffers_with_encoder(
                &mut encoder,
                q_buf,
                k_cache,
                v_cache,
                output_buf,
                n_tokens,
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len,
                max_seq,
                cache_offset,
            )?;
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(())
        }

        pub fn execute_attention_buffers_with_encoder(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            q_buf: &wgpu::Buffer,
            k_cache: &wgpu::Buffer,
            v_cache: &wgpu::Buffer,
            output_buf: &wgpu::Buffer,
            n_tokens: u32,
            n_heads: u32,
            n_kv_heads: u32,
            head_dim: u32,
            seq_len: u32,
            max_seq: u32,
            cache_offset: u32,
        ) -> Result<(), Error> {
            let kv_groups = n_heads / n_kv_heads;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let params = AttentionParams {
                n_tokens,
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len,
                kv_groups,
                scale,
                cache_offset,
            };
            let params_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Attention Params Uniform"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let shader_src = attention_shader_src(max_seq);
            let cache_key = format!("attention_pipeline_{}", max_seq);
            let pipeline = crate::codegen::cache::PipelineCache::global().get_or_compile(
                &cache_key,
                &shader_src,
                &self.inner.device,
            );

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Attention Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: q_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: k_cache.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: v_cache.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: output_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Attention Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(n_tokens, n_heads, 1);
            }
            Ok(())
        }

        pub fn execute_matmul_relu_gpu(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
            let lhs_shape = a.shape().dims();
            let rhs_shape = b.shape().dims();
            if lhs_shape.len() != 2 || rhs_shape.len() != 2 || lhs_shape[1] != rhs_shape[0] {
                return Err(Error::ShapeMismatch(format!(
                    "MatMul shape mismatch: {:?} and {:?}",
                    a.shape(),
                    b.shape()
                )));
            }

            let m = lhs_shape[0] as u32;
            let k = lhs_shape[1] as u32;
            let n = rhs_shape[1] as u32;

            let num_elements = (m * n) as usize;
            let size_bytes = num_elements * std::mem::size_of::<f32>();

            let a_buf = self.create_buffer_with_data(a.data(), wgpu::BufferUsages::STORAGE);
            let b_buf = self.create_buffer_with_data(b.data(), wgpu::BufferUsages::STORAGE);
            let c_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MatMul ReLU Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let dims = MatmulDims {
                m,
                n,
                k,
                padding: 0,
            };
            let dims_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("MatMul ReLU Dims Uniform Buffer"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = self.inner.matmul_relu_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MatMul ReLU Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: c_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("MatMul ReLU Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MatMul ReLU Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.inner.matmul_relu_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroups_x = n.div_ceil(64);
                let workgroups_y = m.div_ceil(64);
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));

            let out_data = self.read_buffer(&c_buf, size_bytes)?;
            let out_shape = Shape::new(vec![lhs_shape[0], rhs_shape[1]]);
            Ok(Tensor::new(out_data, out_shape))
        }

        pub fn execute_matmul_add_gpu(
            &self,
            a: &Tensor,
            b: &Tensor,
            bias: &Tensor,
        ) -> Result<Tensor, Error> {
            let lhs_shape = a.shape().dims();
            let rhs_shape = b.shape().dims();
            if lhs_shape.len() != 2 || rhs_shape.len() != 2 || lhs_shape[1] != rhs_shape[0] {
                return Err(Error::ShapeMismatch(format!(
                    "MatMul shape mismatch: {:?} and {:?}",
                    a.shape(),
                    b.shape()
                )));
            }

            let m = lhs_shape[0] as u32;
            let k = lhs_shape[1] as u32;
            let n = rhs_shape[1] as u32;

            let num_elements = (m * n) as usize;
            let size_bytes = num_elements * std::mem::size_of::<f32>();

            let a_buf = self.create_buffer_with_data(a.data(), wgpu::BufferUsages::STORAGE);
            let b_buf = self.create_buffer_with_data(b.data(), wgpu::BufferUsages::STORAGE);
            let bias_buf = self.create_buffer_with_data(bias.data(), wgpu::BufferUsages::STORAGE);
            let c_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MatMul Add Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let dims = MatmulDims {
                m,
                n,
                k,
                padding: 0,
            };
            let dims_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("MatMul Add Dims Uniform Buffer"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = self.inner.matmul_add_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MatMul Add Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: c_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dims_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: bias_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("MatMul Add Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MatMul Add Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.inner.matmul_add_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroups_x = n.div_ceil(64);
                let workgroups_y = m.div_ceil(64);
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));

            let out_data = self.read_buffer(&c_buf, size_bytes)?;
            let out_shape = Shape::new(vec![lhs_shape[0], rhs_shape[1]]);
            Ok(Tensor::new(out_data, out_shape))
        }

        pub fn execute_matmul_add_relu_gpu(
            &self,
            a: &Tensor,
            b: &Tensor,
            bias: &Tensor,
        ) -> Result<Tensor, Error> {
            let lhs_shape = a.shape().dims();
            let rhs_shape = b.shape().dims();
            if lhs_shape.len() != 2 || rhs_shape.len() != 2 || lhs_shape[1] != rhs_shape[0] {
                return Err(Error::ShapeMismatch(format!(
                    "MatMul shape mismatch: {:?} and {:?}",
                    a.shape(),
                    b.shape()
                )));
            }

            let m = lhs_shape[0] as u32;
            let k = lhs_shape[1] as u32;
            let n = rhs_shape[1] as u32;

            let num_elements = (m * n) as usize;
            let size_bytes = num_elements * std::mem::size_of::<f32>();

            let a_buf = self.create_buffer_with_data(a.data(), wgpu::BufferUsages::STORAGE);
            let b_buf = self.create_buffer_with_data(b.data(), wgpu::BufferUsages::STORAGE);
            let bias_buf = self.create_buffer_with_data(bias.data(), wgpu::BufferUsages::STORAGE);
            let c_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MatMul Add ReLU Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let dims = MatmulDims {
                m,
                n,
                k,
                padding: 0,
            };
            let dims_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("MatMul Add ReLU Dims Uniform Buffer"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = self.inner.matmul_add_relu_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MatMul Add ReLU Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: c_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dims_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: bias_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("MatMul Add ReLU Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MatMul Add ReLU Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.inner.matmul_add_relu_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroups_x = n.div_ceil(64);
                let workgroups_y = m.div_ceil(64);
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));

            let out_data = self.read_buffer(&c_buf, size_bytes)?;
            let out_shape = Shape::new(vec![lhs_shape[0], rhs_shape[1]]);
            Ok(Tensor::new(out_data, out_shape))
        }

        pub fn device(&self) -> &wgpu::Device {
            &self.inner.device
        }

        pub fn queue(&self) -> &wgpu::Queue {
            &self.inner.queue
        }

        pub fn memory_bandwidth_gbps(&self) -> f64 {
            self.inner.memory_bandwidth_gbps
        }

        pub fn compute_flops_gflops(&self) -> f64 {
            self.inner.compute_flops_gflops
        }

        /// Returns a reference to the shared pre-allocated dummy bias buffer.
        /// Used by Conv2d when no bias tensor is provided, avoiding a device.create_buffer() call.
        pub fn dummy_bias(&self) -> &wgpu::Buffer {
            &self.inner.dummy_bias
        }

        /// Returns a reference to the shared 256-byte uniform scratch buffer.
        pub fn uniform_scratch(&self) -> &wgpu::Buffer {
            &self.inner.uniform_scratch
        }

        // ── Zero-allocation view-based dispatch ───────────────────────────────────

        /// Generic zero-allocation GPU pipeline dispatcher.
        ///
        /// Writes output directly into `output_view` (a sub-slice of the static GPU arena).
        /// Uses `uniform_buf` for shader params: writes `uniform_data` bytes into it via
        /// `queue.write_buffer` — zero GPU buffer allocations.
        ///
        /// # Safety
        /// Input and output views must refer to non-overlapping regions of the GPU arena
        /// (enforced by the static memory planner's interval-coloring allocator).
        pub fn run_pipeline_views(
            &self,
            pipeline: &wgpu::ComputePipeline,
            inputs: &[&crate::memory::registry::GpuBufferView],
            output: &crate::memory::registry::GpuBufferView,
            uniform_data: Option<&[u8]>,
            uniform_buf: &wgpu::Buffer,
            workgroups: (u32, u32, u32),
        ) {
            // Write uniform params into the reusable scratch buffer (zero allocation).
            if let Some(data) = uniform_data {
                self.inner.queue.write_buffer(uniform_buf, 0, data);
            }

            let bind_group_layout = pipeline.get_bind_group_layout(0);

            // Build binding entries: all inputs first, then output, then uniform.
            let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(inputs.len() + 2);
            for (i, view) in inputs.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: view.as_binding(),
                });
            }
            entries.push(wgpu::BindGroupEntry {
                binding: inputs.len() as u32,
                resource: output.as_binding(),
            });
            if uniform_data.is_some() {
                entries.push(wgpu::BindGroupEntry {
                    binding: (inputs.len() + 1) as u32,
                    resource: uniform_buf.as_entire_binding(),
                });
            }

            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("run_pipeline_views BindGroup"),
                    layout: &bind_group_layout,
                    entries: &entries,
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("run_pipeline_views Encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("run_pipeline_views Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
            }
            self.inner.queue.submit(Some(encoder.finish()));
        }

        /// Execute ReLU in-place on an output view backed by the GPU arena (zero allocation).
        pub fn execute_relu_view(
            &self,
            input_view: &crate::memory::registry::GpuBufferView,
            output_view: &crate::memory::registry::GpuBufferView,
            num_elements: usize,
        ) {
            let bind_group_layout = self.inner.relu_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ReLU View BindGroup"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_view.as_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_view.as_binding(),
                        },
                    ],
                });
            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("ReLU View Encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ReLU View Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.inner.relu_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let wg = (num_elements as u32).div_ceil(256);
                pass.dispatch_workgroups(wg, 1, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
        }

        /// Execute Add on two input views, writing into an output view (zero allocation).
        pub fn execute_add_view(
            &self,
            lhs_view: &crate::memory::registry::GpuBufferView,
            rhs_view: &crate::memory::registry::GpuBufferView,
            output_view: &crate::memory::registry::GpuBufferView,
            num_elements: usize,
        ) {
            let bind_group_layout = self.inner.add_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Add View BindGroup"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: lhs_view.as_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: rhs_view.as_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: output_view.as_binding(),
                        },
                    ],
                });
            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Add View Encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Add View Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.inner.add_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let total = (num_elements as u32).div_ceil(256);
                pass.dispatch_workgroups(total.min(65535), total.div_ceil(65535), 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
        }

        /// Execute MatMul writing into a pre-existing output view (zero allocation).
        pub fn execute_matmul_view(
            &self,
            a_view: &crate::memory::registry::GpuBufferView,
            b_view: &crate::memory::registry::GpuBufferView,
            out_view: &crate::memory::registry::GpuBufferView,
            m: u32,
            n: u32,
            k: u32,
        ) {
            let dims = MatmulDims {
                m,
                n,
                k,
                padding: 0,
            };
            self.inner.queue.write_buffer(
                &self.inner.uniform_scratch,
                0,
                bytemuck::bytes_of(&dims),
            );

            let bind_group_layout = self.inner.matmul_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MatMul View BindGroup"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a_view.as_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_view.as_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_view.as_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.inner.uniform_scratch.as_entire_binding(),
                        },
                    ],
                });
            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("MatMul View Encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MatMul View Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.inner.matmul_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(n.div_ceil(16), m.div_ceil(16), 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
        }

        /// Execute BatchedMatMul writing into a pre-existing output view (zero allocation).
        pub fn execute_batched_matmul_view(
            &self,
            a_view: &crate::memory::registry::GpuBufferView,
            b_view: &crate::memory::registry::GpuBufferView,
            out_view: &crate::memory::registry::GpuBufferView,
            b: u32,
            m: u32,
            n: u32,
            k: u32,
        ) {
            let dims = BatchedMatmulDims { b, m, n, k };
            self.inner.queue.write_buffer(
                &self.inner.uniform_scratch,
                0,
                bytemuck::bytes_of(&dims),
            );

            let bind_group_layout = self.inner.batched_matmul_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("BatchedMatMul View BindGroup"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a_view.as_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_view.as_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_view.as_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.inner.uniform_scratch.as_entire_binding(),
                        },
                    ],
                });
            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("BatchedMatMul View Encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("BatchedMatMul View Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.inner.batched_matmul_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let wg_x = n.div_ceil(64);
                let wg_y = m.div_ceil(64);
                pass.dispatch_workgroups(wg_x, wg_y, b);
            }
            self.inner.queue.submit(Some(encoder.finish()));
        }

        /// Execute fused MatMul+ReLU writing into a pre-existing output view (zero allocation).
        pub fn execute_matmul_relu_view(
            &self,
            a_view: &crate::memory::registry::GpuBufferView,
            b_view: &crate::memory::registry::GpuBufferView,
            out_view: &crate::memory::registry::GpuBufferView,
            m: u32,
            n: u32,
            k: u32,
        ) {
            let dims = MatmulDims {
                m,
                n,
                k,
                padding: 0,
            };
            self.inner.queue.write_buffer(
                &self.inner.uniform_scratch,
                0,
                bytemuck::bytes_of(&dims),
            );

            let bind_group_layout = self.inner.matmul_relu_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MatMulReLU View BindGroup"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a_view.as_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_view.as_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_view.as_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.inner.uniform_scratch.as_entire_binding(),
                        },
                    ],
                });
            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("MatMulReLU View Encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MatMulReLU View Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.inner.matmul_relu_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(n.div_ceil(16), m.div_ceil(16), 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
        }

        pub fn execute_relu_buffer(
            &self,
            input_buf: &wgpu::Buffer,
            num_elements: usize,
        ) -> Result<wgpu::Buffer, Error> {
            let size_bytes = num_elements * std::mem::size_of::<f32>();
            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Relu Output Buffer (from buffer)"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group_layout = self.inner.relu_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Relu Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Relu Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Relu Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.inner.relu_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let total_workgroups = (num_elements as u32).div_ceil(256);
                let workgroups_x = total_workgroups.min(65535);
                let workgroups_y = total_workgroups.div_ceil(65535);
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(output_buf)
        }

        pub fn execute_add_buffers(
            &self,
            lhs_buf: &wgpu::Buffer,
            rhs_buf: &wgpu::Buffer,
            num_elements: usize,
        ) -> Result<wgpu::Buffer, Error> {
            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Add Encoder"),
                    });
            let result = self.execute_add_buffers_with_encoder(
                &mut encoder,
                lhs_buf,
                rhs_buf,
                num_elements,
            )?;
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(result)
        }

        pub fn execute_add_buffers_with_encoder(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            lhs_buf: &wgpu::Buffer,
            rhs_buf: &wgpu::Buffer,
            num_elements: usize,
        ) -> Result<wgpu::Buffer, Error> {
            let size_bytes = num_elements * std::mem::size_of::<f32>();
            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Add Output Buffer (from buffers)"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group_layout = self.inner.add_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Add Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: lhs_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: rhs_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: output_buf.as_entire_binding(),
                        },
                    ],
                });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Add Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.inner.add_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let total_workgroups = (num_elements as u32).div_ceil(256);
                let workgroups_x = total_workgroups.min(65535);
                let workgroups_y = total_workgroups.div_ceil(65535);
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            Ok(output_buf)
        }

        pub fn dequantize_on_gpu(
            &self,
            quant_bytes: &[u8],
            dtype: crate::loader::gguf::GGUFDtype,
            shape: &[usize],
        ) -> Result<wgpu::Buffer, Error> {
            let num_elements = shape.iter().product::<usize>();
            let output_size = num_elements * 4;

            let mut padded_bytes = quant_bytes.to_vec();
            while !padded_bytes.len().is_multiple_of(4) {
                padded_bytes.push(0);
            }
            let input_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Quantized Input Buffer"),
                        contents: &padded_bytes,
                        usage: wgpu::BufferUsages::STORAGE,
                    });

            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dequantized Output Buffer"),
                size: output_size as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let pipeline = match dtype {
                crate::loader::gguf::GGUFDtype::Q4_K => &self.inner.dequant_q4_k_pipeline,
                crate::loader::gguf::GGUFDtype::Q5_K => &self.inner.dequant_q5_k_pipeline,
                crate::loader::gguf::GGUFDtype::Q6_K => &self.inner.dequant_q6_k_pipeline,
                crate::loader::gguf::GGUFDtype::Q8_0 => &self.inner.dequant_q8_0_pipeline,
                _ => {
                    return Err(Error::ExecutionError(format!(
                        "Unsupported GPU dequantization dtype: {:?}",
                        dtype
                    )))
                }
            };

            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Dequant Bind Group"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Dequant Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Dequant Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let workgroups = num_elements.div_ceil(256) as u32;
                compute_pass.dispatch_workgroups(workgroups, 1, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));

            Ok(output_buf)
        }

        pub fn execute_matmul_buffers(
            &self,
            a_buf: &wgpu::Buffer,
            b_buf: &wgpu::Buffer,
            m: u32,
            n: u32,
            k: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("MatMul Encoder"),
                    });
            let result =
                self.execute_matmul_buffers_with_encoder(&mut encoder, a_buf, b_buf, m, n, k)?;
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(result)
        }

        pub fn execute_matmul_buffers_with_encoder(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            a_buf: &wgpu::Buffer,
            b_buf: &wgpu::Buffer,
            m: u32,
            n: u32,
            k: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let size_bytes = (m * n) as usize * std::mem::size_of::<f32>();
            let c_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MatMul Output Buffer (from buffers)"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let dims = MatmulDims {
                m,
                n,
                k,
                padding: 0,
            };
            let dims_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("MatMul Dims Uniform Buffer"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let use_tiny = m <= 32;
            let pipeline = if use_tiny {
                &self.inner.tiny_matmul_pipeline
            } else {
                &self.inner.matmul_pipeline
            };

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MatMul Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: c_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MatMul Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                if use_tiny {
                    let total = (m * n).div_ceil(256);
                    compute_pass.dispatch_workgroups(total, 1, 1);
                } else {
                    let workgroups_x = n.div_ceil(64);
                    let workgroups_y = m.div_ceil(64);
                    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
                }
            }
            Ok(c_buf)
        }

        /// Fused Q8_0 dequant + matmul. B buffer holds raw Q8_0 block data
        /// as padded `u32` slices. Output is f32 [M, N].
        pub fn execute_matmul_q8_0_buffers_with_encoder(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            a_buf: &wgpu::Buffer,
            b_buf: &wgpu::Buffer,
            m: u32,
            n: u32,
            k: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let size_bytes = (m * n) as usize * std::mem::size_of::<f32>();
            let c_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MatMul Q8_0 Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let dims = MatmulDims {
                m,
                n,
                k,
                padding: 0,
            };
            let dims_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("MatMul Q8_0 Dims Uniform Buffer"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let pipeline = &self.inner.matmul_q8_0_pipeline;
            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MatMul Q8_0 Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: c_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MatMul Q8_0 Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let total = (m * n).div_ceil(256);
                compute_pass.dispatch_workgroups(total, 1, 1);
            }
            Ok(c_buf)
        }

        /// Fused Q4_K dequant + matmul. B buffer holds raw Q4_K block data
        /// as padded `u32` slices. Output is f32 [M, N].
        pub fn execute_matmul_q4_k_buffers_with_encoder(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            a_buf: &wgpu::Buffer,
            b_buf: &wgpu::Buffer,
            m: u32,
            n: u32,
            k: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let size_bytes = (m * n) as usize * std::mem::size_of::<f32>();
            let c_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MatMul Q4_K Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let dims = MatmulDims {
                m,
                n,
                k,
                padding: 0,
            };
            let dims_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("MatMul Q4_K Dims Uniform Buffer"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let pipeline = &self.inner.matmul_q4_k_pipeline;
            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MatMul Q4_K Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: c_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MatMul Q4_K Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let total = (m * n).div_ceil(256);
                compute_pass.dispatch_workgroups(total, 1, 1);
            }
            Ok(c_buf)
        }

        pub fn execute_matmul_relu_buffers(
            &self,
            a_buf: &wgpu::Buffer,
            b_buf: &wgpu::Buffer,
            m: u32,
            n: u32,
            k: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let size_bytes = (m * n) as usize * std::mem::size_of::<f32>();
            let c_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MatMul ReLU Output Buffer (from buffers)"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let dims = MatmulDims {
                m,
                n,
                k,
                padding: 0,
            };
            let dims_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("MatMul ReLU Dims Uniform Buffer"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = self.inner.matmul_relu_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MatMul ReLU Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: c_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("MatMul ReLU Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MatMul ReLU Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.inner.matmul_relu_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroups_x = n.div_ceil(64);
                let workgroups_y = m.div_ceil(64);
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(c_buf)
        }

        pub fn execute_matmul_add_buffers(
            &self,
            a_buf: &wgpu::Buffer,
            b_buf: &wgpu::Buffer,
            bias_buf: &wgpu::Buffer,
            m: u32,
            n: u32,
            k: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let size_bytes = (m * n) as usize * std::mem::size_of::<f32>();
            let c_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MatMul Add Output Buffer (from buffers)"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let dims = MatmulDims {
                m,
                n,
                k,
                padding: 0,
            };
            let dims_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("MatMul Add Dims Uniform Buffer"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = self.inner.matmul_add_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MatMul Add Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: c_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dims_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: bias_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("MatMul Add Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MatMul Add Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.inner.matmul_add_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroups_x = n.div_ceil(64);
                let workgroups_y = m.div_ceil(64);
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(c_buf)
        }

        pub fn execute_matmul_add_relu_buffers(
            &self,
            a_buf: &wgpu::Buffer,
            b_buf: &wgpu::Buffer,
            bias_buf: &wgpu::Buffer,
            m: u32,
            n: u32,
            k: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let size_bytes = (m * n) as usize * std::mem::size_of::<f32>();
            let c_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MatMul Add ReLU Output Buffer (from buffers)"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let dims = MatmulDims {
                m,
                n,
                k,
                padding: 0,
            };
            let dims_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("MatMul Add ReLU Dims Uniform Buffer"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = self.inner.matmul_add_relu_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MatMul Add ReLU Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: c_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dims_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: bias_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("MatMul Add ReLU Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MatMul Add ReLU Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.inner.matmul_add_relu_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroups_x = n.div_ceil(64);
                let workgroups_y = m.div_ceil(64);
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(c_buf)
        }

        pub fn execute_batched_matmul_buffers(
            &self,
            a_buf: &wgpu::Buffer,
            b_buf: &wgpu::Buffer,
            b: u32,
            m: u32,
            n: u32,
            k: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let size_bytes = (b * m * n) as usize * std::mem::size_of::<f32>();
            let c_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("BatchedMatMul Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let dims = BatchedMatmulDims { b, m, n, k };
            let dims_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("BatchedMatMul Dims Uniform Buffer"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = self.inner.batched_matmul_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("BatchedMatMul Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: a_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: c_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("BatchedMatMul Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("BatchedMatMul Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.inner.batched_matmul_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroups_x = n.div_ceil(64);
                let workgroups_y = m.div_ceil(64);
                let workgroups_z = b;
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(c_buf)
        }

        pub fn execute_batched_transpose_buffers(
            &self,
            input_buf: &wgpu::Buffer,
            b: u32,
            m: u32,
            n: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let size_bytes = (b * m * n) as usize * std::mem::size_of::<f32>();
            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("BatchedTranspose Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let dims = BatchedTransposeDims {
                b,
                m,
                n,
                padding: 0,
            };
            let dims_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("BatchedTranspose Dims Uniform Buffer"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = self
                .inner
                .batched_transpose_pipeline
                .get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("BatchedTranspose Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("BatchedTranspose Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("BatchedTranspose Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.inner.batched_transpose_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroups_x = n.div_ceil(16);
                let workgroups_y = m.div_ceil(16);
                let workgroups_z = b;
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(output_buf)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn execute_max_pool2d_buffers(
            &self,
            input_buf: &wgpu::Buffer,
            n: u32,
            c: u32,
            h: u32,
            w: u32,
            out_h: u32,
            out_w: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            ph: u32,
            pw: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let size_bytes = (n * c * out_h * out_w) as usize * std::mem::size_of::<f32>();
            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MaxPool2d Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let dims = PoolDims {
                n,
                c,
                h,
                w,
                out_h,
                out_w,
                kernel_h: kh,
                kernel_w: kw,
                stride_h: sh,
                stride_w: sw,
                padding_h: ph,
                padding_w: pw,
            };
            let dims_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("MaxPool2d Dims Uniform Buffer"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = self.inner.max_pool_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MaxPool2d Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("MaxPool2d Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MaxPool2d Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.inner.max_pool_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroups_x = out_w.div_ceil(16);
                let workgroups_y = out_h.div_ceil(16);
                let workgroups_z = n * c;
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(output_buf)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn execute_avg_pool2d_buffers(
            &self,
            input_buf: &wgpu::Buffer,
            n: u32,
            c: u32,
            h: u32,
            w: u32,
            out_h: u32,
            out_w: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            ph: u32,
            pw: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let size_bytes = (n * c * out_h * out_w) as usize * std::mem::size_of::<f32>();
            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("AvgPool2d Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let dims = PoolDims {
                n,
                c,
                h,
                w,
                out_h,
                out_w,
                kernel_h: kh,
                kernel_w: kw,
                stride_h: sh,
                stride_w: sw,
                padding_h: ph,
                padding_w: pw,
            };
            let dims_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("AvgPool2d Dims Uniform Buffer"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = self.inner.avg_pool_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("AvgPool2d Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("AvgPool2d Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("AvgPool2d Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.inner.avg_pool_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroups_x = out_w.div_ceil(16);
                let workgroups_y = out_h.div_ceil(16);
                let workgroups_z = n * c;
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(output_buf)
        }

        // ── Pooling backward GPU implementations ──────────────────────────────────

        /// Backward pass of MaxPool2d on the GPU.
        /// Gathers gradients by checking which input elements were the argmax of
        /// output windows that overlap them.
        ///
        /// Parameters mirror execute_max_pool2d_buffers:
        ///   dy_buf  – gradient of the pool output   [N, C, out_H, out_W]
        ///   x_buf   – original pool input (for argmax recompute) [N, C, H, W]
        #[allow(clippy::too_many_arguments)]
        pub fn execute_max_pool2d_grad_buffers(
            &self,
            dy_buf: &wgpu::Buffer,
            x_buf: &wgpu::Buffer,
            n: u32,
            c: u32,
            h: u32,
            w: u32,
            out_h: u32,
            out_w: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            ph: u32,
            pw: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let dx_size_bytes = (n * c * h * w) as usize * std::mem::size_of::<f32>();
            let dx_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MaxPool2dGrad dx Buffer"),
                size: dx_size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let dims = PoolGradDims {
                n,
                c,
                h,
                w,
                out_h,
                out_w,
                kernel_h: kh,
                kernel_w: kw,
                stride_h: sh,
                stride_w: sw,
                padding_h: ph,
                padding_w: pw,
            };
            let dims_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("MaxPool2dGrad Dims Uniform"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = self.inner.max_pool_grad_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MaxPool2dGrad Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: dy_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: x_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: dx_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("MaxPool2dGrad Encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MaxPool2dGrad Compute Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.inner.max_pool_grad_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                // Dispatch grid matches input dimensions: W x H x (N * C)
                let wgs_x = w.div_ceil(16);
                let wgs_y = h.div_ceil(16);
                let wgs_z = n * c;
                pass.dispatch_workgroups(wgs_x, wgs_y, wgs_z);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(dx_buf)
        }

        /// Backward pass of AvgPool2d on the GPU.
        /// Gathers gradients by summing dy / count from all output windows
        /// overlapping each input pixel.
        #[allow(clippy::too_many_arguments)]
        pub fn execute_avg_pool2d_grad_buffers(
            &self,
            dy_buf: &wgpu::Buffer,
            n: u32,
            c: u32,
            h: u32,
            w: u32,
            out_h: u32,
            out_w: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            ph: u32,
            pw: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let dx_size_bytes = (n * c * h * w) as usize * std::mem::size_of::<f32>();
            let dx_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("AvgPool2dGrad dx Buffer"),
                size: dx_size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let dims = PoolGradDims {
                n,
                c,
                h,
                w,
                out_h,
                out_w,
                kernel_h: kh,
                kernel_w: kw,
                stride_h: sh,
                stride_w: sw,
                padding_h: ph,
                padding_w: pw,
            };
            let dims_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("AvgPool2dGrad Dims Uniform"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = self.inner.avg_pool_grad_pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("AvgPool2dGrad Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: dy_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: dx_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("AvgPool2dGrad Encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("AvgPool2dGrad Compute Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.inner.avg_pool_grad_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                // Dispatch grid matches input dimensions: W x H x (N * C)
                let wgs_x = w.div_ceil(16);
                let wgs_y = h.div_ceil(16);
                let wgs_z = n * c;
                pass.dispatch_workgroups(wgs_x, wgs_y, wgs_z);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(dx_buf)
        }

        /// Execute matrix transpose on the GPU.
        pub fn execute_transpose_buffers(
            &self,
            input_buf: &wgpu::Buffer,
            m: u32,
            n: u32,
        ) -> Result<wgpu::Buffer, Error> {
            let size_bytes = (m * n) as usize * std::mem::size_of::<f32>();
            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Transpose Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let transpose_wgsl = r#"
                struct TransposeParams {
                    m: u32,
                    n: u32,
                }
                @group(0) @binding(0) var<storage, read> input: array<f32>;
                @group(0) @binding(1) var<storage, read_write> output: array<f32>;
                @group(0) @binding(2) var<uniform> params: TransposeParams;

                @compute @workgroup_size(16, 16)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let r = id.y;
                    let c = id.x;
                    if r < params.m && c < params.n {
                        output[c * params.m + r] = input[r * params.n + c];
                    }
                }
            "#;

            let pipeline = crate::codegen::cache::PipelineCache::global().get_or_compile(
                "transpose_pipeline",
                transpose_wgsl,
                &self.inner.device,
            );

            #[repr(C)]
            #[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
            struct TransposeParams {
                m: u32,
                n: u32,
            }
            let params = TransposeParams { m, n };
            let params_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Transpose Params Uniform"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Transpose Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Transpose Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Transpose Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let w_x = n.div_ceil(16);
                let w_y = m.div_ceil(16);
                compute_pass.dispatch_workgroups(w_x, w_y, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(output_buf)
        }

        /// Sum all elements of a tensor on the GPU.
        pub fn execute_sum_all_buffers(
            &self,
            input_buf: &wgpu::Buffer,
        ) -> Result<wgpu::Buffer, Error> {
            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SumAll Output Buffer"),
                size: std::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let sum_all_wgsl = r#"
                @group(0) @binding(0) var<storage, read> input: array<f32>;
                @group(0) @binding(1) var<storage, read_write> output: array<f32>;

                @compute @workgroup_size(1, 1, 1)
                fn main() {
                    var sum = 0.0;
                    let len = arrayLength(&input);
                    for (var i = 0u; i < len; i++) {
                        sum += input[i];
                    }
                    output[0] = sum;
                }
            "#;

            let pipeline = crate::codegen::cache::PipelineCache::global().get_or_compile(
                "sum_all_pipeline",
                sum_all_wgsl,
                &self.inner.device,
            );

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SumAll Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("SumAll Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SumAll Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(1, 1, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(output_buf)
        }

        /// Sum elements of a tensor along a specific axis on the GPU.
        pub fn execute_sum_dim_buffers(
            &self,
            input_buf: &wgpu::Buffer,
            input_shape: &Shape,
            axis: usize,
        ) -> Result<wgpu::Buffer, Error> {
            let mut out_dims = input_shape.dims().to_vec();
            out_dims[axis] = 1;
            let output_shape = Shape::new(out_dims);
            let num_elements_out = output_shape.num_elements();
            let size_bytes = num_elements_out * std::mem::size_of::<f32>();

            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SumDim Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let sum_dim_wgsl = r#"
                struct SumDimParams {
                    info: vec4<u32>, // info[0] = out_len, info[1] = in_len, info[2] = reduced_axis, info[3] = padding
                    out_shape: vec4<u32>,
                    in_shape: vec4<u32>,
                    in_strides: vec4<u32>,
                }
                @group(0) @binding(0) var<storage, read> input: array<f32>;
                @group(0) @binding(1) var<storage, read_write> output: array<f32>;
                @group(0) @binding(2) var<uniform> params: SumDimParams;

                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let out_idx = global_id.x;
                    if out_idx >= params.info[0] { return; }

                    let c3 = out_idx % params.out_shape[3];
                    let c2 = (out_idx / params.out_shape[3]) % params.out_shape[2];
                    let c1 = (out_idx / (params.out_shape[2] * params.out_shape[3])) % params.out_shape[1];
                    let c0 = out_idx / (params.out_shape[1] * params.out_shape[2] * params.out_shape[3]);

                    let reduced_axis = params.info[2];
                    let loop_limit = params.in_shape[reduced_axis];
                    var sum = 0.0;
                    for (var k = 0u; k < loop_limit; k++) {
                        var in_coords = vec4<u32>(c0, c1, c2, c3);
                        if (reduced_axis == 0u) { in_coords[0] = k; }
                        else if (reduced_axis == 1u) { in_coords[1] = k; }
                        else if (reduced_axis == 2u) { in_coords[2] = k; }
                        else if (reduced_axis == 3u) { in_coords[3] = k; }
                        
                        let in_idx = (in_coords[0] * params.in_strides[0]) + 
                                     (in_coords[1] * params.in_strides[1]) + 
                                     (in_coords[2] * params.in_strides[2]) + 
                                     (in_coords[3] * params.in_strides[3]);
                        sum += input[in_idx];
                    }
                    output[out_idx] = sum;
                }
            "#;

            let pipeline = crate::codegen::cache::PipelineCache::global().get_or_compile(
                "sum_dim_pipeline",
                sum_dim_wgsl,
                &self.inner.device,
            );

            let in_dims = input_shape.dims();
            let mut in_shape_4d = [1; 4];
            let mut out_shape_4d = [1; 4];
            for i in 0..in_dims.len() {
                let axis_idx = 4 - in_dims.len() + i;
                in_shape_4d[axis_idx] = in_dims[i] as u32;
                out_shape_4d[axis_idx] = if i == axis { 1 } else { in_dims[i] as u32 };
            }

            let mut original_strides = vec![1; in_dims.len()];
            for i in (0..in_dims.len() - 1).rev() {
                original_strides[i] = original_strides[i + 1] * in_dims[i + 1];
            }
            let mut stride_4d = [0; 4];
            for (i, &stride) in original_strides.iter().enumerate() {
                let axis_idx = 4 - in_dims.len() + i;
                stride_4d[axis_idx] = stride as u32;
            }

            #[repr(C)]
            #[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
            struct SumDimParams {
                info: [u32; 4],
                out_shape: [u32; 4],
                in_shape: [u32; 4],
                in_strides: [u32; 4],
            }
            let params = SumDimParams {
                info: [
                    num_elements_out as u32,
                    input_shape.num_elements() as u32,
                    (4 - in_dims.len() + axis) as u32,
                    0,
                ],
                out_shape: out_shape_4d,
                in_shape: in_shape_4d,
                in_strides: stride_4d,
            };

            let params_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("SumDim Params Uniform"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SumDim Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("SumDim Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SumDim Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let w_x = (num_elements_out as u32).div_ceil(256);
                compute_pass.dispatch_workgroups(w_x, 1, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(output_buf)
        }

        /// Execute softmax normalization along the last axis on the GPU.
        pub fn execute_softmax_buffers(
            &self,
            input_buf: &wgpu::Buffer,
            input_shape: &Shape,
        ) -> Result<wgpu::Buffer, Error> {
            let num_elements = input_shape.num_elements();
            let size_bytes = num_elements * std::mem::size_of::<f32>();
            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Softmax Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let softmax_wgsl = r#"
                struct SoftmaxParams {
                    rows: u32,
                    cols: u32,
                }
                @group(0) @binding(0) var<storage, read> input: array<f32>;
                @group(0) @binding(1) var<storage, read_write> output: array<f32>;
                @group(0) @binding(2) var<uniform> params: SoftmaxParams;

                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let r = id.x;
                    if r < params.rows {
                        let start_idx = r * params.cols;
                        let end_idx = start_idx + params.cols;
                        
                        var max_val = input[start_idx];
                        for (var i = start_idx + 1u; i < end_idx; i++) {
                            max_val = max(max_val, input[i]);
                        }
                        
                        var sum = 0.0;
                        for (var i = start_idx; i < end_idx; i++) {
                            sum += exp(input[i] - max_val);
                        }
                        
                        let inv_sum = select(1.0 / sum, 0.0, sum == 0.0);
                        for (var i = start_idx; i < end_idx; i++) {
                            output[i] = exp(input[i] - max_val) * inv_sum;
                        }
                    }
                }
            "#;

            let pipeline = crate::codegen::cache::PipelineCache::global().get_or_compile(
                "softmax_pipeline",
                softmax_wgsl,
                &self.inner.device,
            );

            let dims = input_shape.dims();
            let cols = dims[dims.len() - 1] as u32;
            let rows = (num_elements / cols as usize) as u32;

            #[repr(C)]
            #[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
            struct SoftmaxParams {
                rows: u32,
                cols: u32,
            }
            let params = SoftmaxParams { rows, cols };
            let params_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Softmax Params Uniform"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Softmax Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Softmax Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Softmax Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let w_x = rows.div_ceil(256);
                compute_pass.dispatch_workgroups(w_x, 1, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(output_buf)
        }

        /// Execute backward pass softmax gradient computation on the GPU.
        pub fn execute_softmax_grad_buffers(
            &self,
            softmax_x_buf: &wgpu::Buffer,
            d_out_buf: &wgpu::Buffer,
            shape: &Shape,
        ) -> Result<wgpu::Buffer, Error> {
            let num_elements = shape.num_elements();
            let size_bytes = num_elements * std::mem::size_of::<f32>();
            let d_in_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SoftmaxGrad Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let softmax_grad_wgsl = r#"
                struct SoftmaxGradParams {
                    rows: u32,
                    cols: u32,
                }
                @group(0) @binding(0) var<storage, read> softmax_x: array<f32>;
                @group(0) @binding(1) var<storage, read> d_out: array<f32>;
                @group(0) @binding(2) var<storage, read_write> d_in: array<f32>;
                @group(0) @binding(3) var<uniform> params: SoftmaxGradParams;

                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let r = id.x;
                    if r < params.rows {
                        let start_idx = r * params.cols;
                        let end_idx = start_idx + params.cols;
                        
                        var sum = 0.0;
                        for (var i = start_idx; i < end_idx; i++) {
                            sum += d_out[i] * softmax_x[i];
                        }
                        
                        for (var i = start_idx; i < end_idx; i++) {
                            d_in[i] = softmax_x[i] * (d_out[i] - sum);
                        }
                    }
                }
            "#;

            let pipeline = crate::codegen::cache::PipelineCache::global().get_or_compile(
                "softmax_grad_pipeline",
                softmax_grad_wgsl,
                &self.inner.device,
            );

            let dims = shape.dims();
            let cols = dims[dims.len() - 1] as u32;
            let rows = (num_elements / cols as usize) as u32;

            #[repr(C)]
            #[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
            struct SoftmaxGradParams {
                rows: u32,
                cols: u32,
            }
            let params = SoftmaxGradParams { rows, cols };
            let params_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("SoftmaxGrad Params Uniform"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SoftmaxGrad Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: softmax_x_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: d_out_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: d_in_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("SoftmaxGrad Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SoftmaxGrad Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let w_x = rows.div_ceil(256);
                compute_pass.dispatch_workgroups(w_x, 1, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(d_in_buf)
        }

        pub fn execute_ast_buffers(
            &self,
            expr: &crate::codegen::ast::Expr,
            input_buffers: &[&wgpu::Buffer],
            input_shapes: &[Shape],
            output_shape: &Shape,
        ) -> Result<wgpu::Buffer, Error> {
            let num_elements = output_shape.num_elements();
            let size_bytes = num_elements * std::mem::size_of::<f32>();

            // 1. Create output buffer
            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("AST Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // 2. Check if we need a broadcast shader or elementwise shader
            let is_broadcasting = input_shapes.iter().any(|s| s != output_shape);

            let builder = crate::codegen::WgslKernelBuilder {
                num_inputs: input_buffers.len(),
                output_shape: output_shape.clone(),
                input_shapes: input_shapes.to_vec(),
                expr: expr.clone(),
            };

            let wgsl = if is_broadcasting {
                builder.build_broadcast()
            } else {
                builder.build_elementwise()
            };

            // 3. Get or compile pipeline
            let cache_key =
                crate::codegen::ast::hash_ast(expr, input_shapes, crate::tensor::Dtype::F32);
            let pipeline = crate::codegen::cache::PipelineCache::global().get_or_compile(
                &cache_key,
                &wgsl,
                &self.inner.device,
            );

            // 4. Create parameter buffer (uniform)
            let params_buf = if is_broadcasting {
                let mut out_shape_4d = [1; 4];
                let out_dims = output_shape.dims();
                for i in 0..out_dims.len() {
                    out_shape_4d[4 - out_dims.len() + i] = out_dims[i] as u32;
                }

                let mut stride_0_4d = [0; 4];
                let mut stride_1_4d = [0; 4];

                for (idx, in_shape) in input_shapes.iter().enumerate() {
                    let in_dims = in_shape.dims();
                    let mut original_strides = vec![1; in_dims.len()];
                    for i in (0..in_dims.len() - 1).rev() {
                        original_strides[i] = original_strides[i + 1] * in_dims[i + 1];
                    }

                    let mut s_4d = [0; 4];
                    for i in 0..in_dims.len() {
                        let out_axis = out_dims.len() - in_dims.len() + i;
                        if in_dims[i] != 1 {
                            s_4d[4 - out_dims.len() + out_axis] = original_strides[i] as u32;
                        }
                    }

                    if idx == 0 {
                        stride_0_4d = s_4d;
                    } else if idx == 1 {
                        stride_1_4d = s_4d;
                    }
                }

                let params = crate::codegen::BroadcastParams {
                    info: [num_elements as u32, out_dims.len() as u32, 0, 0],
                    output_shape: out_shape_4d,
                    stride_0: stride_0_4d,
                    stride_1: stride_1_4d,
                };

                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Broadcast Params Uniform Buffer"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
            } else {
                let num_elements_u32 = num_elements as u32;
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Elementwise Params Uniform Buffer"),
                        contents: bytemuck::bytes_of(&num_elements_u32),
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
            };

            // 5. Create bind group
            let mut entries = Vec::new();
            for (i, buf) in input_buffers.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: buf.as_entire_binding(),
                });
            }
            entries.push(wgpu::BindGroupEntry {
                binding: input_buffers.len() as u32,
                resource: output_buf.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: (input_buffers.len() + 1) as u32,
                resource: params_buf.as_entire_binding(),
            });

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("AST Execution Bind Group"),
                    layout: &bind_group_layout,
                    entries: &entries,
                });

            // 6. Encode and submit command
            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("AST Execution Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("AST Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let total_workgroups = (num_elements as u32).div_ceil(256);
                let workgroups_x = total_workgroups.min(65535);
                let workgroups_y = total_workgroups.div_ceil(65535);
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));

            Ok(output_buf)
        }

        pub fn execute_rmsnorm_buffers(
            &self,
            x_buf: &wgpu::Buffer,
            w_buf: &wgpu::Buffer,
            shape: Shape,
            epsilon: f32,
        ) -> Result<wgpu::Buffer, Error> {
            let num_elements = shape.num_elements();
            let size_bytes = num_elements * std::mem::size_of::<f32>();
            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RMSNorm Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let rmsnorm_wgsl = r#"
                struct RMSNormParams {
                    rows: u32,
                    cols: u32,
                    eps: f32,
                }
                @group(0) @binding(0) var<storage, read> x: array<f32>;
                @group(0) @binding(1) var<storage, read> w: array<f32>;
                @group(0) @binding(2) var<storage, read_write> output: array<f32>;
                @group(0) @binding(3) var<uniform> params: RMSNormParams;

                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let row = id.x;
                    if (row < params.rows) {
                        var sum_sq = 0.0;
                        let start = row * params.cols;
                        for (var i = 0u; i < params.cols; i++) {
                            let v = x[start + i];
                            sum_sq = sum_sq + v * v;
                        }
                        let rms = sqrt(sum_sq / f32(params.cols) + params.eps);
                        for (var i = 0u; i < params.cols; i++) {
                            output[start + i] = x[start + i] / rms * w[i];
                        }
                    }
                }
            "#;

            let pipeline = crate::codegen::cache::PipelineCache::global().get_or_compile(
                "rmsnorm_pipeline",
                rmsnorm_wgsl,
                &self.inner.device,
            );

            let dims = shape.dims();
            let cols = dims[dims.len() - 1] as u32;
            let rows = (num_elements / cols as usize) as u32;

            #[repr(C)]
            #[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
            struct RMSNormParams {
                rows: u32,
                cols: u32,
                eps: f32,
            }
            let params = RMSNormParams {
                rows,
                cols,
                eps: epsilon,
            };
            let params_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("RMSNorm Params Uniform"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RMSNorm Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: x_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: w_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: output_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("RMSNorm Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("RMSNorm Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let w_x = rows.div_ceil(256);
                compute_pass.dispatch_workgroups(w_x, 1, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(output_buf)
        }

        pub fn execute_concat_buffers(
            &self,
            input_bufs: &[&wgpu::Buffer],
            input_shapes: &[Shape],
            axis: usize,
            output_shape: &Shape,
        ) -> Result<wgpu::Buffer, Error> {
            let output_size_bytes = output_shape.num_elements() * std::mem::size_of::<f32>();
            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Concat Output Buffer"),
                size: output_size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let out_dims = output_shape.dims();

            let concat_copy_wgsl = r#"
                struct ConcatCopyParams {
                    axis_stride_in: u32,
                    axis_stride_out: u32,
                    inner_size: u32,
                    axis_offset: u32,
                    num_elements: u32,
                }
                @group(0) @binding(0) var<storage, read> input: array<f32>;
                @group(0) @binding(1) var<storage, read_write> output: array<f32>;
                @group(0) @binding(2) var<uniform> params: ConcatCopyParams;

                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let i = id.x;
                    if (i < params.num_elements) {
                        let outer = i / params.axis_stride_in;
                        let p = i % params.axis_stride_in;
                        let inner_idx = p % params.inner_size;
                        let axis_idx = p / params.inner_size;
                        let out_idx = outer * params.axis_stride_out + (params.axis_offset + axis_idx) * params.inner_size + inner_idx;
                        output[out_idx] = input[i];
                    }
                }
            "#;

            let pipeline = crate::codegen::cache::PipelineCache::global().get_or_compile(
                "concat_copy_pipeline",
                concat_copy_wgsl,
                &self.inner.device,
            );

            let mut offset_along_axis = 0u32;
            for (buf, shape) in input_bufs.iter().zip(input_shapes) {
                let dims = shape.dims();
                let num_elements = shape.num_elements() as u32;

                let axis_stride_in: u32 = dims[axis..].iter().map(|x| *x as u32).product();
                let axis_stride_out: u32 = out_dims[axis..].iter().map(|x| *x as u32).product();
                let inner_size: u32 = dims[(axis + 1)..].iter().map(|x| *x as u32).product();

                #[repr(C)]
                #[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
                struct ConcatCopyParams {
                    axis_stride_in: u32,
                    axis_stride_out: u32,
                    inner_size: u32,
                    axis_offset: u32,
                    num_elements: u32,
                }
                let params = ConcatCopyParams {
                    axis_stride_in,
                    axis_stride_out,
                    inner_size,
                    axis_offset: offset_along_axis,
                    num_elements,
                };
                let params_buf =
                    self.inner
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("ConcatCopy Params Uniform"),
                            contents: bytemuck::bytes_of(&params),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                let bind_group_layout = pipeline.get_bind_group_layout(0);
                let bind_group = self
                    .inner
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("ConcatCopy Bind Group"),
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: output_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buf.as_entire_binding(),
                            },
                        ],
                    });

                let mut encoder =
                    self.inner
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("ConcatCopy Encoder"),
                        });
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("ConcatCopy Compute Pass"),
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(&pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    let w_x = num_elements.div_ceil(256);
                    compute_pass.dispatch_workgroups(w_x, 1, 1);
                }
                self.inner.queue.submit(Some(encoder.finish()));

                offset_along_axis += dims[axis] as u32;
            }

            Ok(output_buf)
        }

        pub fn execute_layernorm_buffers(
            &self,
            x_buf: &wgpu::Buffer,
            w_buf: &wgpu::Buffer,
            b_buf: &wgpu::Buffer,
            shape: &Shape,
            epsilon: f32,
        ) -> Result<wgpu::Buffer, Error> {
            let num_elements = shape.num_elements();
            let size_bytes = num_elements * std::mem::size_of::<f32>();
            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("LayerNorm Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let layernorm_wgsl = r#"
                struct LayerNormParams {
                    rows: u32,
                    cols: u32,
                    eps: f32,
                }
                @group(0) @binding(0) var<storage, read> x: array<f32>;
                @group(0) @binding(1) var<storage, read> w: array<f32>;
                @group(0) @binding(2) var<storage, read> b: array<f32>;
                @group(0) @binding(3) var<storage, read_write> output: array<f32>;
                @group(0) @binding(4) var<uniform> params: LayerNormParams;

                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let row = id.x;
                    if (row < params.rows) {
                        var sum = 0.0;
                        var sum_sq = 0.0;
                        let start = row * params.cols;
                        for (var i = 0u; i < params.cols; i++) {
                            let v = x[start + i];
                            sum = sum + v;
                            sum_sq = sum_sq + v * v;
                        }
                        let mean = sum / f32(params.cols);
                        let variance = sum_sq / f32(params.cols) - mean * mean;
                        let inv_std = 1.0 / sqrt(variance + params.eps);
                        for (var i = 0u; i < params.cols; i++) {
                            output[start + i] = (x[start + i] - mean) * inv_std * w[i] + b[i];
                        }
                    }
                }
            "#;

            let pipeline = crate::codegen::cache::PipelineCache::global().get_or_compile(
                "layernorm_pipeline",
                layernorm_wgsl,
                &self.inner.device,
            );

            let dims = shape.dims();
            let cols = dims[dims.len() - 1] as u32;
            let rows = (num_elements / cols as usize) as u32;

            #[repr(C)]
            #[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
            struct LayerNormParams {
                rows: u32,
                cols: u32,
                eps: f32,
            }
            let params = LayerNormParams {
                rows,
                cols,
                eps: epsilon,
            };
            let params_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("LayerNorm Params Uniform"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("LayerNorm Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: x_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: w_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: b_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: output_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("LayerNorm Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("LayerNorm Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let w_x = rows.div_ceil(256);
                compute_pass.dispatch_workgroups(w_x, 1, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(output_buf)
        }

        pub fn execute_conv2d_buffers(
            &self,
            x_buf: &wgpu::Buffer,
            w_buf: &wgpu::Buffer,
            bias_buf: Option<&wgpu::Buffer>,
            x_shape: &Shape,
            w_shape: &Shape,
            stride: usize,
            padding: usize,
        ) -> Result<wgpu::Buffer, Error> {
            let x_dims = x_shape.dims();
            let w_dims = w_shape.dims();
            let n = x_dims[0] as u32;
            let c = x_dims[1] as u32;
            let h_in = x_dims[2] as u32;
            let w_in = x_dims[3] as u32;
            let o = w_dims[0] as u32;
            let kh = w_dims[2] as u32;
            let kw = w_dims[3] as u32;
            let h_out = ((h_in + 2 * padding as u32 - kh) / stride as u32) + 1;
            let w_out = ((w_in + 2 * padding as u32 - kw) / stride as u32) + 1;

            let output_num_elements = (n * o * h_out * w_out) as usize;
            let size_bytes = output_num_elements * std::mem::size_of::<f32>();
            let output_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Conv2d Output Buffer"),
                size: size_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let conv2d_wgsl = r#"
                struct Conv2dParams {
                    n: u32, o: u32, c: u32, h_in: u32, w_in: u32,
                    kh: u32, kw: u32, h_out: u32, w_out: u32,
                    stride: u32, padding: u32,
                }
                @group(0) @binding(0) var<storage, read> x: array<f32>;
                @group(0) @binding(1) var<storage, read> w: array<f32>;
                @group(0) @binding(2) var<storage, read> bias: array<f32>;
                @group(0) @binding(3) var<storage, read_write> output: array<f32>;
                @group(0) @binding(4) var<uniform> params: Conv2dParams;

                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let idx = id.x;
                    let total = params.n * params.o * params.h_out * params.w_out;
                    if (idx < total) {
                        let ow = idx % params.w_out;
                        let oh = (idx / params.w_out) % params.h_out;
                        let oo = (idx / (params.w_out * params.h_out)) % params.o;
                        let on = idx / (params.o * params.h_out * params.w_out);

                        var sum = 0.0;
                        for (var ci = 0u; ci < params.c; ci++) {
                            for (var ki = 0u; ki < params.kh; ki++) {
                                for (var kj = 0u; kj < params.kw; kj++) {
                                    let h_in_pos = oh * params.stride + ki;
                                    let w_in_pos = ow * params.stride + kj;
                                    if (h_in_pos >= params.padding && h_in_pos < params.h_in + params.padding &&
                                        w_in_pos >= params.padding && w_in_pos < params.w_in + params.padding) {
                                        let x_idx = on * params.c * params.h_in * params.w_in
                                            + ci * params.h_in * params.w_in
                                            + (h_in_pos - params.padding) * params.w_in
                                            + (w_in_pos - params.padding);
                                        let w_idx = oo * params.c * params.kh * params.kw
                                            + ci * params.kh * params.kw
                                            + ki * params.kw
                                            + kj;
                                        sum = sum + x[x_idx] * w[w_idx];
                                    }
                                }
                            }
                        }
                        let b = bias[oo];
                        output[idx] = sum + b;
                    }
                }
            "#;

            let pipeline = crate::codegen::cache::PipelineCache::global().get_or_compile(
                "conv2d_pipeline",
                conv2d_wgsl,
                &self.inner.device,
            );

            #[repr(C)]
            #[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
            struct Conv2dParams {
                n: u32,
                o: u32,
                c: u32,
                h_in: u32,
                w_in: u32,
                kh: u32,
                kw: u32,
                h_out: u32,
                w_out: u32,
                stride: u32,
                padding: u32,
            }
            let params = Conv2dParams {
                n,
                o,
                c,
                h_in,
                w_in,
                kh,
                kw,
                h_out,
                w_out,
                stride: stride as u32,
                padding: padding as u32,
            };
            let params_buf =
                self.inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Conv2d Params Uniform"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bias_buf = bias_buf.unwrap_or(&self.inner.dummy_bias);

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = self
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Conv2d Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: x_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: w_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: bias_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: output_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Conv2d Encoder"),
                    });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Conv2d Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let total = n * o * h_out * w_out;
                let w_x = total.div_ceil(256);
                compute_pass.dispatch_workgroups(w_x, 1, 1);
            }
            self.inner.queue.submit(Some(encoder.finish()));
            Ok(output_buf)
        }
    }

    impl Backend for WgpuBackend {
        fn execute(&self, op: &Op, inputs: &[&Tensor]) -> Result<Tensor, Error> {
            match op {
                Op::Input(tensor) => Ok(tensor.clone()),
                Op::MatMul => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "MatMul requires exactly 2 inputs".to_string(),
                        ));
                    }
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    let lhs_shape = lhs.shape().dims();
                    let rhs_shape = rhs.shape().dims();
                    if lhs_shape.len() != 2 || rhs_shape.len() != 2 || lhs_shape[1] != rhs_shape[0]
                    {
                        return Err(Error::ShapeMismatch(format!(
                            "MatMul shape mismatch: {:?} and {:?}",
                            lhs.shape(),
                            rhs.shape()
                        )));
                    }

                    let m = lhs_shape[0] as u32;
                    let k = lhs_shape[1] as u32;
                    let n = rhs_shape[1] as u32;

                    let num_elements = (m * n) as usize;
                    let size_bytes = num_elements * std::mem::size_of::<f32>();

                    let a_buf =
                        self.create_buffer_with_data(lhs.data(), wgpu::BufferUsages::STORAGE);
                    let b_buf =
                        self.create_buffer_with_data(rhs.data(), wgpu::BufferUsages::STORAGE);
                    let c_buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("MatMul Output Buffer"),
                        size: size_bytes as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });

                    let dims = MatmulDims {
                        m,
                        n,
                        k,
                        padding: 0,
                    };
                    let dims_buf =
                        self.inner
                            .device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("MatMul Dims Uniform Buffer"),
                                contents: bytemuck::bytes_of(&dims),
                                usage: wgpu::BufferUsages::UNIFORM,
                            });

                    let bind_group_layout = self.inner.matmul_pipeline.get_bind_group_layout(0);
                    let bind_group =
                        self.inner
                            .device
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("MatMul Bind Group"),
                                layout: &bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: a_buf.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: b_buf.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: c_buf.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: dims_buf.as_entire_binding(),
                                    },
                                ],
                            });

                    let mut encoder =
                        self.inner
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("MatMul Encoder"),
                            });
                    {
                        let mut compute_pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("MatMul Compute Pass"),
                                timestamp_writes: None,
                            });
                        compute_pass.set_pipeline(&self.inner.matmul_pipeline);
                        compute_pass.set_bind_group(0, &bind_group, &[]);

                        let workgroups_x = n.div_ceil(64);
                        let workgroups_y = m.div_ceil(64);
                        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
                    }
                    self.inner.queue.submit(Some(encoder.finish()));

                    let out_data = self.read_buffer(&c_buf, size_bytes)?;
                    let out_shape = Shape::new(vec![lhs_shape[0], rhs_shape[1]]);
                    Ok(Tensor::new(out_data, out_shape))
                }
                Op::Transpose => {
                    let dims = inputs[0].shape().dims();
                    let input_buf =
                        self.create_buffer_with_data(inputs[0].data(), wgpu::BufferUsages::STORAGE);
                    let out_buf =
                        self.execute_transpose_buffers(&input_buf, dims[0] as u32, dims[1] as u32)?;
                    let out_shape = Shape::new(vec![dims[1], dims[0]]);
                    let size_bytes = out_shape.num_elements() * std::mem::size_of::<f32>();
                    let out_data = self.read_buffer(&out_buf, size_bytes)?;
                    Ok(Tensor::new(out_data, out_shape))
                }
                Op::SumAll => {
                    let input_buf =
                        self.create_buffer_with_data(inputs[0].data(), wgpu::BufferUsages::STORAGE);
                    let out_buf = self.execute_sum_all_buffers(&input_buf)?;
                    let out_shape = Shape::new(vec![1]);
                    let out_data = self.read_buffer(&out_buf, 4)?;
                    Ok(Tensor::new(out_data, out_shape))
                }
                Op::SumDim { axis } => {
                    let input_buf =
                        self.create_buffer_with_data(inputs[0].data(), wgpu::BufferUsages::STORAGE);
                    let out_buf =
                        self.execute_sum_dim_buffers(&input_buf, inputs[0].shape(), *axis)?;
                    let mut out_dims = inputs[0].shape().dims().to_vec();
                    out_dims[*axis] = 1;
                    let out_shape = Shape::new(out_dims);
                    let size_bytes = out_shape.num_elements() * std::mem::size_of::<f32>();
                    let out_data = self.read_buffer(&out_buf, size_bytes)?;
                    Ok(Tensor::new(out_data, out_shape))
                }
                Op::Reshape { shape } => Ok(Tensor::new(inputs[0].data().to_vec(), shape.clone())),
                Op::Softmax => {
                    let input_buf =
                        self.create_buffer_with_data(inputs[0].data(), wgpu::BufferUsages::STORAGE);
                    let out_buf = self.execute_softmax_buffers(&input_buf, inputs[0].shape())?;
                    let out_shape = inputs[0].shape().clone();
                    let size_bytes = out_shape.num_elements() * std::mem::size_of::<f32>();
                    let out_data = self.read_buffer(&out_buf, size_bytes)?;
                    Ok(Tensor::new(out_data, out_shape))
                }
                Op::SoftmaxGrad => {
                    let x_buf =
                        self.create_buffer_with_data(inputs[0].data(), wgpu::BufferUsages::STORAGE);
                    let grad_buf =
                        self.create_buffer_with_data(inputs[1].data(), wgpu::BufferUsages::STORAGE);
                    let out_buf =
                        self.execute_softmax_grad_buffers(&x_buf, &grad_buf, inputs[0].shape())?;
                    let out_shape = inputs[0].shape().clone();
                    let size_bytes = out_shape.num_elements() * std::mem::size_of::<f32>();
                    let out_data = self.read_buffer(&out_buf, size_bytes)?;
                    Ok(Tensor::new(out_data, out_shape))
                }
                Op::RmsNorm { epsilon } => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "RMSNorm requires exactly 2 inputs: x, weight".to_string(),
                        ));
                    }
                    let x_buf = self.create_buffer_with_data(
                        inputs[0].data(),
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    );
                    let w_buf =
                        self.create_buffer_with_data(inputs[1].data(), wgpu::BufferUsages::STORAGE);
                    let out_buf = self.execute_rmsnorm_buffers(
                        &x_buf,
                        &w_buf,
                        inputs[0].shape().clone(),
                        *epsilon,
                    )?;
                    let size_bytes = inputs[0].shape().num_elements() * std::mem::size_of::<f32>();
                    let out_data = self.read_buffer(&out_buf, size_bytes)?;
                    Ok(Tensor::new(out_data, inputs[0].shape().clone()))
                }
                _ => {
                    let expr = match op {
                        Op::Relu => crate::codegen::ast::Expr::Relu(Box::new(
                            crate::codegen::ast::Expr::Input(0),
                        )),
                        Op::Add => crate::codegen::ast::Expr::Add(
                            Box::new(crate::codegen::ast::Expr::Input(0)),
                            Box::new(crate::codegen::ast::Expr::Input(1)),
                        ),
                        Op::Sub => crate::codegen::ast::Expr::Sub(
                            Box::new(crate::codegen::ast::Expr::Input(0)),
                            Box::new(crate::codegen::ast::Expr::Input(1)),
                        ),
                        Op::Mul => crate::codegen::ast::Expr::Mul(
                            Box::new(crate::codegen::ast::Expr::Input(0)),
                            Box::new(crate::codegen::ast::Expr::Input(1)),
                        ),
                        Op::Div => crate::codegen::ast::Expr::Div(
                            Box::new(crate::codegen::ast::Expr::Input(0)),
                            Box::new(crate::codegen::ast::Expr::Input(1)),
                        ),
                        Op::Tanh => crate::codegen::ast::Expr::Tanh(Box::new(
                            crate::codegen::ast::Expr::Input(0),
                        )),
                        Op::Sigmoid => crate::codegen::ast::Expr::Sigmoid(Box::new(
                            crate::codegen::ast::Expr::Input(0),
                        )),
                        Op::Exp => crate::codegen::ast::Expr::Exp(Box::new(
                            crate::codegen::ast::Expr::Input(0),
                        )),
                        Op::Sqrt => crate::codegen::ast::Expr::Sqrt(Box::new(
                            crate::codegen::ast::Expr::Input(0),
                        )),
                        Op::Neg => crate::codegen::ast::Expr::Neg(Box::new(
                            crate::codegen::ast::Expr::Input(0),
                        )),
                        Op::Step => crate::codegen::ast::Expr::Step(Box::new(
                            crate::codegen::ast::Expr::Input(0),
                        )),
                        Op::BroadcastAdd { .. } => crate::codegen::ast::Expr::Add(
                            Box::new(crate::codegen::ast::Expr::Input(0)),
                            Box::new(crate::codegen::ast::Expr::Input(1)),
                        ),
                        Op::BroadcastMul { .. } => crate::codegen::ast::Expr::Mul(
                            Box::new(crate::codegen::ast::Expr::Input(0)),
                            Box::new(crate::codegen::ast::Expr::Input(1)),
                        ),
                        Op::BroadcastSub { .. } => crate::codegen::ast::Expr::Sub(
                            Box::new(crate::codegen::ast::Expr::Input(0)),
                            Box::new(crate::codegen::ast::Expr::Input(1)),
                        ),
                        Op::BroadcastDiv { .. } => crate::codegen::ast::Expr::Div(
                            Box::new(crate::codegen::ast::Expr::Input(0)),
                            Box::new(crate::codegen::ast::Expr::Input(1)),
                        ),
                        _ => {
                            return Err(Error::ExecutionError(format!(
                                "Unsupported WgpuBackend op: {:?}",
                                op
                            )))
                        }
                    };

                    let mut input_buffers = Vec::new();
                    let mut input_shapes = Vec::new();
                    for input in inputs {
                        let buf =
                            self.create_buffer_with_data(input.data(), wgpu::BufferUsages::STORAGE);
                        input_buffers.push(buf);
                        input_shapes.push(input.shape().clone());
                    }

                    let input_buffer_refs: Vec<&wgpu::Buffer> = input_buffers.iter().collect();

                    let output_shape = match op {
                        Op::BroadcastAdd { .. }
                        | Op::BroadcastMul { .. }
                        | Op::BroadcastSub { .. }
                        | Op::BroadcastDiv { .. } => {
                            crate::graph::broadcast_shapes(&input_shapes[0], &input_shapes[1])
                                .ok_or_else(|| {
                                    Error::ExecutionError(
                                        "Incompatible shapes for broadcast".to_string(),
                                    )
                                })?
                        }
                        _ => input_shapes[0].clone(),
                    };

                    let out_buf = self.execute_ast_buffers(
                        &expr,
                        &input_buffer_refs,
                        &input_shapes,
                        &output_shape,
                    )?;
                    let size_bytes = output_shape.num_elements() * std::mem::size_of::<f32>();
                    let out_data = self.read_buffer(&out_buf, size_bytes)?;

                    Ok(Tensor::new(out_data, output_shape))
                }
            }
        }

        fn execute_matmul_relu(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
            self.execute_matmul_relu_gpu(a, b)
        }

        fn execute_matmul_add(
            &self,
            a: &Tensor,
            b: &Tensor,
            bias: &Tensor,
        ) -> Result<Tensor, Error> {
            self.execute_matmul_add_gpu(a, b, bias)
        }

        fn execute_matmul_add_relu(
            &self,
            a: &Tensor,
            b: &Tensor,
            bias: &Tensor,
        ) -> Result<Tensor, Error> {
            self.execute_matmul_add_relu_gpu(a, b, bias)
        }
    }
}
pub use wgpu_backend_mod::WgpuBackend;
