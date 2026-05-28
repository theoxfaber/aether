pub mod ast;
pub mod cache;

use crate::tensor::Shape;
use ast::Expr;

pub struct WgslKernelBuilder {
    pub num_inputs: usize,
    pub output_shape: Shape,
    pub input_shapes: Vec<Shape>,
    pub expr: Expr,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
pub struct BroadcastParams {
    pub info: [u32; 4], // info[0] = num_elements, info[1] = ndim, info[2] = padding, info[3] = padding
    pub output_shape: [u32; 4],
    pub stride_0: [u32; 4],
    pub stride_1: [u32; 4],
}

impl WgslKernelBuilder {
    pub fn build_elementwise(&self) -> String {
        let mut s = String::new();
        for i in 0..self.num_inputs {
            s.push_str(&format!(
                "@group(0) @binding({}) var<storage, read> in{}: array<f32>;\n",
                i, i
            ));
        }
        s.push_str(&format!(
            "@group(0) @binding({}) var<storage, read_write> out: array<f32>;\n",
            self.num_inputs
        ));
        s.push_str(&format!(
            "@group(0) @binding({}) var<uniform> num_elements: u32;\n\n",
            self.num_inputs + 1
        ));

        let input_indexer = |idx: usize| format!("in{}[global_idx]", idx);
        let expr_wgsl = self.expr.to_wgsl(&input_indexer);

        s.push_str(
            "@compute @workgroup_size(256)\n\
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n\
            \tlet global_idx = global_id.x;\n\
            \tif (global_idx >= num_elements) { return; }\n\
            \tout[global_idx] = ",
        );
        s.push_str(&expr_wgsl);
        s.push_str(";\n}\n");
        s
    }

    pub fn build_broadcast(&self) -> String {
        let mut s = String::new();
        for i in 0..self.num_inputs {
            s.push_str(&format!(
                "@group(0) @binding({}) var<storage, read> in{}: array<f32>;\n",
                i, i
            ));
        }
        s.push_str(&format!(
            "@group(0) @binding({}) var<storage, read_write> out: array<f32>;\n",
            self.num_inputs
        ));
        s.push_str(&format!(
            "struct BroadcastParams {{\n\
            \tinfo: vec4<u32>,\n\
            \toutput_shape: vec4<u32>,\n\
            \tstride_0: vec4<u32>,\n\
            \tstride_1: vec4<u32>,\n\
            }};\n\n\
            @group(0) @binding({}) var<uniform> params: BroadcastParams;\n\n",
            self.num_inputs + 1
        ));

        let input_indexer = |idx: usize| format!("in{}[input_idx_{}]", idx, idx);
        let expr_wgsl = self.expr.to_wgsl(&input_indexer);

        s.push_str(
            "@compute @workgroup_size(256)\n\
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n\
            \tlet global_idx = global_id.x;\n\
            \tif (global_idx >= params.info[0]) { return; }\n\n\
            \tlet c3 = global_idx % params.output_shape[3];\n\
            \tlet c2 = (global_idx / params.output_shape[3]) % params.output_shape[2];\n\
            \tlet c1 = (global_idx / (params.output_shape[2] * params.output_shape[3])) % params.output_shape[1];\n\
            \tlet c0 = global_idx / (params.output_shape[1] * params.output_shape[2] * params.output_shape[3]);\n\n"
        );

        for i in 0..self.num_inputs {
            s.push_str(&format!(
                "\tlet input_idx_{} = (c0 * params.stride_{}[0]) + (c1 * params.stride_{}[1]) + (c2 * params.stride_{}[2]) + (c3 * params.stride_{}[3]);\n",
                i, i, i, i, i
            ));
        }

        s.push_str("\n\tout[global_idx] = ");
        s.push_str(&expr_wgsl);
        s.push_str(";\n}\n");
        s
    }
}
