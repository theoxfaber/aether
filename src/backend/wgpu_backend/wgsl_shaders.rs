    const RELU_SHADER_SRC: &str = r#"
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
            let i = id.y * (num_workgroups.x * 256u) + id.x;
            if i < arrayLength(&input) {
                output[i] = max(input[i], 0.0);
            }
        }
    "#;

    const ADD_SHADER_SRC: &str = r#"
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> c: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
            let i = id.y * (num_workgroups.x * 256u) + id.x;
            if i < arrayLength(&a) {
                c[i] = a[i] + b[i];
            }
        }
    "#;

    const BATCHED_MATMUL_SHADER_SRC: &str = r#"
        struct Dims {
            B: u32,
            M: u32,
            N: u32,
            K: u32,
        }

        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> c: array<f32>;
        @group(0) @binding(3) var<uniform> dims: Dims;

        @compute @workgroup_size(16, 16)
        fn main(
            @builtin(local_invocation_id) local_id: vec3<u32>,
            @builtin(workgroup_id) workgroup_id: vec3<u32>
        ) {
            let batch_idx = workgroup_id.z;
            if (batch_idx >= dims.B) { return; }

            let local_row = local_id.y;
            let local_col = local_id.x;

            let wg_row = workgroup_id.y * 64u;
            let wg_col = workgroup_id.x * 64u;

            let row = wg_row + local_row * 4u;
            let col = wg_col + local_col * 4u;

            var sum00 = 0.0; var sum01 = 0.0; var sum02 = 0.0; var sum03 = 0.0;
            var sum10 = 0.0; var sum11 = 0.0; var sum12 = 0.0; var sum13 = 0.0;
            var sum20 = 0.0; var sum21 = 0.0; var sum22 = 0.0; var sum23 = 0.0;
            var sum30 = 0.0; var sum31 = 0.0; var sum32 = 0.0; var sum33 = 0.0;

            let k_limit = dims.K;
            let a_offset = batch_idx * dims.M * dims.K;
            let b_offset = batch_idx * dims.K * dims.N;
            let c_offset = batch_idx * dims.M * dims.N;

            for (var k = 0u; k < k_limit; k++) {
                var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
                if (row < dims.M) { a0 = a[a_offset + row * dims.K + k]; }
                if (row + 1u < dims.M) { a1 = a[a_offset + (row + 1u) * dims.K + k]; }
                if (row + 2u < dims.M) { a2 = a[a_offset + (row + 2u) * dims.K + k]; }
                if (row + 3u < dims.M) { a3 = a[a_offset + (row + 3u) * dims.K + k]; }

                var b0 = 0.0; var b1 = 0.0; var b2 = 0.0; var b3 = 0.0;
                if (col < dims.N) { b0 = b[b_offset + k * dims.N + col]; }
                if (col + 1u < dims.N) { b1 = b[b_offset + k * dims.N + col + 1u]; }
                if (col + 2u < dims.N) { b2 = b[b_offset + k * dims.N + col + 2u]; }
                if (col + 3u < dims.N) { b3 = b[b_offset + k * dims.N + col + 3u]; }

                sum00 += a0 * b0; sum01 += a0 * b1; sum02 += a0 * b2; sum03 += a0 * b3;
                sum10 += a1 * b0; sum11 += a1 * b1; sum12 += a1 * b2; sum13 += a1 * b3;
                sum20 += a2 * b0; sum21 += a2 * b1; sum22 += a2 * b2; sum23 += a2 * b3;
                sum30 += a3 * b0; sum31 += a3 * b1; sum32 += a3 * b2; sum33 += a3 * b3;
            }

            if (row < dims.M) {
                if (col < dims.N) { c[c_offset + row * dims.N + col] = sum00; }
                if (col + 1u < dims.N) { c[c_offset + row * dims.N + col + 1u] = sum01; }
                if (col + 2u < dims.N) { c[c_offset + row * dims.N + col + 2u] = sum02; }
                if (col + 3u < dims.N) { c[c_offset + row * dims.N + col + 3u] = sum03; }
            }
            if (row + 1u < dims.M) {
                let r = row + 1u;
                if (col < dims.N) { c[c_offset + r * dims.N + col] = sum10; }
                if (col + 1u < dims.N) { c[c_offset + r * dims.N + col + 1u] = sum11; }
                if (col + 2u < dims.N) { c[c_offset + r * dims.N + col + 2u] = sum12; }
                if (col + 3u < dims.N) { c[c_offset + r * dims.N + col + 3u] = sum13; }
            }
            if (row + 2u < dims.M) {
                let r = row + 2u;
                if (col < dims.N) { c[c_offset + r * dims.N + col] = sum20; }
                if (col + 1u < dims.N) { c[c_offset + r * dims.N + col + 1u] = sum21; }
                if (col + 2u < dims.N) { c[c_offset + r * dims.N + col + 2u] = sum22; }
                if (col + 3u < dims.N) { c[c_offset + r * dims.N + col + 3u] = sum23; }
            }
            if (row + 3u < dims.M) {
                let r = row + 3u;
                if (col < dims.N) { c[c_offset + r * dims.N + col] = sum30; }
                if (col + 1u < dims.N) { c[c_offset + r * dims.N + col + 1u] = sum31; }
                if (col + 2u < dims.N) { c[c_offset + r * dims.N + col + 2u] = sum32; }
                if (col + 3u < dims.N) { c[c_offset + r * dims.N + col + 3u] = sum33; }
            }
        }
    "#;

    const BATCHED_TRANSPOSE_SHADER_SRC: &str = r#"
        struct Dims {
            B: u32,
            M: u32,
            N: u32,
            padding: u32,
        }

        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @group(0) @binding(2) var<uniform> dims: Dims;

        @compute @workgroup_size(16, 16)
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>
        ) {
            let batch_idx = global_id.z;
            if (batch_idx >= dims.B) { return; }

            let row = global_id.y;
            let col = global_id.x;

            if (row < dims.M && col < dims.N) {
                let in_idx = batch_idx * dims.M * dims.N + row * dims.N + col;
                let out_idx = batch_idx * dims.M * dims.N + col * dims.M + row;
                output[out_idx] = input[in_idx];
            }
        }
    "#;

    const MAX_POOL_SHADER_SRC: &str = r#"
        struct PoolDims {
            N: u32,
            C: u32,
            H: u32,
            W: u32,
            out_H: u32,
            out_W: u32,
            kernel_H: u32,
            kernel_W: u32,
            stride_H: u32,
            stride_W: u32,
            padding_H: u32,
            padding_W: u32,
        }

        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @group(0) @binding(2) var<uniform> dims: PoolDims;

        @compute @workgroup_size(16, 16)
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>
        ) {
            let out_col = global_id.x; // out_w
            let out_row = global_id.y; // out_h
            let nc = global_id.z;      // batch * C + channel

            let batch = nc / dims.C;
            let channel = nc % dims.C;

            if (out_col >= dims.out_W || out_row >= dims.out_H || batch >= dims.N) {
                return;
            }

            let in_h_start = i32(out_row * dims.stride_H) - i32(dims.padding_H);
            let in_w_start = i32(out_col * dims.stride_W) - i32(dims.padding_W);

            var max_val = -3.402823466e+38; // -INFINITY

            for (var kh = 0u; kh < dims.kernel_H; kh++) {
                let ih = in_h_start + i32(kh);
                if (ih >= 0 && ih < i32(dims.H)) {
                    for (var kw = 0u; kw < dims.kernel_W; kw++) {
                        let iw = in_w_start + i32(kw);
                        if (iw >= 0 && iw < i32(dims.W)) {
                            let idx = ((batch * dims.C + channel) * dims.H + u32(ih)) * dims.W + u32(iw);
                            let val = input[idx];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                }
            }

            let out_idx = ((batch * dims.C + channel) * dims.out_H + out_row) * dims.out_W + out_col;
            output[out_idx] = max_val;
        }
    "#;

    const AVG_POOL_SHADER_SRC: &str = r#"
        struct PoolDims {
            N: u32,
            C: u32,
            H: u32,
            W: u32,
            out_H: u32,
            out_W: u32,
            kernel_H: u32,
            kernel_W: u32,
            stride_H: u32,
            stride_W: u32,
            padding_H: u32,
            padding_W: u32,
        }

        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @group(0) @binding(2) var<uniform> dims: PoolDims;

        @compute @workgroup_size(16, 16)
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>
        ) {
            let out_col = global_id.x; // out_w
            let out_row = global_id.y; // out_h
            let nc = global_id.z;      // batch * C + channel

            let batch = nc / dims.C;
            let channel = nc % dims.C;

            if (out_col >= dims.out_W || out_row >= dims.out_H || batch >= dims.N) {
                return;
            }

            let in_h_start = i32(out_row * dims.stride_H) - i32(dims.padding_H);
            let in_w_start = i32(out_col * dims.stride_W) - i32(dims.padding_W);

            var sum = 0.0;
            var count = 0.0;

            for (var kh = 0u; kh < dims.kernel_H; kh++) {
                let ih = in_h_start + i32(kh);
                if (ih >= 0 && ih < i32(dims.H)) {
                    for (var kw = 0u; kw < dims.kernel_W; kw++) {
                        let iw = in_w_start + i32(kw);
                        if (iw >= 0 && iw < i32(dims.W)) {
                            let idx = ((batch * dims.C + channel) * dims.H + u32(ih)) * dims.W + u32(iw);
                            sum += input[idx];
                            count += 1.0;
                        }
                    }
                }
            }

            let out_idx = ((batch * dims.C + channel) * dims.out_H + out_row) * dims.out_W + out_col;
            output[out_idx] = sum / max(count, 1.0);
        }
    "#;

    /// MaxPool2d backward: for each input position, gather gradients from all output
    /// windows that overlap this input pixel. This is 100% thread-safe and avoids
    /// using platform-incompatible atomic compare-exchange operations.
    const MAX_POOL_GRAD_SHADER_SRC: &str = r#"
        struct PoolGradDims {
            N: u32,
            C: u32,
            H: u32,
            W: u32,
            out_H: u32,
            out_W: u32,
            kernel_H: u32,
            kernel_W: u32,
            stride_H: u32,
            stride_W: u32,
            padding_H: u32,
            padding_W: u32,
        }

        @group(0) @binding(0) var<storage, read>       dy:    array<f32>;
        @group(0) @binding(1) var<storage, read>       x:     array<f32>;
        @group(0) @binding(2) var<storage, read_write> dx:    array<f32>;
        @group(0) @binding(3) var<uniform>             dims:  PoolGradDims;

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let col   = global_id.x;
            let row   = global_id.y;
            let nc    = global_id.z;

            let batch   = nc / dims.C;
            let channel = nc % dims.C;

            if (col >= dims.W || row >= dims.H || batch >= dims.N) {
                return;
            }

            let in_idx = ((batch * dims.C + channel) * dims.H + row) * dims.W + col;

            // Gather gradient contributions from all output windows (oh, ow) that overlap this input pixel
            let ih_pad = i32(row) + i32(dims.padding_H);
            let oh_min_num = ih_pad - i32(dims.kernel_H) + 1;
            var oh_min = 0i;
            if (oh_min_num > 0) {
                oh_min = (oh_min_num + i32(dims.stride_H) - 1) / i32(dims.stride_H);
            }
            let oh_max = min(i32(dims.out_H) - 1, ih_pad / i32(dims.stride_H));

            let iw_pad = i32(col) + i32(dims.padding_W);
            let ow_min_num = iw_pad - i32(dims.kernel_W) + 1;
            var ow_min = 0i;
            if (ow_min_num > 0) {
                ow_min = (ow_min_num + i32(dims.stride_W) - 1) / i32(dims.stride_W);
            }
            let ow_max = min(i32(dims.out_W) - 1, iw_pad / i32(dims.stride_W));

            var accum = 0.0;

            for (var oh = oh_min; oh <= oh_max; oh = oh + 1) {
                let in_h_start = oh * i32(dims.stride_H) - i32(dims.padding_H);
                for (var ow = ow_min; ow <= ow_max; ow = ow + 1) {
                    let in_w_start = ow * i32(dims.stride_W) - i32(dims.padding_W);

                    // Find the argmax inside this output window (oh, ow)
                    var max_val = -3.402823466e+38;
                    var max_ih = -1i;
                    var max_iw = -1i;

                    for (var kh = 0u; kh < dims.kernel_H; kh = kh + 1u) {
                        let ih = in_h_start + i32(kh);
                        if (ih >= 0 && ih < i32(dims.H)) {
                            for (var kw = 0u; kw < dims.kernel_W; kw = kw + 1u) {
                                let iw = in_w_start + i32(kw);
                                if (iw >= 0 && iw < i32(dims.W)) {
                                    let cur_idx = ((batch * dims.C + channel) * dims.H + u32(ih)) * dims.W + u32(iw);
                                    let val = x[cur_idx];
                                    if (val > max_val) {
                                        max_val = val;
                                        max_ih = ih;
                                        max_iw = iw;
                                    }
                                }
                            }
                        }
                    }

                    // If this input pixel is the argmax for the window, add the output gradient
                    if (max_ih == i32(row) && max_iw == i32(col)) {
                        let out_idx = ((batch * dims.C + channel) * dims.out_H + u32(oh)) * dims.out_W + u32(ow);
                        accum = accum + dy[out_idx];
                    }
                }
            }

            dx[in_idx] = accum;
        }
    "#;

    /// AvgPool2d backward: for each input position, gather uniform gradient contributions from overlapping windows.
    const AVG_POOL_GRAD_SHADER_SRC: &str = r#"
        struct PoolGradDims {
            N: u32,
            C: u32,
            H: u32,
            W: u32,
            out_H: u32,
            out_W: u32,
            kernel_H: u32,
            kernel_W: u32,
            stride_H: u32,
            stride_W: u32,
            padding_H: u32,
            padding_W: u32,
        }

        @group(0) @binding(0) var<storage, read>       dy:   array<f32>;
        @group(0) @binding(1) var<storage, read_write> dx:   array<f32>;
        @group(0) @binding(2) var<uniform>             dims: PoolGradDims;

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let col   = global_id.x;
            let row   = global_id.y;
            let nc    = global_id.z;

            let batch   = nc / dims.C;
            let channel = nc % dims.C;

            if (col >= dims.W || row >= dims.H || batch >= dims.N) {
                return;
            }

            let in_idx = ((batch * dims.C + channel) * dims.H + row) * dims.W + col;

            let ih_pad = i32(row) + i32(dims.padding_H);
            let oh_min_num = ih_pad - i32(dims.kernel_H) + 1;
            var oh_min = 0i;
            if (oh_min_num > 0) {
                oh_min = (oh_min_num + i32(dims.stride_H) - 1) / i32(dims.stride_H);
            }
            let oh_max = min(i32(dims.out_H) - 1, ih_pad / i32(dims.stride_H));

            let iw_pad = i32(col) + i32(dims.padding_W);
            let ow_min_num = iw_pad - i32(dims.kernel_W) + 1;
            var ow_min = 0i;
            if (ow_min_num > 0) {
                ow_min = (ow_min_num + i32(dims.stride_W) - 1) / i32(dims.stride_W);
            }
            let ow_max = min(i32(dims.out_W) - 1, iw_pad / i32(dims.stride_W));

            var accum = 0.0;
            let count = f32(dims.kernel_H * dims.kernel_W);

            for (var oh = oh_min; oh <= oh_max; oh = oh + 1) {
                for (var ow = ow_min; ow <= ow_max; ow = ow + 1) {
                    let out_idx = ((batch * dims.C + channel) * dims.out_H + u32(oh)) * dims.out_W + u32(ow);
                    accum = accum + dy[out_idx] / count;
                }
            }

            dx[in_idx] = accum;
        }
    "#;

    const TINY_MATMUL_SHADER_SRC: &str = r#"
        struct Dims {
            M: u32,
            N: u32,
            K: u32,
            padding: u32,
        }

        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> c: array<f32>;
        @group(0) @binding(3) var<uniform> dims: Dims;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let idx = id.x;
            let total = dims.M * dims.N;
            if (idx >= total) { return; }

            let row = idx / dims.N;
            let col = idx % dims.N;

            var sum = 0.0;
            for (var k = 0u; k < dims.K; k++) {
                sum += a[row * dims.K + k] * b[k * dims.N + col];
            }
            c[idx] = sum;
        }
    "#;

    const MATMUL_SHADER_SRC: &str = r#"
        struct Dims {
            M: u32,
            N: u32,
            K: u32,
            padding: u32,
        }

        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> c: array<f32>;
        @group(0) @binding(3) var<uniform> dims: Dims;

        @compute @workgroup_size(16, 16)
        fn main(
            @builtin(local_invocation_id) local_id: vec3<u32>,
            @builtin(workgroup_id) workgroup_id: vec3<u32>
        ) {
            let local_row = local_id.y;
            let local_col = local_id.x;

            let wg_row = workgroup_id.y * 64u;
            let wg_col = workgroup_id.x * 64u;

            let row = wg_row + local_row * 4u;
            let col = wg_col + local_col * 4u;

            var sum00 = 0.0; var sum01 = 0.0; var sum02 = 0.0; var sum03 = 0.0;
            var sum10 = 0.0; var sum11 = 0.0; var sum12 = 0.0; var sum13 = 0.0;
            var sum20 = 0.0; var sum21 = 0.0; var sum22 = 0.0; var sum23 = 0.0;
            var sum30 = 0.0; var sum31 = 0.0; var sum32 = 0.0; var sum33 = 0.0;

            let k_limit = dims.K;
            for (var k = 0u; k < k_limit; k++) {
                var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
                if (row < dims.M) { a0 = a[row * dims.K + k]; }
                if (row + 1u < dims.M) { a1 = a[(row + 1u) * dims.K + k]; }
                if (row + 2u < dims.M) { a2 = a[(row + 2u) * dims.K + k]; }
                if (row + 3u < dims.M) { a3 = a[(row + 3u) * dims.K + k]; }

                var b0 = 0.0; var b1 = 0.0; var b2 = 0.0; var b3 = 0.0;
                if (col < dims.N) { b0 = b[k * dims.N + col]; }
                if (col + 1u < dims.N) { b1 = b[k * dims.N + col + 1u]; }
                if (col + 2u < dims.N) { b2 = b[k * dims.N + col + 2u]; }
                if (col + 3u < dims.N) { b3 = b[k * dims.N + col + 3u]; }

                sum00 += a0 * b0; sum01 += a0 * b1; sum02 += a0 * b2; sum03 += a0 * b3;
                sum10 += a1 * b0; sum11 += a1 * b1; sum12 += a1 * b2; sum13 += a1 * b3;
                sum20 += a2 * b0; sum21 += a2 * b1; sum22 += a2 * b2; sum23 += a2 * b3;
                sum30 += a3 * b0; sum31 += a3 * b1; sum32 += a3 * b2; sum33 += a3 * b3;
            }

            if (row < dims.M) {
                if (col < dims.N) { c[row * dims.N + col] = sum00; }
                if (col + 1u < dims.N) { c[row * dims.N + col + 1u] = sum01; }
                if (col + 2u < dims.N) { c[row * dims.N + col + 2u] = sum02; }
                if (col + 3u < dims.N) { c[row * dims.N + col + 3u] = sum03; }
            }
            if (row + 1u < dims.M) {
                let r = row + 1u;
                if (col < dims.N) { c[r * dims.N + col] = sum10; }
                if (col + 1u < dims.N) { c[r * dims.N + col + 1u] = sum11; }
                if (col + 2u < dims.N) { c[r * dims.N + col + 2u] = sum12; }
                if (col + 3u < dims.N) { c[r * dims.N + col + 3u] = sum13; }
            }
            if (row + 2u < dims.M) {
                let r = row + 2u;
                if (col < dims.N) { c[r * dims.N + col] = sum20; }
                if (col + 1u < dims.N) { c[r * dims.N + col + 1u] = sum21; }
                if (col + 2u < dims.N) { c[r * dims.N + col + 2u] = sum22; }
                if (col + 3u < dims.N) { c[r * dims.N + col + 3u] = sum23; }
            }
            if (row + 3u < dims.M) {
                let r = row + 3u;
                if (col < dims.N) { c[r * dims.N + col] = sum30; }
                if (col + 1u < dims.N) { c[r * dims.N + col + 1u] = sum31; }
                if (col + 2u < dims.N) { c[r * dims.N + col + 2u] = sum32; }
                if (col + 3u < dims.N) { c[r * dims.N + col + 3u] = sum33; }
            }
        }
    "#;

    const MATMUL_RELU_SHADER_SRC: &str = r#"
        struct Dims {
            M: u32,
            N: u32,
            K: u32,
            padding: u32,
        }

        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> out: array<f32>;
        @group(0) @binding(3) var<uniform> dims: Dims;

        @compute @workgroup_size(16, 16)
        fn main(
            @builtin(local_invocation_id) local_id: vec3<u32>,
            @builtin(workgroup_id) workgroup_id: vec3<u32>
        ) {
            let local_row = local_id.y;
            let local_col = local_id.x;

            let wg_row = workgroup_id.y * 64u;
            let wg_col = workgroup_id.x * 64u;

            let row = wg_row + local_row * 4u;
            let col = wg_col + local_col * 4u;

            var sum00 = 0.0; var sum01 = 0.0; var sum02 = 0.0; var sum03 = 0.0;
            var sum10 = 0.0; var sum11 = 0.0; var sum12 = 0.0; var sum13 = 0.0;
            var sum20 = 0.0; var sum21 = 0.0; var sum22 = 0.0; var sum23 = 0.0;
            var sum30 = 0.0; var sum31 = 0.0; var sum32 = 0.0; var sum33 = 0.0;

            let k_limit = dims.K;
            for (var k = 0u; k < k_limit; k++) {
                var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
                if (row < dims.M) { a0 = a[row * dims.K + k]; }
                if (row + 1u < dims.M) { a1 = a[(row + 1u) * dims.K + k]; }
                if (row + 2u < dims.M) { a2 = a[(row + 2u) * dims.K + k]; }
                if (row + 3u < dims.M) { a3 = a[(row + 3u) * dims.K + k]; }

                var b0 = 0.0; var b1 = 0.0; var b2 = 0.0; var b3 = 0.0;
                if (col < dims.N) { b0 = b[k * dims.N + col]; }
                if (col + 1u < dims.N) { b1 = b[k * dims.N + col + 1u]; }
                if (col + 2u < dims.N) { b2 = b[k * dims.N + col + 2u]; }
                if (col + 3u < dims.N) { b3 = b[k * dims.N + col + 3u]; }

                sum00 += a0 * b0; sum01 += a0 * b1; sum02 += a0 * b2; sum03 += a0 * b3;
                sum10 += a1 * b0; sum11 += a1 * b1; sum12 += a1 * b2; sum13 += a1 * b3;
                sum20 += a2 * b0; sum21 += a2 * b1; sum22 += a2 * b2; sum23 += a2 * b3;
                sum30 += a3 * b0; sum31 += a3 * b1; sum32 += a3 * b2; sum33 += a3 * b3;
            }

            if (row < dims.M) {
                if (col < dims.N) { out[row * dims.N + col] = max(sum00, 0.0); }
                if (col + 1u < dims.N) { out[row * dims.N + col + 1u] = max(sum01, 0.0); }
                if (col + 2u < dims.N) { out[row * dims.N + col + 2u] = max(sum02, 0.0); }
                if (col + 3u < dims.N) { out[row * dims.N + col + 3u] = max(sum03, 0.0); }
            }
            if (row + 1u < dims.M) {
                let r = row + 1u;
                if (col < dims.N) { out[r * dims.N + col] = max(sum10, 0.0); }
                if (col + 1u < dims.N) { out[r * dims.N + col + 1u] = max(sum11, 0.0); }
                if (col + 2u < dims.N) { out[r * dims.N + col + 2u] = max(sum12, 0.0); }
                if (col + 3u < dims.N) { out[r * dims.N + col + 3u] = max(sum13, 0.0); }
            }
            if (row + 2u < dims.M) {
                let r = row + 2u;
                if (col < dims.N) { out[r * dims.N + col] = max(sum20, 0.0); }
                if (col + 1u < dims.N) { out[r * dims.N + col + 1u] = max(sum21, 0.0); }
                if (col + 2u < dims.N) { out[r * dims.N + col + 2u] = max(sum22, 0.0); }
                if (col + 3u < dims.N) { out[r * dims.N + col + 3u] = max(sum23, 0.0); }
            }
            if (row + 3u < dims.M) {
                let r = row + 3u;
                if (col < dims.N) { out[r * dims.N + col] = max(sum30, 0.0); }
                if (col + 1u < dims.N) { out[r * dims.N + col + 1u] = max(sum31, 0.0); }
                if (col + 2u < dims.N) { out[r * dims.N + col + 2u] = max(sum32, 0.0); }
                if (col + 3u < dims.N) { out[r * dims.N + col + 3u] = max(sum33, 0.0); }
            }
        }
    "#;

    const MATMUL_ADD_SHADER_SRC: &str = r#"
        struct Dims {
            M: u32,
            N: u32,
            K: u32,
            padding: u32,
        }

        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> out: array<f32>;
        @group(0) @binding(3) var<uniform> dims: Dims;
        @group(0) @binding(4) var<storage, read> bias: array<f32>;

        @compute @workgroup_size(16, 16)
        fn main(
            @builtin(local_invocation_id) local_id: vec3<u32>,
            @builtin(workgroup_id) workgroup_id: vec3<u32>
        ) {
            let local_row = local_id.y;
            let local_col = local_id.x;

            let wg_row = workgroup_id.y * 64u;
            let wg_col = workgroup_id.x * 64u;

            let row = wg_row + local_row * 4u;
            let col = wg_col + local_col * 4u;

            var sum00 = 0.0; var sum01 = 0.0; var sum02 = 0.0; var sum03 = 0.0;
            var sum10 = 0.0; var sum11 = 0.0; var sum12 = 0.0; var sum13 = 0.0;
            var sum20 = 0.0; var sum21 = 0.0; var sum22 = 0.0; var sum23 = 0.0;
            var sum30 = 0.0; var sum31 = 0.0; var sum32 = 0.0; var sum33 = 0.0;

            let k_limit = dims.K;
            for (var k = 0u; k < k_limit; k++) {
                var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
                if (row < dims.M) { a0 = a[row * dims.K + k]; }
                if (row + 1u < dims.M) { a1 = a[(row + 1u) * dims.K + k]; }
                if (row + 2u < dims.M) { a2 = a[(row + 2u) * dims.K + k]; }
                if (row + 3u < dims.M) { a3 = a[(row + 3u) * dims.K + k]; }

                var b0 = 0.0; var b1 = 0.0; var b2 = 0.0; var b3 = 0.0;
                if (col < dims.N) { b0 = b[k * dims.N + col]; }
                if (col + 1u < dims.N) { b1 = b[k * dims.N + col + 1u]; }
                if (col + 2u < dims.N) { b2 = b[k * dims.N + col + 2u]; }
                if (col + 3u < dims.N) { b3 = b[k * dims.N + col + 3u]; }

                sum00 += a0 * b0; sum01 += a0 * b1; sum02 += a0 * b2; sum03 += a0 * b3;
                sum10 += a1 * b0; sum11 += a1 * b1; sum12 += a1 * b2; sum13 += a1 * b3;
                sum20 += a2 * b0; sum21 += a2 * b1; sum22 += a2 * b2; sum23 += a2 * b3;
                sum30 += a3 * b0; sum31 += a3 * b1; sum32 += a3 * b2; sum33 += a3 * b3;
            }

            let is_1d_bias = (arrayLength(&bias) == dims.N);

            if (row < dims.M) {
                if (col < dims.N) {
                    let idx = row * dims.N + col;
                    if (is_1d_bias) { out[idx] = sum00 + bias[col]; } else { out[idx] = sum00 + bias[idx]; }
                }
                if (col + 1u < dims.N) {
                    let idx = row * dims.N + col + 1u;
                    if (is_1d_bias) { out[idx] = sum01 + bias[col + 1u]; } else { out[idx] = sum01 + bias[idx]; }
                }
                if (col + 2u < dims.N) {
                    let idx = row * dims.N + col + 2u;
                    if (is_1d_bias) { out[idx] = sum02 + bias[col + 2u]; } else { out[idx] = sum02 + bias[idx]; }
                }
                if (col + 3u < dims.N) {
                    let idx = row * dims.N + col + 3u;
                    if (is_1d_bias) { out[idx] = sum03 + bias[col + 3u]; } else { out[idx] = sum03 + bias[idx]; }
                }
            }
            if (row + 1u < dims.M) {
                let r = row + 1u;
                if (col < dims.N) {
                    let idx = r * dims.N + col;
                    if (is_1d_bias) { out[idx] = sum10 + bias[col]; } else { out[idx] = sum10 + bias[idx]; }
                }
                if (col + 1u < dims.N) {
                    let idx = r * dims.N + col + 1u;
                    if (is_1d_bias) { out[idx] = sum11 + bias[col + 1u]; } else { out[idx] = sum11 + bias[idx]; }
                }
                if (col + 2u < dims.N) {
                    let idx = r * dims.N + col + 2u;
                    if (is_1d_bias) { out[idx] = sum12 + bias[col + 2u]; } else { out[idx] = sum12 + bias[idx]; }
                }
                if (col + 3u < dims.N) {
                    let idx = r * dims.N + col + 3u;
                    if (is_1d_bias) { out[idx] = sum13 + bias[col + 3u]; } else { out[idx] = sum13 + bias[idx]; }
                }
            }
            if (row + 2u < dims.M) {
                let r = row + 2u;
                if (col < dims.N) {
                    let idx = r * dims.N + col;
                    if (is_1d_bias) { out[idx] = sum20 + bias[col]; } else { out[idx] = sum20 + bias[idx]; }
                }
                if (col + 1u < dims.N) {
                    let idx = r * dims.N + col + 1u;
                    if (is_1d_bias) { out[idx] = sum21 + bias[col + 1u]; } else { out[idx] = sum21 + bias[idx]; }
                }
                if (col + 2u < dims.N) {
                    let idx = r * dims.N + col + 2u;
                    if (is_1d_bias) { out[idx] = sum22 + bias[col + 2u]; } else { out[idx] = sum22 + bias[idx]; }
                }
                if (col + 3u < dims.N) {
                    let idx = r * dims.N + col + 3u;
                    if (is_1d_bias) { out[idx] = sum23 + bias[col + 3u]; } else { out[idx] = sum23 + bias[idx]; }
                }
            }
            if (row + 3u < dims.M) {
                let r = row + 3u;
                if (col < dims.N) {
                    let idx = r * dims.N + col;
                    if (is_1d_bias) { out[idx] = sum30 + bias[col]; } else { out[idx] = sum30 + bias[idx]; }
                }
                if (col + 1u < dims.N) {
                    let idx = r * dims.N + col + 1u;
                    if (is_1d_bias) { out[idx] = sum31 + bias[col + 1u]; } else { out[idx] = sum31 + bias[idx]; }
                }
                if (col + 2u < dims.N) {
                    let idx = r * dims.N + col + 2u;
                    if (is_1d_bias) { out[idx] = sum32 + bias[col + 2u]; } else { out[idx] = sum32 + bias[idx]; }
                }
                if (col + 3u < dims.N) {
                    let idx = r * dims.N + col + 3u;
                    if (is_1d_bias) { out[idx] = sum33 + bias[col + 3u]; } else { out[idx] = sum33 + bias[idx]; }
                }
            }
        }
    "#;

    pub const MATMUL_ADD_RELU_SHADER_SRC: &str = r#"
        struct Dims {
            M: u32,
            N: u32,
            K: u32,
            padding: u32,
        }

        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> out: array<f32>;
        @group(0) @binding(3) var<uniform> dims: Dims;
        @group(0) @binding(4) var<storage, read> bias: array<f32>;

        @compute @workgroup_size(16, 16)
        fn main(
            @builtin(local_invocation_id) local_id: vec3<u32>,
            @builtin(workgroup_id) workgroup_id: vec3<u32>
        ) {
            let local_row = local_id.y;
            let local_col = local_id.x;

            let wg_row = workgroup_id.y * 64u;
            let wg_col = workgroup_id.x * 64u;

            let row = wg_row + local_row * 4u;
            let col = wg_col + local_col * 4u;

            var sum00 = 0.0; var sum01 = 0.0; var sum02 = 0.0; var sum03 = 0.0;
            var sum10 = 0.0; var sum11 = 0.0; var sum12 = 0.0; var sum13 = 0.0;
            var sum20 = 0.0; var sum21 = 0.0; var sum22 = 0.0; var sum23 = 0.0;
            var sum30 = 0.0; var sum31 = 0.0; var sum32 = 0.0; var sum33 = 0.0;

            let k_limit = dims.K;
            for (var k = 0u; k < k_limit; k++) {
                var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
                if (row < dims.M) { a0 = a[row * dims.K + k]; }
                if (row + 1u < dims.M) { a1 = a[(row + 1u) * dims.K + k]; }
                if (row + 2u < dims.M) { a2 = a[(row + 2u) * dims.K + k]; }
                if (row + 3u < dims.M) { a3 = a[(row + 3u) * dims.K + k]; }

                var b0 = 0.0; var b1 = 0.0; var b2 = 0.0; var b3 = 0.0;
                if (col < dims.N) { b0 = b[k * dims.N + col]; }
                if (col + 1u < dims.N) { b1 = b[k * dims.N + col + 1u]; }
                if (col + 2u < dims.N) { b2 = b[k * dims.N + col + 2u]; }
                if (col + 3u < dims.N) { b3 = b[k * dims.N + col + 3u]; }

                sum00 += a0 * b0; sum01 += a0 * b1; sum02 += a0 * b2; sum03 += a0 * b3;
                sum10 += a1 * b0; sum11 += a1 * b1; sum12 += a1 * b2; sum13 += a1 * b3;
                sum20 += a2 * b0; sum21 += a2 * b1; sum22 += a2 * b2; sum23 += a2 * b3;
                sum30 += a3 * b0; sum31 += a3 * b1; sum32 += a3 * b2; sum33 += a3 * b3;
            }

            let is_1d_bias = (arrayLength(&bias) == dims.N);

            if (row < dims.M) {
                if (col < dims.N) {
                    let idx = row * dims.N + col;
                    if (is_1d_bias) { out[idx] = max(sum00 + bias[col], 0.0); } else { out[idx] = max(sum00 + bias[idx], 0.0); }
                }
                if (col + 1u < dims.N) {
                    let idx = row * dims.N + col + 1u;
                    if (is_1d_bias) { out[idx] = max(sum01 + bias[col + 1u], 0.0); } else { out[idx] = max(sum01 + bias[idx], 0.0); }
                }
                if (col + 2u < dims.N) {
                    let idx = row * dims.N + col + 2u;
                    if (is_1d_bias) { out[idx] = max(sum02 + bias[col + 2u], 0.0); } else { out[idx] = max(sum02 + bias[idx], 0.0); }
                }
                if (col + 3u < dims.N) {
                    let idx = row * dims.N + col + 3u;
                    if (is_1d_bias) { out[idx] = max(sum03 + bias[col + 3u], 0.0); } else { out[idx] = max(sum03 + bias[idx], 0.0); }
                }
            }
            if (row + 1u < dims.M) {
                let r = row + 1u;
                if (col < dims.N) {
                    let idx = r * dims.N + col;
                    if (is_1d_bias) { out[idx] = max(sum10 + bias[col], 0.0); } else { out[idx] = max(sum10 + bias[idx], 0.0); }
                }
                if (col + 1u < dims.N) {
                    let idx = r * dims.N + col + 1u;
                    if (is_1d_bias) { out[idx] = max(sum11 + bias[col + 1u], 0.0); } else { out[idx] = max(sum11 + bias[idx], 0.0); }
                }
                if (col + 2u < dims.N) {
                    let idx = r * dims.N + col + 2u;
                    if (is_1d_bias) { out[idx] = max(sum12 + bias[col + 2u], 0.0); } else { out[idx] = max(sum12 + bias[idx], 0.0); }
                }
                if (col + 3u < dims.N) {
                    let idx = r * dims.N + col + 3u;
                    if (is_1d_bias) { out[idx] = max(sum13 + bias[col + 3u], 0.0); } else { out[idx] = max(sum13 + bias[idx], 0.0); }
                }
            }
            if (row + 2u < dims.M) {
                let r = row + 2u;
                if (col < dims.N) {
                    let idx = r * dims.N + col;
                    if (is_1d_bias) { out[idx] = max(sum20 + bias[col], 0.0); } else { out[idx] = max(sum20 + bias[idx], 0.0); }
                }
                if (col + 1u < dims.N) {
                    let idx = r * dims.N + col + 1u;
                    if (is_1d_bias) { out[idx] = max(sum21 + bias[col + 1u], 0.0); } else { out[idx] = max(sum21 + bias[idx], 0.0); }
                }
                if (col + 2u < dims.N) {
                    let idx = r * dims.N + col + 2u;
                    if (is_1d_bias) { out[idx] = max(sum22 + bias[col + 2u], 0.0); } else { out[idx] = max(sum22 + bias[idx], 0.0); }
                }
                if (col + 3u < dims.N) {
                    let idx = r * dims.N + col + 3u;
                    if (is_1d_bias) { out[idx] = max(sum23 + bias[col + 3u], 0.0); } else { out[idx] = max(sum23 + bias[idx], 0.0); }
                }
            }
            if (row + 3u < dims.M) {
                let r = row + 3u;
                if (col < dims.N) {
                    let idx = r * dims.N + col;
                    if (is_1d_bias) { out[idx] = max(sum30 + bias[col], 0.0); } else { out[idx] = max(sum30 + bias[idx], 0.0); }
                }
                if (col + 1u < dims.N) {
                    let idx = r * dims.N + col + 1u;
                    if (is_1d_bias) { out[idx] = max(sum31 + bias[col + 1u], 0.0); } else { out[idx] = max(sum31 + bias[idx], 0.0); }
                }
                if (col + 2u < dims.N) {
                    let idx = r * dims.N + col + 2u;
                    if (is_1d_bias) { out[idx] = max(sum32 + bias[col + 2u], 0.0); } else { out[idx] = max(sum32 + bias[idx], 0.0); }
                }
                if (col + 3u < dims.N) {
                    let idx = r * dims.N + col + 3u;
                    if (is_1d_bias) { out[idx] = max(sum33 + bias[col + 3u], 0.0); } else { out[idx] = max(sum33 + bias[idx], 0.0); }
                }
            }
        }
    "#;

    const ROPE_SHADER_SRC: &str = r#"
        struct RoPEParams {
            n_tokens: u32,
            n_heads: u32,
            head_dim: u32,
            start_pos: u32,
        }

        @group(0) @binding(0) var<storage, read_write> buf: array<f32>;
        @group(0) @binding(1) var<storage, read> sin: array<f32>;
        @group(0) @binding(2) var<storage, read> cos: array<f32>;
        @group(0) @binding(3) var<uniform> params: RoPEParams;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let idx = id.x;
            let total = params.n_tokens * params.n_heads * (params.head_dim / 2u);
            if (idx >= total) { return; }

            let pair = idx % (params.head_dim / 2u);
            let head = (idx / (params.head_dim / 2u)) % params.n_heads;
            let token = idx / (params.n_heads * (params.head_dim / 2u));

            let pos = params.start_pos + token;
            let dim_offset = pair * 2u;
            let base = token * params.n_heads * params.head_dim + head * params.head_dim;

            let x0 = buf[base + dim_offset];
            let x1 = buf[base + dim_offset + 1u];

            let sin_val = sin[pos * (params.head_dim / 2u) + pair];
            let cos_val = cos[pos * (params.head_dim / 2u) + pair];

            buf[base + dim_offset] = x0 * cos_val - x1 * sin_val;
            buf[base + dim_offset + 1u] = x0 * sin_val + x1 * cos_val;
        }
    "#;

    const DEQUANT_Q4_K_SHADER_SRC: &str = r#"
        @group(0) @binding(0) var<storage, read> quant_data: array<u32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;

        fn read_byte(byte_idx: u32) -> u32 {
            let word = quant_data[byte_idx / 4u];
            return (word >> ((byte_idx % 4u) * 8u)) & 0xFFu;
        }

        fn f16_to_f32(bits: u32) -> f32 {
            let sign = select(1.0, -1.0, (bits & 0x8000u) != 0u);
            let exponent = (bits >> 10u) & 0x1Fu;
            let mantissa = bits & 0x3FFu;
            
            if (exponent == 0u) {
                if (mantissa == 0u) {
                    return 0.0;
                } else {
                    return sign * f32(mantissa) / 1024.0 * 0.00006103515625;
                }
            } else if (exponent == 31u) {
                if (mantissa == 0u) {
                    return sign * bitcast<f32>(0x7F800000u);
                } else {
                    return sign * bitcast<f32>(0x7FC00000u);
                }
            } else {
                return sign * (1.0 + f32(mantissa) / 1024.0) * pow(2.0, f32(exponent) - 15.0);
            }
        }

        fn get_scale_min_k4(j: u32, bo: u32) -> vec2<f32> {
            if (j < 4u) {
                let s_j = read_byte(bo + 4u + j);
                let s_j4 = read_byte(bo + 4u + j + 4u);
                return vec2<f32>(f32(s_j & 63u), f32(s_j4 & 63u));
            } else {
                let s_j4 = read_byte(bo + 4u + j + 4u);
                let s_j_minus_4 = read_byte(bo + 4u + j - 4u);
                let s_j = read_byte(bo + 4u + j);
                let sc = (s_j4 & 0x0Fu) | ((s_j_minus_4 >> 6u) << 4u);
                let mm = (s_j4 >> 4u) | ((s_j >> 6u) << 4u);
                return vec2<f32>(f32(sc), f32(mm));
            }
        }

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx >= arrayLength(&output)) { return; }

            let block_idx = idx / 256u;
            let local_idx = idx % 256u;
            let bo = block_idx * 144u;

            let d_bits = (read_byte(bo + 1u) << 8u) | read_byte(bo + 0u);
            let d = f16_to_f32(d_bits);
            let dmin_bits = (read_byte(bo + 3u) << 8u) | read_byte(bo + 2u);
            let dmin = f16_to_f32(dmin_bits);

            let chunk = local_idx / 64u;
            let is_val = chunk * 2u;
            let l = local_idx % 32u;
            let is_hi = (local_idx % 64u) >= 32u;

            let qs_byte = read_byte(bo + 16u + chunk * 32u + l);

            if (!is_hi) {
                let sc_mm = get_scale_min_k4(is_val, bo);
                let q = f32(qs_byte & 0x0Fu);
                output[idx] = d * sc_mm.x * q - dmin * sc_mm.y;
            } else {
                let sc_mm = get_scale_min_k4(is_val + 1u, bo);
                let q = f32((qs_byte >> 4u) & 0x0Fu);
                output[idx] = d * sc_mm.x * q - dmin * sc_mm.y;
            }
        }
    "#;

    const DEQUANT_Q5_K_SHADER_SRC: &str = r#"
        @group(0) @binding(0) var<storage, read> quant_data: array<u32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;

        fn read_byte(byte_idx: u32) -> u32 {
            let word = quant_data[byte_idx / 4u];
            return (word >> ((byte_idx % 4u) * 8u)) & 0xFFu;
        }

        fn f16_to_f32(bits: u32) -> f32 {
            let sign = select(1.0, -1.0, (bits & 0x8000u) != 0u);
            let exponent = (bits >> 10u) & 0x1Fu;
            let mantissa = bits & 0x3FFu;
            
            if (exponent == 0u) {
                if (mantissa == 0u) {
                    return 0.0;
                } else {
                    return sign * f32(mantissa) / 1024.0 * 0.00006103515625;
                }
            } else if (exponent == 31u) {
                if (mantissa == 0u) {
                    return sign * bitcast<f32>(0x7F800000u);
                } else {
                    return sign * bitcast<f32>(0x7FC00000u);
                }
            } else {
                return sign * (1.0 + f32(mantissa) / 1024.0) * pow(2.0, f32(exponent) - 15.0);
            }
        }

        fn get_scale_min_k4(j: u32, bo: u32) -> vec2<f32> {
            if (j < 4u) {
                let s_j = read_byte(bo + 4u + j);
                let s_j4 = read_byte(bo + 4u + j + 4u);
                return vec2<f32>(f32(s_j & 63u), f32(s_j4 & 63u));
            } else {
                let s_j4 = read_byte(bo + 4u + j + 4u);
                let s_j_minus_4 = read_byte(bo + 4u + j - 4u);
                let s_j = read_byte(bo + 4u + j);
                let sc = (s_j4 & 0x0Fu) | ((s_j_minus_4 >> 6u) << 4u);
                let mm = (s_j4 >> 4u) | ((s_j >> 6u) << 4u);
                return vec2<f32>(f32(sc), f32(mm));
            }
        }

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx >= arrayLength(&output)) { return; }

            let block_idx = idx / 256u;
            let local_idx = idx % 256u;
            let bo = block_idx * 176u;

            let d_bits = (read_byte(bo + 1u) << 8u) | read_byte(bo + 0u);
            let d = f16_to_f32(d_bits);
            let dmin_bits = (read_byte(bo + 3u) << 8u) | read_byte(bo + 2u);
            let dmin = f16_to_f32(dmin_bits);

            let chunk = local_idx / 64u;
            let is_val = chunk * 2u;
            let l = local_idx % 32u;
            let is_hi = (local_idx % 64u) >= 32u;

            let qs_byte = read_byte(bo + 48u + chunk * 32u + l);

            if (!is_hi) {
                let sc_mm = get_scale_min_k4(is_val, bo);
                let ql = qs_byte & 0x0Fu;
                let qh_byte = read_byte(bo + 16u + (is_val * 32u + l) / 8u);
                let qh_bit = (qh_byte >> ((is_val * 32u + l) % 8u)) & 1u;
                let q = i32(ql | (qh_bit << 4u));
                output[idx] = d * sc_mm.x * f32(q) - dmin * sc_mm.y;
            } else {
                let sc_mm = get_scale_min_k4(is_val + 1u, bo);
                let ql = (qs_byte >> 4u) & 0x0Fu;
                let qh_byte = read_byte(bo + 16u + ((is_val + 1u) * 32u + l) / 8u);
                let qh_bit = (qh_byte >> (((is_val + 1u) * 32u + l) % 8u)) & 1u;
                let q = i32(ql | (qh_bit << 4u));
                output[idx] = d * sc_mm.x * f32(q) - dmin * sc_mm.y;
            }
        }
    "#;

    const DEQUANT_Q6_K_SHADER_SRC: &str = r#"
        @group(0) @binding(0) var<storage, read> quant_data: array<u32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;

        fn read_byte(byte_idx: u32) -> u32 {
            let word = quant_data[byte_idx / 4u];
            return (word >> ((byte_idx % 4u) * 8u)) & 0xFFu;
        }

        fn f16_to_f32(bits: u32) -> f32 {
            let sign = select(1.0, -1.0, (bits & 0x8000u) != 0u);
            let exponent = (bits >> 10u) & 0x1Fu;
            let mantissa = bits & 0x3FFu;
            
            if (exponent == 0u) {
                if (mantissa == 0u) {
                    return 0.0;
                } else {
                    return sign * f32(mantissa) / 1024.0 * 0.00006103515625;
                }
            } else if (exponent == 31u) {
                if (mantissa == 0u) {
                    return sign * bitcast<f32>(0x7F800000u);
                } else {
                    return sign * bitcast<f32>(0x7FC00000u);
                }
            } else {
                return sign * (1.0 + f32(mantissa) / 1024.0) * pow(2.0, f32(exponent) - 15.0);
            }
        }

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx >= arrayLength(&output)) { return; }

            let block_idx = idx / 256u;
            let local_idx = idx % 256u;
            let bo = block_idx * 210u;

            let d_bits = (read_byte(bo + 209u) << 8u) | read_byte(bo + 208u);
            let d = f16_to_f32(d_bits);

            let half_val = local_idx / 128u;
            let half_idx = local_idx % 128u;
            let l = half_idx % 32u;
            let group = half_idx / 32u;

            let ql_off = half_val * 64u;
            let qh_off = half_val * 32u;
            let sc_off = half_val * 8u;
            let is_val = l / 16u;

            let ql_l = read_byte(bo + ql_off + l);
            let ql_l32 = read_byte(bo + ql_off + l + 32u);
            let qh_l = read_byte(bo + 128u + qh_off + l);

            var q: i32 = 0;
            var scale_byte: u32 = 0u;

            if (group == 0u) {
                q = i32((ql_l & 0x0Fu) | ((qh_l & 0x03u) << 4u)) - 32;
                scale_byte = read_byte(bo + 192u + sc_off + is_val + 0u);
            } else if (group == 1u) {
                q = i32((ql_l32 & 0x0Fu) | ((qh_l & 0x0Cu) << 2u)) - 32;
                scale_byte = read_byte(bo + 192u + sc_off + is_val + 2u);
            } else if (group == 2u) {
                q = i32((ql_l >> 4u) | (qh_l & 0x30u)) - 32;
                scale_byte = read_byte(bo + 192u + sc_off + is_val + 4u);
            } else {
                q = i32((ql_l32 >> 4u) | ((qh_l & 0xC0u) >> 2u)) - 32;
                scale_byte = read_byte(bo + 192u + sc_off + is_val + 6u);
            }

            let scale_val = f32(i32(scale_byte) - select(0, 256, (scale_byte & 128u) != 0u));
            output[idx] = d * scale_val * f32(q);
        }
    "#;

    const DEQUANT_Q8_0_SHADER_SRC: &str = r#"
        @group(0) @binding(0) var<storage, read> quant_data: array<u32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;

        fn read_byte(byte_idx: u32) -> u32 {
            let word = quant_data[byte_idx / 4u];
            return (word >> ((byte_idx % 4u) * 8u)) & 0xFFu;
        }

        fn f16_to_f32(bits: u32) -> f32 {
            let sign = select(1.0, -1.0, (bits & 0x8000u) != 0u);
            let exponent = (bits >> 10u) & 0x1Fu;
            let mantissa = bits & 0x3FFu;
            
            if (exponent == 0u) {
                if (mantissa == 0u) {
                    return 0.0;
                } else {
                    return sign * f32(mantissa) / 1024.0 * 0.00006103515625;
                }
            } else if (exponent == 31u) {
                if (mantissa == 0u) {
                    return sign * bitcast<f32>(0x7F800000u);
                } else {
                    return sign * bitcast<f32>(0x7FC00000u);
                }
            } else {
                return sign * (1.0 + f32(mantissa) / 1024.0) * pow(2.0, f32(exponent) - 15.0);
            }
        }

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx >= arrayLength(&output)) { return; }

            let block_idx = idx / 32u;
            let local_idx = idx % 32u;
            let bo = block_idx * 34u;

            let d_bits = (read_byte(bo + 1u) << 8u) | read_byte(bo + 0u);
            let d = f16_to_f32(d_bits);

            let qs_byte = read_byte(bo + 2u + local_idx);
            let q = f32(i32(qs_byte) - select(0, 256, (qs_byte & 128u) != 0u));
            output[idx] = d * q;
        }
    "#;
