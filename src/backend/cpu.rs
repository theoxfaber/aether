pub mod cpu_mod {
    //! Host CPU backend with scalar, NEON, and BLAS matmul kernels.
    //!
    //! # Safety
    //!
    //! This module calls `crate::blas::sgemm` which takes raw pointers.
    //! The pointers are derived from safe `&[f32]` / `&mut [f32]` slices
    //! whose lengths have been validated, so the FFI calls are safe in
    //! practice, but the Rust compiler cannot verify this.
    #![allow(unsafe_code)]
    use crate::backend::Backend;
    use crate::graph::Op;
    use crate::tensor::{Shape, Tensor};
    use crate::Error;
    use ndarray;

    /// CPU execution backend using ndarray for linear algebra and multi-dimensional broadcasting.
    pub struct CpuBackend;

    impl CpuBackend {
        /// Create a new CPU backend instance.
        pub fn new() -> Self {
            Self
        }
    }

    impl Default for CpuBackend {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Backend for CpuBackend {
        /// Execute an operation on host CPU buffers.
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
                    crate::blas::matmul(lhs, rhs)
                }
                Op::Relu => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "Relu requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let out_data =
                        crate::parallel::parallel_map(
                            input.data(),
                            |x| if x > 0.0 { x } else { 0.0 },
                        );
                    Ok(Tensor::new(out_data, input.shape().clone()))
                }
                Op::Add => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "Add requires exactly 2 inputs".to_string(),
                        ));
                    }
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    if lhs.shape() != rhs.shape() {
                        return Err(Error::ShapeMismatch(format!(
                            "Add shape mismatch: {:?} and {:?}",
                            lhs.shape(),
                            rhs.shape()
                        )));
                    }
                    let out_data =
                        crate::parallel::parallel_map2(lhs.data(), rhs.data(), |a, b| a + b);
                    Ok(Tensor::new(out_data, lhs.shape().clone()))
                }
                Op::Sub => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "Sub requires exactly 2 inputs".to_string(),
                        ));
                    }
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    let out_data =
                        crate::parallel::parallel_map2(lhs.data(), rhs.data(), |a, b| a - b);
                    Ok(Tensor::new(out_data, lhs.shape().clone()))
                }
                Op::Mul => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "Mul requires exactly 2 inputs".to_string(),
                        ));
                    }
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    let out_data =
                        crate::parallel::parallel_map2(lhs.data(), rhs.data(), |a, b| a * b);
                    Ok(Tensor::new(out_data, lhs.shape().clone()))
                }
                Op::Div => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "Div requires exactly 2 inputs".to_string(),
                        ));
                    }
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    let out_data =
                        crate::parallel::parallel_map2(lhs.data(), rhs.data(), |a, b| a / b);
                    Ok(Tensor::new(out_data, lhs.shape().clone()))
                }
                Op::Tanh => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "Tanh requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let out_data = crate::parallel::parallel_map(input.data(), |x| x.tanh());
                    Ok(Tensor::new(out_data, input.shape().clone()))
                }
                Op::Sigmoid => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "Sigmoid requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let out_data =
                        crate::parallel::parallel_map(input.data(), |x| 1.0 / (1.0 + (-x).exp()));
                    Ok(Tensor::new(out_data, input.shape().clone()))
                }
                Op::Exp => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "Exp requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let out_data = crate::parallel::parallel_map(input.data(), |x| x.exp());
                    Ok(Tensor::new(out_data, input.shape().clone()))
                }
                Op::Sqrt => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "Sqrt requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let out_data = crate::parallel::parallel_map(input.data(), |x| x.sqrt());
                    Ok(Tensor::new(out_data, input.shape().clone()))
                }
                Op::Neg => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "Neg requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let out_data = crate::parallel::parallel_map(input.data(), |x| -x);
                    Ok(Tensor::new(out_data, input.shape().clone()))
                }
                Op::BroadcastAdd { .. } => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "BroadcastAdd requires exactly 2 inputs".to_string(),
                        ));
                    }
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    let out_shape = crate::graph::broadcast_shapes(lhs.shape(), rhs.shape())
                        .ok_or_else(|| {
                            Error::ExecutionError(
                                "Shapes are not broadcastable for Add".to_string(),
                            )
                        })?;
                    Ok(broadcast_op(lhs, rhs, &out_shape, |a, b| a + b))
                }
                Op::BroadcastMul { .. } => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "BroadcastMul requires exactly 2 inputs".to_string(),
                        ));
                    }
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    let out_shape = crate::graph::broadcast_shapes(lhs.shape(), rhs.shape())
                        .ok_or_else(|| {
                            Error::ExecutionError(
                                "Shapes are not broadcastable for Mul".to_string(),
                            )
                        })?;
                    Ok(broadcast_op(lhs, rhs, &out_shape, |a, b| a * b))
                }
                Op::BroadcastSub { .. } => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "BroadcastSub requires exactly 2 inputs".to_string(),
                        ));
                    }
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    let out_shape = crate::graph::broadcast_shapes(lhs.shape(), rhs.shape())
                        .ok_or_else(|| {
                            Error::ExecutionError(
                                "Shapes are not broadcastable for Sub".to_string(),
                            )
                        })?;
                    Ok(broadcast_op(lhs, rhs, &out_shape, |a, b| a - b))
                }
                Op::BroadcastDiv { .. } => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "BroadcastDiv requires exactly 2 inputs".to_string(),
                        ));
                    }
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    let out_shape = crate::graph::broadcast_shapes(lhs.shape(), rhs.shape())
                        .ok_or_else(|| {
                            Error::ExecutionError(
                                "Shapes are not broadcastable for Div".to_string(),
                            )
                        })?;
                    Ok(broadcast_op(lhs, rhs, &out_shape, |a, b| a / b))
                }
                Op::Transpose => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "Transpose requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let dims = input.shape().dims();
                    if dims.len() != 2 {
                        return Err(Error::ExecutionError(
                            "Transpose is only supported for 2D matrices".to_string(),
                        ));
                    }
                    let m = dims[0];
                    let n = dims[1];
                    let data = input.data();
                    let mut out_data = vec![0.0; m * n];
                    for r in 0..m {
                        for c in 0..n {
                            out_data[c * m + r] = data[r * n + c];
                        }
                    }
                    Ok(Tensor::new(out_data, Shape::new(vec![n, m])))
                }
                Op::SumAll => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "SumAll requires exactly 1 input".to_string(),
                        ));
                    }
                    let sum: f32 = inputs[0].data().iter().sum();
                    Ok(Tensor::new(vec![sum], Shape::new(vec![1])))
                }
                Op::SumDim { axis } => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "SumDim requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let shape = input.shape();
                    if *axis >= shape.ndim() {
                        return Err(Error::ExecutionError(
                            "SumDim axis out of bounds".to_string(),
                        ));
                    }
                    let dims = shape.dims();
                    let mut out_dims = dims.to_vec();
                    out_dims[*axis] = 1;
                    let out_shape = Shape::new(out_dims);

                    let num_elements = out_shape.num_elements();
                    let mut out_data = vec![0.0; num_elements];

                    let out_strides = get_strides(out_shape.dims());

                    for idx in 0..input.data().len() {
                        let mut temp = idx;
                        let mut coords = vec![0; dims.len()];
                        for i in (0..dims.len()).rev() {
                            coords[i] = temp % dims[i];
                            temp /= dims[i];
                        }
                        coords[*axis] = 0;
                        let mut out_idx = 0;
                        for i in 0..dims.len() {
                            out_idx += coords[i] * out_strides[i];
                        }
                        out_data[out_idx] += input.data()[idx];
                    }

                    Ok(Tensor::new(out_data, out_shape))
                }
                Op::Reshape { shape } => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "Reshape requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    if input.data().len() != shape.num_elements() {
                        return Err(Error::ExecutionError(
                            "Reshape element count mismatch".to_string(),
                        ));
                    }
                    Ok(Tensor::new(input.data().to_vec(), shape.clone()))
                }
                Op::Softmax => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "Softmax requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let dims = input.shape().dims();
                    if dims.is_empty() {
                        return Err(Error::ExecutionError(
                            "Softmax requires at least 1 dimension".to_string(),
                        ));
                    }
                    let cols = dims[dims.len() - 1];
                    let rows = input.data().len() / cols;
                    let data = input.data();
                    let mut out_data = vec![0.0; data.len()];

                    for r in 0..rows {
                        let start = r * cols;
                        let end = start + cols;
                        let row_slice = &data[start..end];

                        let max_val = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let sum: f32 = row_slice.iter().map(|&x| (x - max_val).exp()).sum();
                        for c in 0..cols {
                            out_data[start + c] = ((row_slice[c] - max_val).exp()) / sum;
                        }
                    }
                    Ok(Tensor::new(out_data, input.shape().clone()))
                }
                Op::SoftmaxGrad => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "SoftmaxGrad requires exactly 2 inputs (softmax_x, d_out)".to_string(),
                        ));
                    }
                    let softmax_x = inputs[0];
                    let d_out = inputs[1];
                    let dims = softmax_x.shape().dims();
                    let cols = dims[dims.len() - 1];
                    let rows = softmax_x.data().len() / cols;
                    let s_data = softmax_x.data();
                    let d_data = d_out.data();
                    let mut out_data = vec![0.0; s_data.len()];

                    for r in 0..rows {
                        let start = r * cols;

                        let mut sum = 0.0;
                        for c in 0..cols {
                            sum += d_data[start + c] * s_data[start + c];
                        }

                        for c in 0..cols {
                            let idx = start + c;
                            out_data[idx] = s_data[idx] * (d_data[idx] - sum);
                        }
                    }
                    Ok(Tensor::new(out_data, softmax_x.shape().clone()))
                }
                Op::Step => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "Step requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let out_data = input
                        .data()
                        .iter()
                        .map(|&x| if x >= 0.0 { 1.0 } else { 0.0 })
                        .collect();
                    Ok(Tensor::new(out_data, input.shape().clone()))
                }
                Op::Concat { axis } => {
                    if inputs.is_empty() {
                        return Err(Error::ExecutionError(
                            "Concat requires at least 1 input".to_string(),
                        ));
                    }
                    let first_shape = inputs[0].shape();
                    let ndim = first_shape.ndim();
                    if *axis >= ndim {
                        return Err(Error::ExecutionError(
                            "Concat axis out of bounds".to_string(),
                        ));
                    }

                    let mut concat_sizes = Vec::new();
                    let mut total_concat_size = 0;
                    for t in inputs {
                        let size = t.shape().dims()[*axis];
                        concat_sizes.push(size);
                        total_concat_size += size;
                    }

                    let mut out_dims = first_shape.dims().to_vec();
                    out_dims[*axis] = total_concat_size;
                    let out_shape = Shape::new(out_dims);

                    let num_elements = out_shape.num_elements();
                    let mut out_data = vec![0.0; num_elements];

                    let _out_strides = get_strides(out_shape.dims());
                    let input_strides: Vec<Vec<usize>> = inputs
                        .iter()
                        .map(|t| get_strides(t.shape().dims()))
                        .collect();

                    for (idx, out_val) in out_data.iter_mut().enumerate() {
                        let mut temp = idx;
                        let mut coords = vec![0; ndim];
                        for i in (0..ndim).rev() {
                            coords[i] = temp % out_shape.dims()[i];
                            temp /= out_shape.dims()[i];
                        }

                        let val_at_axis = coords[*axis];
                        let mut accum = 0;
                        let mut tensor_idx = 0;
                        let mut axis_offset = 0;
                        for (i, &size) in concat_sizes.iter().enumerate() {
                            if val_at_axis < accum + size {
                                tensor_idx = i;
                                axis_offset = val_at_axis - accum;
                                break;
                            }
                            accum += size;
                        }

                        let mut input_coords = coords.clone();
                        input_coords[*axis] = axis_offset;

                        let t = inputs[tensor_idx];
                        let mut input_idx = 0;
                        for i in 0..ndim {
                            input_idx += input_coords[i] * input_strides[tensor_idx][i];
                        }
                        *out_val = t.data()[input_idx];
                    }
                    Ok(Tensor::new(out_data, out_shape))
                }
                Op::LayerNorm { epsilon } => {
                    if inputs.len() != 3 {
                        return Err(Error::ExecutionError(
                            "LayerNorm requires exactly 3 inputs: x, weight, bias".to_string(),
                        ));
                    }
                    let x = inputs[0];
                    let weight = inputs[1];
                    let bias = inputs[2];

                    let dims = x.shape().dims();
                    if dims.is_empty() {
                        return Err(Error::ExecutionError(
                            "LayerNorm requires at least 1 dimension".to_string(),
                        ));
                    }
                    let last_dim = dims[dims.len() - 1];
                    let rows = x.data().len() / last_dim;

                    let x_data = x.data();
                    let w_data = weight.data();
                    let b_data = bias.data();

                    assert_eq!(w_data.len(), last_dim, "LayerNorm weight size mismatch");
                    assert_eq!(b_data.len(), last_dim, "LayerNorm bias size mismatch");

                    let mut out_data = vec![0.0; x_data.len()];
                    for r in 0..rows {
                        let start = r * last_dim;
                        let end = start + last_dim;
                        let row_slice = &x_data[start..end];

                        let mean: f32 = row_slice.iter().sum::<f32>() / last_dim as f32;
                        let var: f32 = row_slice
                            .iter()
                            .map(|&val| (val - mean).powi(2))
                            .sum::<f32>()
                            / last_dim as f32;
                        let std = (var + epsilon).sqrt();

                        for c in 0..last_dim {
                            let idx = start + c;
                            out_data[idx] = ((x_data[idx] - mean) / std) * w_data[c] + b_data[c];
                        }
                    }
                    Ok(Tensor::new(out_data, x.shape().clone()))
                }
                Op::RmsNorm { epsilon } => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "RMSNorm requires exactly 2 inputs: x, weight".to_string(),
                        ));
                    }
                    let x = inputs[0];
                    let weight = inputs[1];
                    let dims = x.shape().dims();
                    if dims.is_empty() {
                        return Err(Error::ExecutionError(
                            "RMSNorm requires at least 1 dimension".to_string(),
                        ));
                    }
                    let last_dim = dims[dims.len() - 1];
                    let rows = x.data().len() / last_dim;
                    let x_data = x.data();
                    let w_data = weight.data();
                    assert_eq!(w_data.len(), last_dim, "RMSNorm weight size mismatch");
                    let mut out_data = vec![0.0; x_data.len()];
                    for r in 0..rows {
                        let start = r * last_dim;
                        let row_slice = &x_data[start..(start + last_dim)];
                        let ssq: f32 =
                            row_slice.iter().map(|&v| v * v).sum::<f32>() / last_dim as f32;
                        let rms = (ssq + epsilon).sqrt();
                        for (c, w) in w_data.iter().enumerate() {
                            let idx = start + c;
                            out_data[idx] = (x_data[idx] / rms) * w;
                        }
                    }
                    Ok(Tensor::new(out_data, x.shape().clone()))
                }
                Op::Conv2d { stride, padding } => {
                    if inputs.len() < 2 || inputs.len() > 3 {
                        return Err(Error::ExecutionError(
                            "Conv2d requires 2 or 3 inputs (x, weight, bias)".to_string(),
                        ));
                    }
                    let x = inputs[0];
                    let weight = inputs[1];
                    let bias = if inputs.len() == 3 {
                        Some(inputs[2])
                    } else {
                        None
                    };

                    let x_dims = x.shape().dims();
                    let w_dims = weight.shape().dims();
                    assert_eq!(x_dims.len(), 4);
                    assert_eq!(w_dims.len(), 4);

                    let batch = x_dims[0];
                    let in_channels = x_dims[1];
                    let in_height = x_dims[2];
                    let in_width = x_dims[3];

                    let out_channels = w_dims[0];
                    let kh = w_dims[2];
                    let kw = w_dims[3];

                    let out_height = (in_height + 2 * padding - kh) / stride + 1;
                    let out_width = (in_width + 2 * padding - kw) / stride + 1;

                    let mut out_data = vec![0.0; batch * out_channels * out_height * out_width];
                    let x_data = x.data();
                    let w_data = weight.data();
                    let b_data = bias.map(|b| b.data());

                    let xs_b = in_channels * in_height * in_width;
                    let xs_c = in_height * in_width;
                    let xs_h = in_width;

                    let ws_oc = in_channels * kh * kw;
                    let ws_ic = kh * kw;
                    let ws_kh = kw;

                    let os_b = out_channels * out_height * out_width;
                    let os_oc = out_height * out_width;
                    let os_h = out_width;

                    for b in 0..batch {
                        for oc in 0..out_channels {
                            let bias_val = b_data.map(|d| d[oc]).unwrap_or(0.0);
                            for oh in 0..out_height {
                                for ow in 0..out_width {
                                    let mut sum = 0.0;
                                    let ih_start = (oh * stride) as isize - *padding as isize;
                                    let iw_start = (ow * stride) as isize - *padding as isize;

                                    for ic in 0..in_channels {
                                        for k_h in 0..kh {
                                            let ih = ih_start + k_h as isize;
                                            if ih >= 0 && ih < in_height as isize {
                                                for k_w in 0..kw {
                                                    let iw = iw_start + k_w as isize;
                                                    if iw >= 0 && iw < in_width as isize {
                                                        let x_idx = b * xs_b
                                                            + ic * xs_c
                                                            + (ih as usize) * xs_h
                                                            + (iw as usize);
                                                        let w_idx = oc * ws_oc
                                                            + ic * ws_ic
                                                            + k_h * ws_kh
                                                            + k_w;
                                                        sum += x_data[x_idx] * w_data[w_idx];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    let out_idx = b * os_b + oc * os_oc + oh * os_h + ow;
                                    out_data[out_idx] = sum + bias_val;
                                }
                            }
                        }
                    }
                    Ok(Tensor::new(
                        out_data,
                        Shape::new(vec![batch, out_channels, out_height, out_width]),
                    ))
                }
                Op::Slice { axis, start, end } => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "Slice requires exactly 1 input".to_string(),
                        ));
                    }
                    let x = inputs[0];
                    let x_dims = x.shape().dims();
                    let ndim = x_dims.len();
                    if *axis >= ndim {
                        return Err(Error::ExecutionError(
                            "Slice axis out of bounds".to_string(),
                        ));
                    }
                    if *start > *end || *end > x_dims[*axis] {
                        return Err(Error::ExecutionError("Invalid slice range".to_string()));
                    }

                    let mut out_dims = x_dims.to_vec();
                    out_dims[*axis] = end - start;
                    let out_shape = Shape::new(out_dims);

                    let num_elements = out_shape.num_elements();
                    let mut out_data = vec![0.0; num_elements];

                    let _out_strides = get_strides(out_shape.dims());
                    let x_strides = get_strides(x_dims);

                    for (idx, out_val) in out_data.iter_mut().enumerate() {
                        let mut temp = idx;
                        let mut coords = vec![0; ndim];
                        for i in (0..ndim).rev() {
                            coords[i] = temp % out_shape.dims()[i];
                            temp /= out_shape.dims()[i];
                        }

                        let mut x_coords = coords.clone();
                        x_coords[*axis] += start;

                        let mut x_idx = 0;
                        for i in 0..ndim {
                            x_idx += x_coords[i] * x_strides[i];
                        }
                        *out_val = x.data()[x_idx];
                    }
                    Ok(Tensor::new(out_data, out_shape))
                }
                Op::SliceGrad {
                    axis,
                    start,
                    end: _,
                } => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "SliceGrad requires exactly 2 inputs: x, dy".to_string(),
                        ));
                    }
                    let x = inputs[0];
                    let dy = inputs[1];
                    let x_dims = x.shape().dims();
                    let ndim = x_dims.len();

                    let mut out_data = vec![0.0; x.data().len()];
                    let dy_shape = dy.shape();
                    let dy_num_elements = dy_shape.num_elements();

                    let _dy_strides = get_strides(dy_shape.dims());
                    let x_strides = get_strides(x_dims);

                    for idx in 0..dy_num_elements {
                        let mut temp = idx;
                        let mut coords = vec![0; ndim];
                        for i in (0..ndim).rev() {
                            coords[i] = temp % dy_shape.dims()[i];
                            temp /= dy_shape.dims()[i];
                        }

                        let mut x_coords = coords.clone();
                        x_coords[*axis] += start;

                        let mut x_idx = 0;
                        for i in 0..ndim {
                            x_idx += x_coords[i] * x_strides[i];
                        }
                        out_data[x_idx] = dy.data()[idx];
                    }
                    Ok(Tensor::new(out_data, x.shape().clone()))
                }
                Op::LayerNormGradX { epsilon } => {
                    if inputs.len() != 3 {
                        return Err(Error::ExecutionError(
                            "LayerNormGradX requires exactly 3 inputs: x, weight, dy".to_string(),
                        ));
                    }
                    let x = inputs[0];
                    let weight = inputs[1];
                    let dy = inputs[2];

                    let dims = x.shape().dims();
                    let last_dim = dims[dims.len() - 1];
                    let rows = x.data().len() / last_dim;

                    let x_data = x.data();
                    let w_data = weight.data();
                    let dy_data = dy.data();

                    let mut dx_data = vec![0.0; x_data.len()];
                    for r in 0..rows {
                        let start = r * last_dim;
                        let end = start + last_dim;
                        let x_row = &x_data[start..end];
                        let dy_row = &dy_data[start..end];

                        let mean = x_row.iter().sum::<f32>() / last_dim as f32;
                        let var = x_row.iter().map(|&val| (val - mean).powi(2)).sum::<f32>()
                            / last_dim as f32;
                        let std = (var + epsilon).sqrt();

                        let mut sum_dy_w = 0.0;
                        let mut sum_dy_w_xhat = 0.0;
                        for c in 0..last_dim {
                            let x_hat = (x_row[c] - mean) / std;
                            let dy_w = dy_row[c] * w_data[c];
                            sum_dy_w += dy_w;
                            sum_dy_w_xhat += dy_w * x_hat;
                        }

                        for c in 0..last_dim {
                            let idx = start + c;
                            let x_hat = (x_row[c] - mean) / std;
                            dx_data[idx] = (dy_data[idx] * w_data[c]
                                - sum_dy_w / last_dim as f32
                                - x_hat * sum_dy_w_xhat / last_dim as f32)
                                / std;
                        }
                    }
                    Ok(Tensor::new(dx_data, x.shape().clone()))
                }
                Op::LayerNormGradW { epsilon } => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "LayerNormGradW requires exactly 2 inputs: x, dy".to_string(),
                        ));
                    }
                    let x = inputs[0];
                    let dy = inputs[1];

                    let dims = x.shape().dims();
                    let last_dim = dims[dims.len() - 1];
                    let rows = x.data().len() / last_dim;

                    let x_data = x.data();
                    let dy_data = dy.data();

                    let mut dw_data = vec![0.0; last_dim];
                    for r in 0..rows {
                        let start = r * last_dim;
                        let x_row = &x_data[start..(start + last_dim)];

                        let mean = x_row.iter().sum::<f32>() / last_dim as f32;
                        let var = x_row.iter().map(|&val| (val - mean).powi(2)).sum::<f32>()
                            / last_dim as f32;
                        let std = (var + epsilon).sqrt();

                        for c in 0..last_dim {
                            let x_hat = (x_row[c] - mean) / std;
                            dw_data[c] += dy_data[start + c] * x_hat;
                        }
                    }
                    Ok(Tensor::new(dw_data, Shape::new(vec![last_dim])))
                }
                Op::LayerNormGradB => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "LayerNormGradB requires exactly 1 input: dy".to_string(),
                        ));
                    }
                    let dy = inputs[0];
                    let dims = dy.shape().dims();
                    let last_dim = dims[dims.len() - 1];
                    let rows = dy.data().len() / last_dim;
                    let dy_data = dy.data();

                    let mut db_data = vec![0.0; last_dim];
                    for r in 0..rows {
                        let start = r * last_dim;
                        for c in 0..last_dim {
                            db_data[c] += dy_data[start + c];
                        }
                    }
                    Ok(Tensor::new(db_data, Shape::new(vec![last_dim])))
                }
                Op::Conv2dGradX { stride, padding } => {
                    if inputs.len() != 3 {
                        return Err(Error::ExecutionError(
                            "Conv2dGradX requires exactly 3 inputs: x, weight, dy".to_string(),
                        ));
                    }
                    let x = inputs[0];
                    let weight = inputs[1];
                    let dy = inputs[2];

                    let x_dims = x.shape().dims();
                    let w_dims = weight.shape().dims();
                    let dy_dims = dy.shape().dims();

                    let batch = x_dims[0];
                    let in_channels = x_dims[1];
                    let in_height = x_dims[2];
                    let in_width = x_dims[3];

                    let out_channels = w_dims[0];
                    let kh = w_dims[2];
                    let kw = w_dims[3];

                    let out_height = dy_dims[2];
                    let out_width = dy_dims[3];

                    let mut dx_data = vec![0.0; x.data().len()];
                    let dy_data = dy.data();
                    let w_data = weight.data();

                    let xs_b = in_channels * in_height * in_width;
                    let xs_c = in_height * in_width;
                    let xs_h = in_width;

                    let ws_oc = in_channels * kh * kw;
                    let ws_ic = kh * kw;
                    let ws_kh = kw;

                    let os_b = out_channels * out_height * out_width;
                    let os_oc = out_height * out_width;
                    let os_h = out_width;

                    for b in 0..batch {
                        for oc in 0..out_channels {
                            for oh in 0..out_height {
                                for ow in 0..out_width {
                                    let dy_val = dy_data[b * os_b + oc * os_oc + oh * os_h + ow];
                                    let ih_start = (oh * stride) as isize - *padding as isize;
                                    let iw_start = (ow * stride) as isize - *padding as isize;

                                    for ic in 0..in_channels {
                                        for k_h in 0..kh {
                                            let ih = ih_start + k_h as isize;
                                            if ih >= 0 && ih < in_height as isize {
                                                for k_w in 0..kw {
                                                    let iw = iw_start + k_w as isize;
                                                    if iw >= 0 && iw < in_width as isize {
                                                        let x_idx = b * xs_b
                                                            + ic * xs_c
                                                            + (ih as usize) * xs_h
                                                            + (iw as usize);
                                                        let w_idx = oc * ws_oc
                                                            + ic * ws_ic
                                                            + k_h * ws_kh
                                                            + k_w;
                                                        dx_data[x_idx] += dy_val * w_data[w_idx];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Ok(Tensor::new(dx_data, x.shape().clone()))
                }
                Op::Conv2dGradW { stride, padding } => {
                    if inputs.len() != 3 {
                        return Err(Error::ExecutionError(
                            "Conv2dGradW requires exactly 3 inputs: x, weight, dy".to_string(),
                        ));
                    }
                    let x = inputs[0];
                    let weight = inputs[1];
                    let dy = inputs[2];

                    let x_dims = x.shape().dims();
                    let w_dims = weight.shape().dims();
                    let dy_dims = dy.shape().dims();

                    let batch = x_dims[0];
                    let in_channels = x_dims[1];
                    let in_height = x_dims[2];
                    let in_width = x_dims[3];

                    let out_channels = w_dims[0];
                    let kh = w_dims[2];
                    let kw = w_dims[3];

                    let out_height = dy_dims[2];
                    let out_width = dy_dims[3];

                    let mut dw_data = vec![0.0; weight.data().len()];
                    let x_data = x.data();
                    let dy_data = dy.data();

                    let xs_b = in_channels * in_height * in_width;
                    let xs_c = in_height * in_width;
                    let xs_h = in_width;

                    let ws_oc = in_channels * kh * kw;
                    let ws_ic = kh * kw;
                    let ws_kh = kw;

                    let os_b = out_channels * out_height * out_width;
                    let os_oc = out_height * out_width;
                    let os_h = out_width;

                    for b in 0..batch {
                        for oc in 0..out_channels {
                            for oh in 0..out_height {
                                for ow in 0..out_width {
                                    let dy_val = dy_data[b * os_b + oc * os_oc + oh * os_h + ow];
                                    let ih_start = (oh * stride) as isize - *padding as isize;
                                    let iw_start = (ow * stride) as isize - *padding as isize;

                                    for ic in 0..in_channels {
                                        for k_h in 0..kh {
                                            let ih = ih_start + k_h as isize;
                                            if ih >= 0 && ih < in_height as isize {
                                                for k_w in 0..kw {
                                                    let iw = iw_start + k_w as isize;
                                                    if iw >= 0 && iw < in_width as isize {
                                                        let x_idx = b * xs_b
                                                            + ic * xs_c
                                                            + (ih as usize) * xs_h
                                                            + (iw as usize);
                                                        let w_idx = oc * ws_oc
                                                            + ic * ws_ic
                                                            + k_h * ws_kh
                                                            + k_w;
                                                        dw_data[w_idx] += x_data[x_idx] * dy_val;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Ok(Tensor::new(dw_data, weight.shape().clone()))
                }
                Op::Conv2dGradB => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "Conv2dGradB requires exactly 1 input: dy".to_string(),
                        ));
                    }
                    let dy = inputs[0];
                    let dy_dims = dy.shape().dims();
                    let batch = dy_dims[0];
                    let out_channels = dy_dims[1];
                    let out_height = dy_dims[2];
                    let out_width = dy_dims[3];
                    let dy_data = dy.data();

                    let mut db_data = vec![0.0; out_channels];
                    let os_b = out_channels * out_height * out_width;
                    let os_oc = out_height * out_width;

                    for b in 0..batch {
                        for oc in 0..out_channels {
                            for oh in 0..out_height {
                                for ow in 0..out_width {
                                    db_data[oc] +=
                                        dy_data[b * os_b + oc * os_oc + oh * out_width + ow];
                                }
                            }
                        }
                    }
                    Ok(Tensor::new(db_data, Shape::new(vec![out_channels])))
                }
                Op::BatchedMatMul => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "BatchedMatMul requires exactly 2 inputs".to_string(),
                        ));
                    }
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    let lhs_shape = lhs.shape().dims();
                    let rhs_shape = rhs.shape().dims();
                    if lhs_shape.len() != 3 || rhs_shape.len() != 3 {
                        return Err(Error::ExecutionError(
                            "BatchedMatMul inputs must be 3D".to_string(),
                        ));
                    }
                    let b = lhs_shape[0];
                    let m = lhs_shape[1];
                    let k = lhs_shape[2];
                    let n = rhs_shape[2];

                    let mut out_data = Vec::with_capacity(b * m * n);
                    for i in 0..b {
                        let lhs_slice = ndarray::ArrayView2::from_shape(
                            (m, k),
                            &lhs.data()[i * m * k..(i + 1) * m * k],
                        )
                        .map_err(|e| {
                            Error::ExecutionError(format!("BatchedMatMul lhs slice: {}", e))
                        })?;
                        let rhs_slice = ndarray::ArrayView2::from_shape(
                            (k, n),
                            &rhs.data()[i * k * n..(i + 1) * k * n],
                        )
                        .map_err(|e| {
                            Error::ExecutionError(format!("BatchedMatMul rhs slice: {}", e))
                        })?;
                        let out_arr = lhs_slice.dot(&rhs_slice);
                        out_data.extend(out_arr.iter().cloned());
                    }
                    let out_shape = Shape::new(vec![b, m, n]);
                    Ok(Tensor::new(out_data, out_shape))
                }
                Op::BatchedTranspose => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "BatchedTranspose requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let dims = input.shape().dims();
                    if dims.len() != 3 {
                        return Err(Error::ExecutionError(
                            "BatchedTranspose input must be 3D".to_string(),
                        ));
                    }
                    let b = dims[0];
                    let m = dims[1];
                    let n = dims[2];
                    let data = input.data();
                    let mut out_data = vec![0.0; b * m * n];
                    for i in 0..b {
                        let batch_offset = i * m * n;
                        for r in 0..m {
                            for c in 0..n {
                                out_data[batch_offset + c * m + r] = data[batch_offset + r * n + c];
                            }
                        }
                    }
                    Ok(Tensor::new(out_data, Shape::new(vec![b, n, m])))
                }
                Op::MaxPool2d {
                    pool_size,
                    stride,
                    padding,
                } => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "MaxPool2d requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let dims = input.shape().dims();
                    if dims.len() != 4 {
                        return Err(Error::ExecutionError(
                            "MaxPool2d input must be 4D [B, C, H, W]".to_string(),
                        ));
                    }
                    let b = dims[0];
                    let c = dims[1];
                    let h = dims[2];
                    let w = dims[3];
                    let h_out = (h + 2 * padding - pool_size) / stride + 1;
                    let w_out = (w + 2 * padding - pool_size) / stride + 1;

                    let mut out_data = Vec::with_capacity(b * c * h_out * w_out);
                    let data = input.data();

                    for batch in 0..b {
                        for channel in 0..c {
                            let ch_offset = (batch * c + channel) * h * w;
                            for oh in 0..h_out {
                                let ih_start = (oh * stride) as isize - *padding as isize;
                                for ow in 0..w_out {
                                    let iw_start = (ow * stride) as isize - *padding as isize;
                                    let mut max_val = f32::MIN;
                                    for kh in 0..*pool_size {
                                        let ih = ih_start + kh as isize;
                                        for kw in 0..*pool_size {
                                            let iw = iw_start + kw as isize;
                                            if ih >= 0
                                                && ih < h as isize
                                                && iw >= 0
                                                && iw < w as isize
                                            {
                                                let val = data
                                                    [ch_offset + (ih as usize) * w + (iw as usize)];
                                                if val > max_val {
                                                    max_val = val;
                                                }
                                            }
                                        }
                                    }
                                    out_data.push(max_val);
                                }
                            }
                        }
                    }
                    Ok(Tensor::new(out_data, Shape::new(vec![b, c, h_out, w_out])))
                }
                Op::MaxPool2dGrad {
                    pool_size,
                    stride,
                    padding,
                } => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "MaxPool2dGrad requires exactly 2 inputs: dy, x".to_string(),
                        ));
                    }
                    let dy = inputs[0];
                    let x = inputs[1];
                    let dims = x.shape().dims();
                    let b = dims[0];
                    let c = dims[1];
                    let h = dims[2];
                    let w = dims[3];

                    let dy_dims = dy.shape().dims();
                    let h_out = dy_dims[2];
                    let w_out = dy_dims[3];

                    let mut dx_data = vec![0.0; b * c * h * w];
                    let x_data = x.data();
                    let dy_data = dy.data();

                    for batch in 0..b {
                        for channel in 0..c {
                            let ch_offset = (batch * c + channel) * h * w;
                            let dy_offset = (batch * c + channel) * h_out * w_out;
                            for oh in 0..h_out {
                                let ih_start = (oh * stride) as isize - *padding as isize;
                                for ow in 0..w_out {
                                    let iw_start = (ow * stride) as isize - *padding as isize;
                                    let grad = dy_data[dy_offset + oh * w_out + ow];

                                    let mut max_val = f32::MIN;
                                    let mut max_idx = None;

                                    for kh in 0..*pool_size {
                                        let ih = ih_start + kh as isize;
                                        for kw in 0..*pool_size {
                                            let iw = iw_start + kw as isize;
                                            if ih >= 0
                                                && ih < h as isize
                                                && iw >= 0
                                                && iw < w as isize
                                            {
                                                let idx =
                                                    ch_offset + (ih as usize) * w + (iw as usize);
                                                let val = x_data[idx];
                                                if val > max_val {
                                                    max_val = val;
                                                    max_idx = Some(idx);
                                                }
                                            }
                                        }
                                    }
                                    if let Some(idx) = max_idx {
                                        dx_data[idx] += grad;
                                    }
                                }
                            }
                        }
                    }
                    Ok(Tensor::new(dx_data, x.shape().clone()))
                }
                Op::AvgPool2d {
                    pool_size,
                    stride,
                    padding,
                } => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "AvgPool2d requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let dims = input.shape().dims();
                    if dims.len() != 4 {
                        return Err(Error::ExecutionError(
                            "AvgPool2d input must be 4D [B, C, H, W]".to_string(),
                        ));
                    }
                    let b = dims[0];
                    let c = dims[1];
                    let h = dims[2];
                    let w = dims[3];
                    let h_out = (h + 2 * padding - pool_size) / stride + 1;
                    let w_out = (w + 2 * padding - pool_size) / stride + 1;

                    let mut out_data = Vec::with_capacity(b * c * h_out * w_out);
                    let data = input.data();

                    for batch in 0..b {
                        for channel in 0..c {
                            let ch_offset = (batch * c + channel) * h * w;
                            for oh in 0..h_out {
                                let ih_start = (oh * stride) as isize - *padding as isize;
                                for ow in 0..w_out {
                                    let iw_start = (ow * stride) as isize - *padding as isize;
                                    let mut sum = 0.0;
                                    let mut count = 0.0;
                                    for kh in 0..*pool_size {
                                        let ih = ih_start + kh as isize;
                                        for kw in 0..*pool_size {
                                            let iw = iw_start + kw as isize;
                                            if ih >= 0
                                                && ih < h as isize
                                                && iw >= 0
                                                && iw < w as isize
                                            {
                                                sum += data
                                                    [ch_offset + (ih as usize) * w + (iw as usize)];
                                            }
                                            count += 1.0;
                                        }
                                    }
                                    out_data.push(sum / count);
                                }
                            }
                        }
                    }
                    Ok(Tensor::new(out_data, Shape::new(vec![b, c, h_out, w_out])))
                }
                Op::AvgPool2dGrad {
                    pool_size,
                    stride,
                    padding,
                } => {
                    if inputs.len() != 2 {
                        return Err(Error::ExecutionError(
                            "AvgPool2dGrad requires exactly 2 inputs: dy, x".to_string(),
                        ));
                    }
                    let dy = inputs[0];
                    let x = inputs[1];
                    let dims = x.shape().dims();
                    let b = dims[0];
                    let c = dims[1];
                    let h = dims[2];
                    let w = dims[3];

                    let dy_dims = dy.shape().dims();
                    let h_out = dy_dims[2];
                    let w_out = dy_dims[3];

                    let mut dx_data = vec![0.0; b * c * h * w];
                    let dy_data = dy.data();

                    for batch in 0..b {
                        for channel in 0..c {
                            let ch_offset = (batch * c + channel) * h * w;
                            let dy_offset = (batch * c + channel) * h_out * w_out;
                            for oh in 0..h_out {
                                let ih_start = (oh * stride) as isize - *padding as isize;
                                for ow in 0..w_out {
                                    let iw_start = (ow * stride) as isize - *padding as isize;
                                    let grad = dy_data[dy_offset + oh * w_out + ow];

                                    let count = (*pool_size * *pool_size) as f32;
                                    let val = grad / count;

                                    for kh in 0..*pool_size {
                                        let ih = ih_start + kh as isize;
                                        for kw in 0..*pool_size {
                                            let iw = iw_start + kw as isize;
                                            if ih >= 0
                                                && ih < h as isize
                                                && iw >= 0
                                                && iw < w as isize
                                            {
                                                let idx =
                                                    ch_offset + (ih as usize) * w + (iw as usize);
                                                dx_data[idx] += val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Ok(Tensor::new(dx_data, x.shape().clone()))
                }
                Op::Attention { scale } => {
                    if inputs.len() != 3 {
                        return Err(Error::ExecutionError(
                            "Attention requires exactly 3 inputs: Q, K, V".to_string(),
                        ));
                    }
                    let q = inputs[0];
                    let k = inputs[1];
                    let v = inputs[2];

                    let q_dims = q.shape().dims();
                    let k_dims = k.shape().dims();
                    let v_dims = v.shape().dims();

                    if q_dims.len() != 3 || k_dims.len() != 3 || v_dims.len() != 3 {
                        return Err(Error::ExecutionError(
                            "Attention inputs must be 3D [B, S, D]".to_string(),
                        ));
                    }

                    let b = q_dims[0];
                    let s = q_dims[1];
                    let d = q_dims[2];

                    let mut out_data = Vec::with_capacity(b * s * d);

                    for i in 0..b {
                        let q_slice = ndarray::ArrayView2::from_shape(
                            (s, d),
                            &q.data()[i * s * d..(i + 1) * s * d],
                        )
                        .map_err(|e| Error::ExecutionError(format!("Attention q slice: {}", e)))?;
                        let k_slice = ndarray::ArrayView2::from_shape(
                            (s, d),
                            &k.data()[i * s * d..(i + 1) * s * d],
                        )
                        .map_err(|e| Error::ExecutionError(format!("Attention k slice: {}", e)))?;
                        let v_slice = ndarray::ArrayView2::from_shape(
                            (s, d),
                            &v.data()[i * s * d..(i + 1) * s * d],
                        )
                        .map_err(|e| Error::ExecutionError(format!("Attention v slice: {}", e)))?;

                        let mut a_b = q_slice.dot(&k_slice.t());
                        a_b.mapv_inplace(|x| x * scale);

                        let mut s_b = ndarray::Array2::<f32>::zeros((s, s));
                        for r in 0..s {
                            let mut row = a_b.row(r).to_owned();
                            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
                            row.mapv_inplace(|x| (x - max_val).exp() / sum);
                            s_b.row_mut(r).assign(&row);
                        }

                        let o_b = s_b.dot(&v_slice);
                        out_data.extend(o_b.iter().cloned());
                    }
                    Ok(Tensor::new(out_data, q.shape().clone()))
                }
                Op::AttentionGradQ { scale } => {
                    let dy = inputs[0];
                    let q = inputs[1];
                    let k = inputs[2];
                    let v = inputs[3];

                    let dims = q.shape().dims();
                    let b = dims[0];
                    let s = dims[1];
                    let d = dims[2];

                    let mut dq_data = Vec::with_capacity(b * s * d);

                    for i in 0..b {
                        let dy_slice = ndarray::ArrayView2::from_shape(
                            (s, d),
                            &dy.data()[i * s * d..(i + 1) * s * d],
                        )
                        .map_err(|e| {
                            Error::ExecutionError(format!("AttentionGradQ dy slice: {}", e))
                        })?;
                        let q_slice = ndarray::ArrayView2::from_shape(
                            (s, d),
                            &q.data()[i * s * d..(i + 1) * s * d],
                        )
                        .map_err(|e| {
                            Error::ExecutionError(format!("AttentionGradQ q slice: {}", e))
                        })?;
                        let k_slice = ndarray::ArrayView2::from_shape(
                            (s, d),
                            &k.data()[i * s * d..(i + 1) * s * d],
                        )
                        .map_err(|e| {
                            Error::ExecutionError(format!("AttentionGradQ k slice: {}", e))
                        })?;
                        let v_slice = ndarray::ArrayView2::from_shape(
                            (s, d),
                            &v.data()[i * s * d..(i + 1) * s * d],
                        )
                        .map_err(|e| {
                            Error::ExecutionError(format!("AttentionGradQ v slice: {}", e))
                        })?;

                        let mut a_b = q_slice.dot(&k_slice.t());
                        a_b.mapv_inplace(|x| x * scale);

                        let mut s_b = ndarray::Array2::<f32>::zeros((s, s));
                        for r in 0..s {
                            let mut row = a_b.row(r).to_owned();
                            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
                            row.mapv_inplace(|x| (x - max_val).exp() / sum);
                            s_b.row_mut(r).assign(&row);
                        }

                        let ds_b = dy_slice.dot(&v_slice.t());

                        let mut da_b = ndarray::Array2::<f32>::zeros((s, s));
                        for r in 0..s {
                            let mut sum = 0.0;
                            for c in 0..s {
                                sum += ds_b[(r, c)] * s_b[(r, c)];
                            }
                            for c in 0..s {
                                da_b[(r, c)] = s_b[(r, c)] * (ds_b[(r, c)] - sum) * scale;
                            }
                        }

                        let dq_b = da_b.dot(&k_slice);
                        dq_data.extend(dq_b.iter().cloned());
                    }
                    Ok(Tensor::new(dq_data, q.shape().clone()))
                }
                Op::AttentionGradK { scale } => {
                    let dy = inputs[0];
                    let q = inputs[1];
                    let k = inputs[2];
                    let v = inputs[3];

                    let dims = q.shape().dims();
                    let b = dims[0];
                    let s = dims[1];
                    let d = dims[2];

                    let mut dk_data = Vec::with_capacity(b * s * d);

                    for i in 0..b {
                        let dy_slice = ndarray::ArrayView2::from_shape(
                            (s, d),
                            &dy.data()[i * s * d..(i + 1) * s * d],
                        )
                        .map_err(|e| {
                            Error::ExecutionError(format!("AttentionGradK dy slice: {}", e))
                        })?;
                        let q_slice = ndarray::ArrayView2::from_shape(
                            (s, d),
                            &q.data()[i * s * d..(i + 1) * s * d],
                        )
                        .map_err(|e| {
                            Error::ExecutionError(format!("AttentionGradK q slice: {}", e))
                        })?;
                        let k_slice = ndarray::ArrayView2::from_shape(
                            (s, d),
                            &k.data()[i * s * d..(i + 1) * s * d],
                        )
                        .map_err(|e| {
                            Error::ExecutionError(format!("AttentionGradK k slice: {}", e))
                        })?;
                        let v_slice = ndarray::ArrayView2::from_shape(
                            (s, d),
                            &v.data()[i * s * d..(i + 1) * s * d],
                        )
                        .map_err(|e| {
                            Error::ExecutionError(format!("AttentionGradK v slice: {}", e))
                        })?;

                        let mut a_b = q_slice.dot(&k_slice.t());
                        a_b.mapv_inplace(|x| x * scale);

                        let mut s_b = ndarray::Array2::<f32>::zeros((s, s));
                        for r in 0..s {
                            let mut row = a_b.row(r).to_owned();
                            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
                            row.mapv_inplace(|x| (x - max_val).exp() / sum);
                            s_b.row_mut(r).assign(&row);
                        }

                        let ds_b = dy_slice.dot(&v_slice.t());

                        let mut da_b = ndarray::Array2::<f32>::zeros((s, s));
                        for r in 0..s {
                            let mut sum = 0.0;
                            for c in 0..s {
                                sum += ds_b[(r, c)] * s_b[(r, c)];
                            }
                            for c in 0..s {
                                da_b[(r, c)] = s_b[(r, c)] * (ds_b[(r, c)] - sum) * scale;
                            }
                        }

                        let dk_b = da_b.t().dot(&q_slice);
                        dk_data.extend(dk_b.iter().cloned());
                    }
                    Ok(Tensor::new(dk_data, k.shape().clone()))
                }
                Op::AttentionGradV { scale } => {
                    let dy = inputs[0];
                    let q = inputs[1];
                    let k = inputs[2];
                    let v = inputs[3];

                    let dims = q.shape().dims();
                    let b = dims[0];
                    let s = dims[1];
                    let d = dims[2];

                    let mut dv_data = Vec::with_capacity(b * s * d);

                    for i in 0..b {
                        let dy_slice = ndarray::ArrayView2::from_shape(
                            (s, d),
                            &dy.data()[i * s * d..(i + 1) * s * d],
                        )
                        .map_err(|e| {
                            Error::ExecutionError(format!("AttentionGradV dy slice: {}", e))
                        })?;
                        let q_slice = ndarray::ArrayView2::from_shape(
                            (s, d),
                            &q.data()[i * s * d..(i + 1) * s * d],
                        )
                        .map_err(|e| {
                            Error::ExecutionError(format!("AttentionGradV q slice: {}", e))
                        })?;
                        let k_slice = ndarray::ArrayView2::from_shape(
                            (s, d),
                            &k.data()[i * s * d..(i + 1) * s * d],
                        )
                        .map_err(|e| {
                            Error::ExecutionError(format!("AttentionGradV k slice: {}", e))
                        })?;

                        let mut a_b = q_slice.dot(&k_slice.t());
                        a_b.mapv_inplace(|x| x * scale);

                        let mut s_b = ndarray::Array2::<f32>::zeros((s, s));
                        for r in 0..s {
                            let mut row = a_b.row(r).to_owned();
                            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
                            row.mapv_inplace(|x| (x - max_val).exp() / sum);
                            s_b.row_mut(r).assign(&row);
                        }

                        let dv_b = s_b.t().dot(&dy_slice);
                        dv_data.extend(dv_b.iter().cloned());
                    }
                    Ok(Tensor::new(dv_data, v.shape().clone()))
                }
                Op::CastF32ToF16 => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "CastF32ToF16 requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let f16_data: Vec<half::f16> = input
                        .data()
                        .iter()
                        .map(|&x| half::f16::from_f32(x))
                        .collect();
                    Ok(Tensor::new_with_data(
                        crate::tensor::AnyData::F16(f16_data),
                        input.shape().clone(),
                        crate::tensor::Dtype::F16,
                    ))
                }
                Op::CastF16ToF32 => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "CastF16ToF32 requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let f16_data = match input.data_raw() {
                        crate::tensor::AnyData::F16(v) => v,
                        _ => {
                            return Err(Error::ExecutionError(
                                "CastF16ToF32 input must be F16 data".to_string(),
                            ))
                        }
                    };
                    let f32_data: Vec<f32> = f16_data.iter().map(|&x| f32::from(x)).collect();
                    Ok(Tensor::new(f32_data, input.shape().clone()))
                }
                Op::CastF32ToBF16 => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "CastF32ToBF16 requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    // BF16 via half crate: convert f32 -> f32 bits, truncate lower 16 bits
                    let bf16_data: Vec<half::bf16> = input
                        .data()
                        .iter()
                        .map(|&x| half::bf16::from_f32(x))
                        .collect();
                    Ok(Tensor::new_with_data(
                        crate::tensor::AnyData::F16(
                            bf16_data
                                .iter()
                                .map(|&x| half::f16::from_f32(f32::from(x)))
                                .collect(),
                        ),
                        input.shape().clone(),
                        crate::tensor::Dtype::BF16,
                    ))
                }
                Op::CastBF16ToF32 => {
                    if inputs.len() != 1 {
                        return Err(Error::ExecutionError(
                            "CastBF16ToF32 requires exactly 1 input".to_string(),
                        ));
                    }
                    let input = inputs[0];
                    let f32_data = match input.data_raw() {
                        crate::tensor::AnyData::F32(v) => v.clone(),
                        crate::tensor::AnyData::F16(v) => v
                            .iter()
                            .map(|&x| f32::from(half::bf16::from_f32(f32::from(x))))
                            .collect(),
                    };
                    Ok(Tensor::new(f32_data, input.shape().clone()))
                }
                Op::CausalAttention { scale, num_heads } => {
                    if inputs.len() != 3 {
                        return Err(Error::ExecutionError(
                            "CausalAttention requires exactly 3 inputs: Q, K, V".to_string(),
                        ));
                    }
                    let q = inputs[0];
                    let k = inputs[1];
                    let v = inputs[2];
                    let q_dims = q.shape().dims();
                    let k_dims = k.shape().dims();
                    let v_dims = v.shape().dims();
                    if q_dims.len() != 3 || k_dims.len() != 3 || v_dims.len() != 3 {
                        return Err(Error::ExecutionError(
                            "CausalAttention Q, K, V must be 3D [B, S, D]".to_string(),
                        ));
                    }
                    let batch = q_dims[0];
                    let seq_len = q_dims[1];
                    let d = q_dims[2];
                    let head_dim = d / num_heads;

                    // Compute Q * K^T * scale, apply causal mask, softmax, multiply by V
                    let q_data = q.data();
                    let k_data = k.data();
                    let v_data = v.data();
                    let mut out_data = vec![0.0; q_data.len()];

                    for b in 0..batch {
                        for h in 0..*num_heads {
                            for i in 0..seq_len {
                                let mut score_row = vec![f32::NEG_INFINITY; seq_len];
                                for (j, score) in score_row.iter_mut().enumerate() {
                                    if j > i {
                                        continue;
                                    }
                                    let mut dot = 0.0;
                                    for d_ in 0..head_dim {
                                        let q_idx = b * seq_len * d + i * d + h * head_dim + d_;
                                        let k_idx = b * seq_len * d + j * d + h * head_dim + d_;
                                        dot += q_data[q_idx] * k_data[k_idx];
                                    }
                                    *score = dot * scale;
                                }
                                // softmax
                                let max_val =
                                    score_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                                let mut sum = 0.0;
                                for score in score_row.iter_mut() {
                                    if *score != f32::NEG_INFINITY {
                                        *score = (*score - max_val).exp();
                                        sum += *score;
                                    }
                                }
                                // weighted sum of V
                                for d_ in 0..head_dim {
                                    let mut weighted = 0.0;
                                    for (j, score) in score_row.iter().enumerate() {
                                        if *score != f32::NEG_INFINITY {
                                            let v_idx = b * seq_len * d + j * d + h * head_dim + d_;
                                            weighted += (*score / sum) * v_data[v_idx];
                                        }
                                    }
                                    let out_idx = b * seq_len * d + i * d + h * head_dim + d_;
                                    out_data[out_idx] = weighted;
                                }
                            }
                        }
                    }
                    Ok(Tensor::new(out_data, q.shape().clone()))
                }
                Op::MultiHeadAttention { scale, num_heads } => {
                    if inputs.len() != 3 {
                        return Err(Error::ExecutionError(
                            "MultiHeadAttention requires exactly 3 inputs: Q, K, V".to_string(),
                        ));
                    }
                    let q = inputs[0];
                    let k = inputs[1];
                    let v = inputs[2];
                    let q_dims = q.shape().dims();
                    if q_dims.len() != 3 {
                        return Err(Error::ExecutionError(
                            "MultiHeadAttention Q must be 3D [B, S, D]".to_string(),
                        ));
                    }
                    let batch = q_dims[0];
                    let seq_len = q_dims[1];
                    let d = q_dims[2];
                    let head_dim = d / num_heads;

                    let q_data = q.data();
                    let k_data = k.data();
                    let v_data = v.data();
                    let mut out_data = vec![0.0; q_data.len()];

                    for b in 0..batch {
                        for h in 0..*num_heads {
                            let mut scores = vec![vec![0.0; seq_len]; seq_len];
                            for (i, row) in scores.iter_mut().enumerate() {
                                for (j, score) in row.iter_mut().enumerate() {
                                    let mut dot = 0.0;
                                    for d_ in 0..head_dim {
                                        let q_idx = b * seq_len * d + i * d + h * head_dim + d_;
                                        let k_idx = b * seq_len * d + j * d + h * head_dim + d_;
                                        dot += q_data[q_idx] * k_data[k_idx];
                                    }
                                    *score = dot * scale;
                                }
                            }
                            // Softmax over each row
                            for (i, row) in scores.iter_mut().enumerate() {
                                let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                                let mut sum = 0.0;
                                for score in row.iter_mut() {
                                    *score = (*score - max_val).exp();
                                    sum += *score;
                                }
                                for d_ in 0..head_dim {
                                    let mut weighted = 0.0;
                                    for (j, score) in row.iter().enumerate() {
                                        let v_idx = b * seq_len * d + j * d + h * head_dim + d_;
                                        weighted += (*score / sum) * v_data[v_idx];
                                    }
                                    let out_idx = b * seq_len * d + i * d + h * head_dim + d_;
                                    out_data[out_idx] = weighted;
                                }
                            }
                        }
                    }
                    Ok(Tensor::new(out_data, q.shape().clone()))
                }
                Op::FlashAttention { scale, causal } => {
                    if inputs.len() != 3 {
                        return Err(Error::ExecutionError(
                            "FlashAttention requires exactly 3 inputs: Q, K, V".to_string(),
                        ));
                    }
                    let q = inputs[0];
                    let k = inputs[1];
                    let v = inputs[2];
                    let q_dims = q.shape().dims();
                    if q_dims.len() != 3 {
                        return Err(Error::ExecutionError(
                            "FlashAttention Q must be 3D [B, S, D]".to_string(),
                        ));
                    }
                    let batch = q_dims[0];
                    let seq_len = q_dims[1];
                    let d = q_dims[2];

                    // Flash Attention v2: tiled online softmax with BLAS matmuls
                    let tile_size: usize = 32.min(seq_len);
                    let q_data = q.data();
                    let k_data = k.data();
                    let v_data = v.data();
                    let mut out_data = vec![0.0; q_data.len()];

                    for b in 0..batch {
                        let out_start = b * seq_len * d;
                        let out_slice = &mut out_data[out_start..out_start + seq_len * d];

                        // Process Q in tiles
                        for q_tile_start in (0..seq_len).step_by(tile_size) {
                            let q_tile_end = (q_tile_start + tile_size).min(seq_len);
                            let q_tile_size = q_tile_end - q_tile_start;

                            // Online softmax stats for each query in tile
                            let mut m = vec![f32::NEG_INFINITY; q_tile_size];
                            let mut l = vec![0.0f32; q_tile_size];
                            let mut acc = vec![0.0f32; q_tile_size * d];

                            // Iterate over K,V tiles
                            for kv_tile_start in (0..seq_len).step_by(tile_size) {
                                let kv_tile_end = (kv_tile_start + tile_size).min(seq_len);
                                let kv_tile_size = kv_tile_end - kv_tile_start;

                                // --- BLAS S = Q_tile · K_tile^T ---
                                // Q_tile: [q_tile_size × d], K_tile: [kv_tile_size × d]
                                // S = Q_tile · K_tile^T: [q_tile_size × kv_tile_size]
                                let mut s = vec![0.0f32; q_tile_size * kv_tile_size];
                                let q_offset = b * seq_len * d + q_tile_start * d;
                                let k_offset = b * seq_len * d + kv_tile_start * d;

                                unsafe {
                                    crate::blas::sgemm(
                                        false,
                                        true,
                                        q_tile_size,
                                        kv_tile_size,
                                        d,
                                        *scale,
                                        q_data.as_ptr().add(q_offset),
                                        d,
                                        k_data.as_ptr().add(k_offset),
                                        d,
                                        0.0,
                                        s.as_mut_ptr(),
                                        kv_tile_size,
                                    );
                                }

                                // Apply causal mask: set s[i][j] = -inf for j > i
                                if *causal && q_tile_size > 0 {
                                    for i in 0..q_tile_size {
                                        let qi = q_tile_start + i;
                                        for j in 0..kv_tile_size {
                                            let kj = kv_tile_start + j;
                                            if kj > qi {
                                                s[i * kv_tile_size + j] = f32::NEG_INFINITY;
                                            }
                                        }
                                    }
                                }

                                // Online softmax update with BLAS P·V
                                for i in 0..q_tile_size {
                                    let m_old = m[i];
                                    let mut m_new = m_old;
                                    for j in 0..kv_tile_size {
                                        let sv = s[i * kv_tile_size + j];
                                        if sv != f32::NEG_INFINITY {
                                            m_new = m_new.max(sv);
                                        }
                                    }

                                    if m_new != m_old {
                                        let rescale = (m_old - m_new).exp();
                                        l[i] *= rescale;
                                        for d_ in 0..d {
                                            acc[i * d + d_] *= rescale;
                                        }
                                    }

                                    // Compute P = exp(S - m_new) and accumulate l
                                    for j in 0..kv_tile_size {
                                        let sv = s[i * kv_tile_size + j];
                                        if sv != f32::NEG_INFINITY {
                                            let p = (sv - m_new).exp();
                                            l[i] += p;
                                            s[i * kv_tile_size + j] = p;
                                        } else {
                                            s[i * kv_tile_size + j] = 0.0;
                                        }
                                    }

                                    m[i] = m_new;
                                }

                                // --- BLAS P · V_tile ---
                                // P: [q_tile_size × kv_tile_size], V_tile: [kv_tile_size × d]
                                // result accumulates into acc: [q_tile_size × d]
                                let v_offset = b * seq_len * d + kv_tile_start * d;
                                unsafe {
                                    crate::blas::sgemm(
                                        false,
                                        false,
                                        q_tile_size,
                                        d,
                                        kv_tile_size,
                                        1.0,
                                        s.as_ptr(),
                                        kv_tile_size,
                                        v_data.as_ptr().add(v_offset),
                                        d,
                                        1.0,
                                        acc.as_mut_ptr(),
                                        d,
                                    );
                                }
                            }

                            // Write output: acc / l
                            for i in 0..q_tile_size {
                                let qi = q_tile_start + i;
                                let inv_l = if l[i] > 0.0 { 1.0 / l[i] } else { 0.0 };
                                for d_ in 0..d {
                                    out_slice[qi * d + d_] = acc[i * d + d_] * inv_l;
                                }
                            }
                        }
                    }
                    Ok(Tensor::new(out_data, q.shape().clone()))
                }
            }
        }
    }

    impl CpuBackend {
        pub fn execute_slices(
            &self,
            op: &Op,
            inputs: &[&[f32]],
            input_shapes: &[Shape],
            output: &mut [f32],
            output_shape: &Shape,
        ) -> Result<(), Error> {
            match op {
                Op::Input(_) => Ok(()),
                Op::MatMul => {
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    let lhs_shape = &input_shapes[0];
                    let rhs_shape = &input_shapes[1];
                    let m = lhs_shape.dims()[0];
                    let k = lhs_shape.dims()[1];
                    let n = rhs_shape.dims()[1];
                    unsafe {
                        crate::blas::sgemm(
                            false,
                            false,
                            m,
                            n,
                            k,
                            1.0,
                            lhs.as_ptr(),
                            k,
                            rhs.as_ptr(),
                            n,
                            0.0,
                            output.as_mut_ptr(),
                            n,
                        );
                    }
                    Ok(())
                }
                Op::Relu => {
                    let input = inputs[0];
                    crate::parallel::parallel_map_inplace(input, output, |x| {
                        if x > 0.0 {
                            x
                        } else {
                            0.0
                        }
                    });
                    Ok(())
                }
                Op::Add => {
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    crate::parallel::parallel_map2_inplace(lhs, rhs, output, |a, b| a + b);
                    Ok(())
                }
                Op::Sub => {
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    crate::parallel::parallel_map2_inplace(lhs, rhs, output, |a, b| a - b);
                    Ok(())
                }
                Op::Mul => {
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    crate::parallel::parallel_map2_inplace(lhs, rhs, output, |a, b| a * b);
                    Ok(())
                }
                Op::Div => {
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    crate::parallel::parallel_map2_inplace(lhs, rhs, output, |a, b| a / b);
                    Ok(())
                }
                Op::Tanh => {
                    let input = inputs[0];
                    crate::parallel::parallel_map_inplace(input, output, |x| x.tanh());
                    Ok(())
                }
                Op::Sigmoid => {
                    let input = inputs[0];
                    crate::parallel::parallel_map_inplace(input, output, |x| {
                        1.0 / (1.0 + (-x).exp())
                    });
                    Ok(())
                }
                Op::Exp => {
                    let input = inputs[0];
                    crate::parallel::parallel_map_inplace(input, output, |x| x.exp());
                    Ok(())
                }
                Op::Sqrt => {
                    let input = inputs[0];
                    crate::parallel::parallel_map_inplace(input, output, |x| x.sqrt());
                    Ok(())
                }
                Op::Neg => {
                    let input = inputs[0];
                    crate::parallel::parallel_map_inplace(input, output, |x| -x);
                    Ok(())
                }
                Op::BroadcastAdd { .. } => {
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    broadcast_op_slices(
                        lhs,
                        &input_shapes[0],
                        rhs,
                        &input_shapes[1],
                        output,
                        output_shape,
                        |a, b| a + b,
                    );
                    Ok(())
                }
                Op::BroadcastMul { .. } => {
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    broadcast_op_slices(
                        lhs,
                        &input_shapes[0],
                        rhs,
                        &input_shapes[1],
                        output,
                        output_shape,
                        |a, b| a * b,
                    );
                    Ok(())
                }
                Op::BroadcastSub { .. } => {
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    broadcast_op_slices(
                        lhs,
                        &input_shapes[0],
                        rhs,
                        &input_shapes[1],
                        output,
                        output_shape,
                        |a, b| a - b,
                    );
                    Ok(())
                }
                Op::BroadcastDiv { .. } => {
                    let lhs = inputs[0];
                    let rhs = inputs[1];
                    broadcast_op_slices(
                        lhs,
                        &input_shapes[0],
                        rhs,
                        &input_shapes[1],
                        output,
                        output_shape,
                        |a, b| a / b,
                    );
                    Ok(())
                }
                _ => {
                    let input_tensors: Vec<Tensor> = inputs
                        .iter()
                        .zip(input_shapes.iter())
                        .map(|(slice, shape)| Tensor::new(slice.to_vec(), shape.clone()))
                        .collect();
                    let input_refs: Vec<&Tensor> = input_tensors.iter().collect();
                    let out_tensor = self.execute(op, &input_refs)?;
                    output.copy_from_slice(out_tensor.data());
                    Ok(())
                }
            }
        }

        pub fn execute_matmul_relu_slices(
            &self,
            a: &[f32],
            a_shape: &Shape,
            b: &[f32],
            b_shape: &Shape,
            output: &mut [f32],
        ) -> Result<(), Error> {
            let m = a_shape.dims()[0];
            let k = a_shape.dims()[1];
            let n = b_shape.dims()[1];
            unsafe {
                crate::blas::sgemm(
                    false,
                    false,
                    m,
                    n,
                    k,
                    1.0,
                    a.as_ptr(),
                    k,
                    b.as_ptr(),
                    n,
                    0.0,
                    output.as_mut_ptr(),
                    n,
                );
            }
            for x in output.iter_mut() {
                if *x < 0.0 {
                    *x = 0.0;
                }
            }
            Ok(())
        }

        pub fn execute_matmul_add_slices(
            &self,
            a: &[f32],
            a_shape: &Shape,
            b: &[f32],
            b_shape: &Shape,
            bias: &[f32],
            output: &mut [f32],
        ) -> Result<(), Error> {
            let m = a_shape.dims()[0];
            let k = a_shape.dims()[1];
            let n = b_shape.dims()[1];

            for r in 0..m {
                let row_offset = r * n;
                output[row_offset..(row_offset + n)].copy_from_slice(bias);
            }

            unsafe {
                crate::blas::sgemm(
                    false,
                    false,
                    m,
                    n,
                    k,
                    1.0,
                    a.as_ptr(),
                    k,
                    b.as_ptr(),
                    n,
                    1.0,
                    output.as_mut_ptr(),
                    n,
                );
            }
            Ok(())
        }

        pub fn execute_elementwise_chain_slices(
            &self,
            expr: &crate::codegen::ast::Expr,
            input: &[f32],
            input_shape: &Shape,
            output: &mut [f32],
            output_shape: &Shape,
        ) -> Result<(), Error> {
            let inputs_slices = [input];
            let input_shapes = [input_shape.clone()];
            for (i, out) in output.iter_mut().enumerate() {
                *out = crate::codegen::ast::evaluate_ast(
                    expr,
                    &[&inputs_slices[0]],
                    &input_shapes,
                    output_shape,
                    i,
                );
            }
            Ok(())
        }
    }

    fn broadcast_op_slices<F>(
        lhs: &[f32],
        lhs_shape: &Shape,
        rhs: &[f32],
        rhs_shape: &Shape,
        out: &mut [f32],
        out_shape: &Shape,
        op: F,
    ) where
        F: Fn(f32, f32) -> f32,
    {
        let out_dims = out_shape.dims();
        let num_elements = out_shape.num_elements();

        let lhs_dims = lhs_shape.dims();
        let rhs_dims = rhs_shape.dims();

        let lhs_strides = get_strides(lhs_dims);
        let rhs_strides = get_strides(rhs_dims);

        let mut coords = [0usize; 8];
        let ndim = out_dims.len();
        assert!(ndim <= 8, "Max dims exceeded in stack allocation");

        for idx in 0..num_elements {
            let mut temp = idx;
            for i in (0..ndim).rev() {
                coords[i] = temp % out_dims[i];
                temp /= out_dims[i];
            }

            let mut lhs_idx = 0;
            for i in 0..lhs_dims.len() {
                let out_axis = out_dims.len() - lhs_dims.len() + i;
                let c = coords[out_axis];
                let lhs_c = if lhs_dims[i] == 1 { 0 } else { c };
                lhs_idx += lhs_c * lhs_strides[i];
            }

            let mut rhs_idx = 0;
            for i in 0..rhs_dims.len() {
                let out_axis = out_dims.len() - rhs_dims.len() + i;
                let c = coords[out_axis];
                let rhs_c = if rhs_dims[i] == 1 { 0 } else { c };
                rhs_idx += rhs_c * rhs_strides[i];
            }

            out[idx] = op(lhs[lhs_idx], rhs[rhs_idx]);
        }
    }

    fn broadcast_op<F>(lhs: &Tensor, rhs: &Tensor, out_shape: &Shape, op: F) -> Tensor
    where
        F: Fn(f32, f32) -> f32,
    {
        let out_dims = out_shape.dims();
        let num_elements = out_shape.num_elements();
        let mut out_data = Vec::with_capacity(num_elements);

        let lhs_dims = lhs.shape().dims();
        let rhs_dims = rhs.shape().dims();

        let lhs_strides = get_strides(lhs_dims);
        let rhs_strides = get_strides(rhs_dims);

        for idx in 0..num_elements {
            let mut temp = idx;
            let mut coords = vec![0; out_dims.len()];
            for i in (0..out_dims.len()).rev() {
                coords[i] = temp % out_dims[i];
                temp /= out_dims[i];
            }

            let mut lhs_idx = 0;
            for i in 0..lhs_dims.len() {
                let out_axis = out_dims.len() - lhs_dims.len() + i;
                let c = coords[out_axis];
                let lhs_c = if lhs_dims[i] == 1 { 0 } else { c };
                lhs_idx += lhs_c * lhs_strides[i];
            }

            let mut rhs_idx = 0;
            for i in 0..rhs_dims.len() {
                let out_axis = out_dims.len() - rhs_dims.len() + i;
                let c = coords[out_axis];
                let rhs_c = if rhs_dims[i] == 1 { 0 } else { c };
                rhs_idx += rhs_c * rhs_strides[i];
            }

            out_data.push(op(lhs.data()[lhs_idx], rhs.data()[rhs_idx]));
        }

        Tensor::new(out_data, out_shape.clone())
    }

    fn get_strides(dims: &[usize]) -> Vec<usize> {
        if dims.is_empty() {
            return vec![];
        }
        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }
}
pub use cpu_mod::CpuBackend;
