use crate::{Device, Graph, GraphTensor, Tensor};

/// Compares analytical gradients (computed via backprop) with numerical gradients
/// computed using finite-difference approximation.
/// Returns Ok(()) if the differences are within tolerance, or an Err describing the mismatch.
pub fn gradcheck<F>(
    f: F,
    inputs: Vec<Tensor>,
    device: Device,
    eps: f32,
    tol: f32,
) -> Result<(), String>
where
    F: Fn(&[GraphTensor]) -> GraphTensor,
{
    // 1. Run analytical backward pass
    let graph = Graph::new();
    let graph_inputs: Vec<GraphTensor> = inputs
        .iter()
        .map(|t| graph.tensor(t.data().to_vec(), t.shape().clone()))
        .collect();

    let output = f(&graph_inputs);
    let grads = output
        .backward()
        .map_err(|e| format!("Backward pass failed: {:?}", e))?;

    let mut analytical_grads = Vec::new();
    for (i, input) in graph_inputs.iter().enumerate() {
        let grad_tensor = grads
            .get(&input.id())
            .ok_or_else(|| format!("No gradient computed for input tensor {}", i))?;
        let grad_eval = grad_tensor
            .run(device)
            .map_err(|e| format!("Failed to run analytical gradient: {:?}", e))?;
        analytical_grads.push(grad_eval.data().to_vec());
    }

    // 2. Compute numerical gradients using finite differences
    for i in 0..inputs.len() {
        let input_shape = inputs[i].shape().clone();
        let num_elements = input_shape.num_elements();
        let analytical_grad = &analytical_grads[i];

        for j in 0..num_elements {
            // Compute y_plus: inputs[i][j] += eps
            let mut inputs_plus = inputs.clone();
            if let crate::tensor::AnyData::F32(ref mut data) = inputs_plus[i].data {
                data[j] += eps;
            }

            let graph_plus = Graph::new();
            let graph_inputs_plus: Vec<GraphTensor> = inputs_plus
                .iter()
                .map(|t| graph_plus.tensor(t.data().to_vec(), t.shape().clone()))
                .collect();
            let output_plus = f(&graph_inputs_plus);
            let val_plus_t = output_plus
                .run(device)
                .map_err(|e| format!("Failed to run output_plus: {:?}", e))?;
            let val_plus = val_plus_t.data()[0];

            // Compute y_minus: inputs[i][j] -= eps
            let mut inputs_minus = inputs.clone();
            if let crate::tensor::AnyData::F32(ref mut data) = inputs_minus[i].data {
                data[j] -= eps;
            }

            let graph_minus = Graph::new();
            let graph_inputs_minus: Vec<GraphTensor> = inputs_minus
                .iter()
                .map(|t| graph_minus.tensor(t.data().to_vec(), t.shape().clone()))
                .collect();
            let output_minus = f(&graph_inputs_minus);
            let val_minus_t = output_minus
                .run(device)
                .map_err(|e| format!("Failed to run output_minus: {:?}", e))?;
            let val_minus = val_minus_t.data()[0];

            let numerical_grad = (val_plus - val_minus) / (2.0 * eps);
            let analytical_val = analytical_grad[j];

            let diff = (analytical_val - numerical_grad).abs();
            let denominator = analytical_val.abs().max(numerical_grad.abs()).max(1e-5);
            let rel_diff = diff / denominator;

            if diff > tol && rel_diff > tol {
                return Err(format!(
                    "Gradient mismatch at tensor {}, element {} / {}.\n\
                     Shape: {:?}\n\
                     Analytical: {}\n\
                     Numerical: {}\n\
                     Absolute diff: {}\n\
                     Relative diff: {}",
                    i,
                    j,
                    num_elements,
                    input_shape.dims(),
                    analytical_val,
                    numerical_grad,
                    diff,
                    rel_diff
                ));
            }
        }
    }

    Ok(())
}
