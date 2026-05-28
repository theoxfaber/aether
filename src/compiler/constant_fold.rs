use crate::backend::{Backend, CpuBackend};
use crate::graph::graph_mod::get_indexed_inputs;
use crate::graph::{Graph, Op};
use crate::tensor::{Dtype, Tensor, TensorId};
use crate::Error;

/// Constant folding pass: evaluate sub-graphs with all-constant inputs at compile time.
/// Replaces constant expressions with a single Input node containing the pre-computed value.
pub fn run_constant_fold_pass(graph: &Graph) -> Result<(), Error> {
    let cpu_backend = CpuBackend::new();
    let mut folded = true;

    while folded {
        folded = false;
        let nodes_to_check: Vec<_> = {
            let inner = graph
                .inner
                .read()
                .expect("graph lock poisoned collecting nodes in constant fold");
            inner.dag.node_indices().collect()
        };

        for node_idx in nodes_to_check {
            let op = {
                let inner = graph
                    .inner
                    .read()
                    .expect("graph lock poisoned reading node op in constant fold");
                inner.dag[node_idx].op.clone()
            };

            // Skip Input nodes (they are already constants)
            if matches!(op, Op::Input(_)) {
                continue;
            }

            // Check if all inputs are Input nodes (constants)
            let all_inputs_const = {
                let inner = graph
                    .inner
                    .read()
                    .expect("graph lock poisoned checking inputs in constant fold");
                let inputs = match get_indexed_inputs(&inner.dag, node_idx) {
                    Ok(inputs) => inputs,
                    Err(_) => continue,
                };
                inputs
                    .iter()
                    .all(|&in_node| matches!(inner.dag[in_node].op, Op::Input(_)))
            };

            if all_inputs_const {
                // Evaluate the op on CPU
                let inputs = {
                    let inner = graph
                        .inner
                        .read()
                        .expect("graph lock poisoned getting indexed inputs in constant fold");
                    get_indexed_inputs(&inner.dag, node_idx).ok()
                };

                if let Some(inputs) = inputs {
                    let mut input_tensors = Vec::new();
                    {
                        let inner = graph.inner.read().expect(
                            "graph lock poisoned extracting input tensors in constant fold",
                        );
                        for &in_node in &inputs {
                            if let Op::Input(ref tensor) = inner.dag[in_node].op {
                                input_tensors.push(tensor.clone());
                            }
                        }
                    }

                    if !input_tensors.is_empty() {
                        let input_refs: Vec<&Tensor> = input_tensors.iter().collect();
                        if let Ok(out_tensor) = cpu_backend.execute(&op, &input_refs) {
                            // Replace the node with a constant Input
                            let out_shape = out_tensor.shape().clone();
                            let const_tensor =
                                Tensor::new(out_tensor.data().to_vec(), out_shape.clone());
                            let new_node = crate::graph::Node {
                                op: Op::Input(const_tensor),
                                shape: out_shape,
                                dtype: Dtype::F32,
                                tensor_id: TensorId::next(),
                            };
                            let mut inner = graph.inner.write().expect(
                                "graph lock poisoned replacing node with constant in constant fold",
                            );
                            inner.dag[node_idx] = new_node;
                            folded = true;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
