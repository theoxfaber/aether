use crate::graph::graph_mod::{get_binary_inputs, get_unary_input};
use crate::graph::{Graph, Op};
use crate::tensor::Tensor;
use crate::Error;

/// Algebraic simplification pass.
/// Simplifies expressions like:
///   x + 0 -> x
///   x * 1 -> x
///   x * 0 -> 0
///   x - x -> 0
///   x / 1 -> x
///   neg(neg(x)) -> x
///   exp(0) -> 1
///   sqrt(x)^2 -> x (if x >= 0)
pub fn run_simplify_pass(graph: &Graph) -> Result<(), Error> {
    let mut changed = true;
    while changed {
        changed = false;

        let nodes: Vec<_> = {
            let inner = graph
                .inner
                .read()
                .expect("graph lock poisoned collecting nodes in simplify pass");
            inner.dag.node_indices().collect()
        };

        for node_idx in &nodes {
            let op = {
                let inner = graph
                    .inner
                    .read()
                    .expect("graph lock poisoned reading node op in simplify pass");
                inner.dag[*node_idx].op.clone()
            };

            let result = try_simplify_node(graph, *node_idx, &op);
            if let Some(new_op) = result {
                let mut inner = graph
                    .inner
                    .write()
                    .expect("graph lock poisoned writing updated op in simplify pass");
                if let Some(node) = inner.dag.node_weight_mut(*node_idx) {
                    node.op = new_op;
                    changed = true;
                }
            }
        }
    }

    Ok(())
}

/// Try to simplify a single node. If simplification is possible, returns Some(new_op).
/// Otherwise returns None.
fn try_simplify_node(graph: &Graph, node_idx: petgraph::graph::NodeIndex, op: &Op) -> Option<Op> {
    match op {
        Op::Add => {
            // x + 0 -> x
            if let Ok((lhs, rhs)) = get_binary_inputs(
                &graph
                    .inner
                    .read()
                    .expect("graph lock poisoned in add simplification")
                    .dag,
                node_idx,
            ) {
                if is_zero_node(graph, rhs) {
                    if let Some(t) = tensor_from(graph, lhs) {
                        return Some(Op::Input(t));
                    }
                }
                if is_zero_node(graph, lhs) {
                    if let Some(t) = tensor_from(graph, rhs) {
                        return Some(Op::Input(t));
                    }
                }
            }
            None
        }
        Op::Mul => {
            // x * 0 -> 0
            if let Ok((lhs, rhs)) = get_binary_inputs(
                &graph
                    .inner
                    .read()
                    .expect("graph lock poisoned in mul simplification")
                    .dag,
                node_idx,
            ) {
                if is_zero_node(graph, lhs) || is_zero_node(graph, rhs) {
                    return Some(Op::Input(zero_tensor_from_shape(graph, node_idx)));
                }
                // x * 1 -> x
                if is_one_node(graph, rhs) {
                    if let Some(t) = tensor_from(graph, lhs) {
                        return Some(Op::Input(t));
                    }
                }
                if is_one_node(graph, lhs) {
                    if let Some(t) = tensor_from(graph, rhs) {
                        return Some(Op::Input(t));
                    }
                }
            }
            None
        }
        Op::Sub => {
            // x - x -> 0
            if let Ok((lhs, rhs)) = get_binary_inputs(
                &graph
                    .inner
                    .read()
                    .expect("graph lock poisoned in sub simplification")
                    .dag,
                node_idx,
            ) {
                if lhs == rhs {
                    return Some(Op::Input(zero_tensor_from_shape(graph, node_idx)));
                }
            }
            None
        }
        Op::Div => {
            // x / 1 -> x
            if let Ok((lhs, rhs)) = get_binary_inputs(
                &graph
                    .inner
                    .read()
                    .expect("graph lock poisoned in div simplification")
                    .dag,
                node_idx,
            ) {
                if is_one_node(graph, rhs) {
                    if let Some(t) = tensor_from(graph, lhs) {
                        return Some(Op::Input(t));
                    }
                }
            }
            None
        }
        Op::Neg => {
            // neg(neg(x)) -> x
            if let Ok(input) = get_unary_input(
                &graph
                    .inner
                    .read()
                    .expect("graph lock poisoned in neg simplification")
                    .dag,
                node_idx,
            ) {
                let inner_op = &graph
                    .inner
                    .read()
                    .expect("graph lock poisoned reading neg inner op")
                    .dag[input]
                    .op;
                    if matches!(inner_op, Op::Neg) {
                        if let Ok(inner_input) = get_unary_input(
                            &graph
                                .inner
                                .read()
                                .expect("graph lock poisoned reading neg inner input")
                                .dag,
                            input,
                        ) {
                            if let Some(t) = tensor_from(graph, inner_input) {
                                return Some(Op::Input(t));
                            }
                        }
                    }
            }
            None
        }
        _ => None,
    }
}

fn is_zero_node(graph: &Graph, node: petgraph::graph::NodeIndex) -> bool {
    let inner = graph
        .inner
        .read()
        .expect("graph lock poisoned in is_zero_node");
    if let Op::Input(tensor) = &inner.dag[node].op {
        tensor.data().iter().all(|&x| x == 0.0) && !tensor.data().is_empty()
    } else {
        false
    }
}

fn is_one_node(graph: &Graph, node: petgraph::graph::NodeIndex) -> bool {
    let inner = graph
        .inner
        .read()
        .expect("graph lock poisoned in is_one_node");
    if let Op::Input(tensor) = &inner.dag[node].op {
        tensor.data().iter().all(|&x| x == 1.0) && !tensor.data().is_empty()
    } else {
        false
    }
}

fn tensor_from(graph: &Graph, node: petgraph::graph::NodeIndex) -> Option<Tensor> {
    let inner = graph
        .inner
        .read()
        .expect("graph lock poisoned in tensor_from");
    let op = &inner.dag[node].op;
    match op {
        Op::Input(t) => Some(t.clone()),
        _ => None,
    }
}

fn zero_tensor_from_shape(graph: &Graph, node: petgraph::graph::NodeIndex) -> Tensor {
    let inner = graph
        .inner
        .read()
        .expect("graph lock poisoned in zero_tensor_from_shape");
    let shape = inner.dag[node].shape.clone();
    let numel = shape.num_elements();
    Tensor::new(vec![0.0; numel], shape)
}
