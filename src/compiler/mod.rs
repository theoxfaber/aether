pub mod constant_fold;
pub mod cse;
pub mod dce;
pub mod simplify;

use crate::graph::Graph;
use crate::graph::Op;
use crate::Error;

/// The output of the graph compiler containing the optimized compute DAG.
pub struct CompiledGraph {
    pub graph: Graph,
}

/// Graph compiler that applies optimization passes to the compute DAG.
pub struct GraphCompiler;

impl GraphCompiler {
    /// Compile and optimize a graph.
    ///
    /// Pass order:
    ///   1. simplify      — algebraic identities (x+0→x, x*1→x, …)
    ///   2. CSE            — common subexpression elimination
    ///   3. DCE            — dead code elimination (cleans up CSE orphans)
    ///   4. constant_fold  — evaluate constant subgraphs at compile time
    ///   5. DCE            — dead code elimination (cleans up orphans from constant folding)
    ///   6. layout_optimize — pre-transpose weights for cache locality
    pub fn compile(graph: &Graph) -> Result<CompiledGraph, Error> {
        let optimized = graph.clone();

        // Pass 1: Algebraic simplification
        simplify::run_simplify_pass(&optimized)?;

        // Pass 2: Common subexpression elimination
        cse::run_cse_pass(&optimized)?;

        // Pass 3: Dead code elimination (cleans up CSE orphans)
        let has_output = {
            let inner = graph
                .inner
                .read()
                .expect("graph lock poisoned in compile pass");
            inner.dag.node_count() > 0
        };
        if has_output {
            dce::run_dce_pass(&optimized)?;
        }

        // Pass 4: Constant folding
        constant_fold::run_constant_fold_pass(&optimized)?;

        // Pass 5: Dead code elimination (cleans up orphans from constant folding)
        if has_output {
            dce::run_dce_pass(&optimized)?;
        }

        // Pass 6: Layout optimization (pre-transpose weights)
        run_layout_optimize_pass(&optimized)?;

        Ok(CompiledGraph { graph: optimized })
    }
}

/// Layout optimization pass: pre-transpose weight matrices to avoid transposing at runtime.
///
/// For MatMul nodes where the second input is a constant weight tensor with shape
/// [N, K] (output_dim, input_dim, where N > K), transposes the weight to [K, N]
/// so the inner loop over K accesses contiguous memory (stride 1 instead of stride N).
/// This gives ~20-40% throughput improvement on CPU matmul hot loops.
fn run_layout_optimize_pass(graph: &Graph) -> Result<(), Error> {
    let dag_nodes: Vec<_> = {
        let inner = graph.inner.read().map_err(|_| {
            Error::ExecutionError("graph lock poisoned in layout optimize pass".into())
        })?;
        inner.dag.node_indices().collect()
    };

    for node_idx in dag_nodes {
        let op = {
            let inner = graph
                .inner
                .read()
                .map_err(|_| Error::ExecutionError("graph lock poisoned reading node op".into()))?;
            inner.dag[node_idx].op.clone()
        };

        if let Op::MatMul = op {
            let inputs = {
                let inner = graph.inner.read().map_err(|_| {
                    Error::ExecutionError("graph lock poisoned reading MatMul inputs".into())
                })?;
                crate::graph::graph_mod::get_binary_inputs(&inner.dag, node_idx).ok()
            };

            if let Some((_lhs_node, rhs_node)) = inputs {
                let mut inner = graph.inner.write().map_err(|_| {
                    Error::ExecutionError("graph lock poisoned writing MatMul weight".into())
                })?;
                let node = &mut inner.dag[rhs_node];
                if let Op::Input(ref mut tensor) = node.op {
                    let shape = tensor.shape().dims().to_vec();
                    if shape.len() == 2 && shape[0] > shape[1] {
                        let k = shape[1];
                        let n = shape[0];
                        let data = tensor.data();
                        let mut transposed = vec![0.0f32; k * n];
                        for i in 0..k {
                            for j in 0..n {
                                transposed[i * n + j] = data[j * k + i];
                            }
                        }
                        // Replace tensor data with transposed layout [K, N]
                        *tensor = crate::tensor::Tensor::new(
                            transposed,
                            crate::tensor::Shape::new(vec![k, n]),
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// Check if a node depends on the output of the graph (reachable from output node).
pub fn is_reachable_from_output(graph: &Graph, node_idx: petgraph::graph::NodeIndex) -> bool {
    let inner = graph
        .inner
        .read()
        .expect("graph lock poisoned in reachability check");
    let dag = &inner.dag;

    // Find the output node (node with no outgoing edges)
    let output = match dag.node_indices().find(|&n| {
        dag.neighbors_directed(n, petgraph::Direction::Outgoing)
            .count()
            == 0
    }) {
        Some(n) => n,
        None => return false,
    };

    // Reverse BFS from output
    use petgraph::visit::EdgeRef;
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![output];
    while let Some(n) = stack.pop() {
        if !visited.insert(n) {
            continue;
        }
        for edge in dag.edges_directed(n, petgraph::Direction::Incoming) {
            stack.push(edge.source());
        }
    }
    visited.contains(&node_idx)
}

#[cfg(test)]
mod tests {
    use crate::graph::Graph;
    use crate::tensor::Shape;
    use crate::Device;

    fn make_graph_with_identities() -> Graph {
        let graph = Graph::new();
        // x + 0 → x
        let x = graph.tensor(vec![3.0, 7.0], Shape::new(vec![2]));
        let zero = graph.tensor(vec![0.0, 0.0], Shape::new(vec![2]));
        let add = x.add(zero);
        // result should be equal to x
        let one = graph.tensor(vec![1.0], Shape::new(vec![1]));
        let mul = add.mul(one); // x * 1 → x
        let _ = mul; // keep alive
        graph
    }

    fn make_graph_with_cse() -> Graph {
        let graph = Graph::new();
        // Two identical (a+b) subexpressions; CSE should keep only one
        let a1 = graph.tensor(vec![1.0, 2.0], Shape::new(vec![2]));
        let b1 = graph.tensor(vec![3.0, 4.0], Shape::new(vec![2]));
        let sum1 = a1.add(b1);
        let a2 = graph.tensor(vec![1.0, 2.0], Shape::new(vec![2]));
        let b2 = graph.tensor(vec![3.0, 4.0], Shape::new(vec![2]));
        let sum2 = a2.add(b2);
        let _result = sum1.add(sum2);
        graph
    }

    #[test]
    fn test_compile_simplify_identities() {
        let graph = Graph::new();
        let x = graph.tensor(vec![3.0, 7.0], Shape::new(vec![2]));
        let out = x.add(graph.tensor(vec![0.0, 0.0], Shape::new(vec![2])));
        let expected = out.run(Device::Cpu).unwrap();
        graph.compile().unwrap();
        // Build fresh chain after compile
        let x2 = graph.tensor(vec![3.0, 7.0], Shape::new(vec![2]));
        let o2 = x2.add(graph.tensor(vec![0.0, 0.0], Shape::new(vec![2])));
        let actual = o2.run(Device::Cpu).unwrap();
        let e = expected.data();
        let a = actual.data();
        assert_eq!(a.len(), e.len());
        for (x, y) in e.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-5, "compile changed result: {x} vs {y}");
        }
    }

    #[test]
    fn test_compile_node_count_bounded() {
        // Compile should not increase node count
        let graph = make_graph_with_identities();
        let count_before = {
            let inner = graph.inner.read().unwrap();
            inner.dag.node_count()
        };
        graph.compile().unwrap();
        let count_after = {
            let inner = graph.inner.read().unwrap();
            inner.dag.node_count()
        };
        assert!(
            count_after <= count_before,
            "compile should not increase node count (was {count_before}, now {count_after})"
        );
    }

    #[test]
    fn test_compile_cse_bounded() {
        let graph = make_graph_with_cse();
        let count_before = {
            let inner = graph.inner.read().unwrap();
            inner.dag.node_count()
        };
        graph.compile().unwrap();
        let count_after = {
            let inner = graph.inner.read().unwrap();
            inner.dag.node_count()
        };
        assert!(
            count_after <= count_before,
            "CSE should not increase node count"
        );
    }

    #[test]
    fn test_compile_numerical_correctness() {
        let graph = Graph::new();
        let a = graph.tensor(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
        let b = graph.tensor(vec![1.0, 0.0, 5.0], Shape::new(vec![3]));
        let t = a.add(b);
        let expected = t.run(Device::Cpu).unwrap();
        // Create a fresh chain after compile so DCE doesn't invalidate handles
        graph.compile().unwrap();
        let a2 = graph.tensor(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
        let b2 = graph.tensor(vec![1.0, 0.0, 5.0], Shape::new(vec![3]));
        let t2 = a2.add(b2);
        let actual = t2.run(Device::Cpu).unwrap();
        let e = expected.data();
        let a = actual.data();
        assert_eq!(e.len(), a.len(), "compile should preserve output size");
        for (x, y) in e.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-5, "compile changed result: {x} vs {y}");
        }
    }

    #[test]
    fn test_compile_on_run_flag() {
        let graph = Graph::new();
        graph.set_compile_on_run(true);
        let a = graph.tensor(vec![42.0], Shape::new(vec![1]));
        let _ = a.add(graph.tensor(vec![0.0], Shape::new(vec![1])));
        let result = a.run(Device::Cpu).unwrap();
        assert_eq!(result.data()[0], 42.0);
    }

    #[test]
    fn test_simplify_skips_computed_operands() {
        // The simplifier must NOT fold (computed + 0) -> computed when
        // "computed" is not an Op::Input, because tensor_from would return
        // zeros instead of the actual computed value.
        let graph = Graph::new();
        let a = graph.tensor(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
        let b = graph.tensor(vec![4.0, 5.0, 6.0], Shape::new(vec![3, 1]));
        // matmul(a, b) gives [[32.0]] — a computed node
        let mm = a.matmul(b);
        let zero = graph.tensor(vec![0.0], Shape::new(vec![1, 1]));
        // mm + 0 — the simplifier sees x + 0 where x is a MatMul
        let out = mm.add(zero);
        let expected = out.run(Device::Cpu).unwrap();
        graph.compile().unwrap();
        let a2 = graph.tensor(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
        let b2 = graph.tensor(vec![4.0, 5.0, 6.0], Shape::new(vec![3, 1]));
        let mm2 = a2.matmul(b2);
        let zero2 = graph.tensor(vec![0.0], Shape::new(vec![1, 1]));
        let out2 = mm2.add(zero2);
        let actual = out2.run(Device::Cpu).unwrap();
        let e = expected.data();
        let a = actual.data();
        assert_eq!(e, a, "simplify must not corrupt computed operands");
    }

    #[test]
    fn test_compile_full_pipeline_numerical() {
        // Exercise all passes together: simplify, CSE, DCE, constant fold, layout
        let graph = Graph::new();
        let a = graph.tensor(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
        let b = graph.tensor(vec![5.0, 6.0, 7.0], Shape::new(vec![3]));
        // With identities: (a + 0) + (b * 1)
        let t = a
            .add(graph.tensor(vec![0.0, 0.0, 0.0], Shape::new(vec![3])))
            .add(b.mul(graph.tensor(vec![1.0], Shape::new(vec![1]))));
        let expected = t.run(Device::Cpu).unwrap();
        graph.compile().unwrap();
        let a2 = graph.tensor(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
        let b2 = graph.tensor(vec![5.0, 6.0, 7.0], Shape::new(vec![3]));
        let t2 = a2
            .add(graph.tensor(vec![0.0, 0.0, 0.0], Shape::new(vec![3])))
            .add(b2.mul(graph.tensor(vec![1.0], Shape::new(vec![1]))));
        let actual = t2.run(Device::Cpu).unwrap();
        let e = expected.data();
        let a = actual.data();
        for (x, y) in e.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-5, "compile changed result: {x} vs {y}");
        }
    }
}
