use petgraph::graph::NodeIndex;
use tracing::{debug, info};

use crate::graph::{Edge, GraphTensor, Node, Op};
use crate::Error;
use std::collections::HashSet;

/// Defines an operation that is fused into a single combined kernel execution.
#[derive(Debug, Clone)]
pub enum FusedOp {
    /// Fused matrix multiplication and ReLU activation.
    MatMulRelu {
        /// Left input node.
        a: NodeIndex,
        /// Right input node.
        b: NodeIndex,
        /// Output node.
        output: NodeIndex,
    },
    /// Fused matrix multiplication and bias addition.
    MatMulAdd {
        /// Left input node.
        a: NodeIndex,
        /// Right input node.
        b: NodeIndex,
        /// Bias tensor node.
        bias: NodeIndex,
        /// Output node.
        output: NodeIndex,
    },
    /// Fused matrix multiplication, bias addition, and ReLU activation.
    MatMulAddRelu {
        /// Left input node.
        a: NodeIndex,
        /// Right input node.
        b: NodeIndex,
        /// Bias tensor node.
        bias: NodeIndex,
        /// Output node.
        output: NodeIndex,
    },
    /// Fused chain of elementwise operations.
    ElementwiseChain {
        /// Entry input node.
        input: NodeIndex,
        /// Sequential operations to apply.
        ops: Vec<Op>,
        /// Output node.
        output: NodeIndex,
    },
}

/// Represents either a single step or a fused block of steps in an execution plan.
#[derive(Debug, Clone)]
pub enum ScheduledOp {
    /// A group of operations fused together.
    Fused(FusedOp),
    /// A single, unfused operation.
    Plain(NodeIndex, Op),
}

/// An optimization scheduler pass that merges adjacent operations to minimize host-device memory roundtrips.
pub struct FusionPass;

impl FusionPass {
    /// Run the operator fusion compiler pass starting from a target output tensor.
    pub fn run(target: &GraphTensor) -> Result<Vec<ScheduledOp>, Error> {
        let inner = target.graph.inner.read().unwrap();
        let dag = &inner.dag;

        // 1. Find all ancestors of the target node (including target node itself)
        let mut ancestors = HashSet::new();
        Self::collect_ancestors(dag, target.id, &mut ancestors);

        // 2. Perform global topological sort
        let sorted_nodes = petgraph::algo::toposort(dag, None)
            .map_err(|_| Error::ExecutionError("Cycle detected in graph".to_string()))?;

        // 3. Filter topological sort to only include ancestors
        let filtered_sorted: Vec<NodeIndex> = sorted_nodes
            .into_iter()
            .filter(|id| ancestors.contains(id))
            .collect();

        // 4. Generate scheduled ops with fusion detection
        let mut schedule = Vec::new();
        let mut fused_set = HashSet::new();
        let mut fused_count = 0usize;

        for id in filtered_sorted {
            if fused_set.contains(&id) {
                continue;
            }

            let node = &dag[id];

            if matches!(node.op, Op::MatMul) {
                let consumers: Vec<NodeIndex> = dag
                    .neighbors_directed(id, petgraph::Direction::Outgoing)
                    .collect();
                if consumers.len() == 1 && ancestors.contains(&consumers[0]) {
                    let consumer_id = consumers[0];
                    let consumer_node = &dag[consumer_id];

                    let inputs = Self::get_inputs(dag, id)?;
                    if inputs.len() == 2 {
                        let a_shape = &dag[inputs[0]].shape;
                        let b_shape = &dag[inputs[1]].shape;
                        let m = a_shape.dims()[0] as u32;
                        let k = a_shape.dims()[1] as u32;
                        let n = b_shape.dims()[1] as u32;

                        let (bandwidth_gbps, compute_flops_gflops) =
                            if let Ok(backend) = crate::backend::WgpuBackend::get_or_init() {
                                (
                                    backend.memory_bandwidth_gbps(),
                                    backend.compute_flops_gflops(),
                                )
                            } else {
                                (100.0, 3600.0)
                            };

                        match &consumer_node.op {
                            Op::Relu => {
                                // Evaluate cost model
                                let bytes_unfused = (m * k + k * n + 3 * m * n) as f64 * 4.0;
                                let bytes_fused = (m * k + k * n + m * n) as f64 * 4.0;
                                let saved_bytes = bytes_unfused - bytes_fused;
                                let mem_time_saved = saved_bytes / (bandwidth_gbps * 1e9);

                                let matmul_flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
                                let math_time_unfused = matmul_flops / (compute_flops_gflops * 1e9);
                                let overhead_factor = 0.01;
                                let math_time_overhead = math_time_unfused * overhead_factor;

                                if mem_time_saved > math_time_overhead {
                                    fused_set.insert(consumer_id);
                                    schedule.push(ScheduledOp::Fused(FusedOp::MatMulRelu {
                                        a: inputs[0],
                                        b: inputs[1],
                                        output: consumer_id,
                                    }));
                                    fused_count += 1;
                                    continue;
                                }
                                debug!(
                                    "MatMulRelu skipped: cost model rejected {}×{} (bw={:.0} GB/s, flops={:.0} GFLOP/s, mem_saved={:.2} µs, math_overhead={:.2} µs)",
                                    m, n, bandwidth_gbps, compute_flops_gflops, mem_time_saved * 1e6, math_time_overhead * 1e6
                                );
                            }
                            Op::Add => {
                                let add_inputs = Self::get_inputs(dag, consumer_id)?;
                                if add_inputs.len() == 2 {
                                    let bias = if add_inputs[0] == id {
                                        add_inputs[1]
                                    } else {
                                        add_inputs[0]
                                    };
                                    let bias_shape = &dag[bias].shape;
                                    let bias_len =
                                        bias_shape.dims().iter().product::<usize>() as u32;

                                    // Check if we can fuse all the way to Relu: MatMul + Add + Relu -> MatMulAddRelu
                                    let mut fused_to_relu = false;
                                    let add_consumers: Vec<NodeIndex> = dag
                                        .neighbors_directed(
                                            consumer_id,
                                            petgraph::Direction::Outgoing,
                                        )
                                        .collect();
                                    if add_consumers.len() == 1
                                        && ancestors.contains(&add_consumers[0])
                                    {
                                        let next_id = add_consumers[0];
                                        if matches!(dag[next_id].op, Op::Relu) {
                                            // Cost model for MatMulAddRelu
                                            let bytes_unfused =
                                                (m * k + k * n + 5 * m * n + bias_len) as f64 * 4.0;
                                            let bytes_fused =
                                                (m * k + k * n + m * n + bias_len) as f64 * 4.0;
                                            let saved_bytes = bytes_unfused - bytes_fused;
                                            let mem_time_saved =
                                                saved_bytes / (bandwidth_gbps * 1e9);

                                            let matmul_flops =
                                                2.0 * (m as f64) * (n as f64) * (k as f64);
                                            let math_time_unfused =
                                                matmul_flops / (compute_flops_gflops * 1e9);
                                            let overhead_factor = 0.01;
                                            let math_time_overhead =
                                                math_time_unfused * overhead_factor;

                                            if mem_time_saved > math_time_overhead {
                                                fused_set.insert(consumer_id);
                                                fused_set.insert(next_id);
                                                schedule.push(ScheduledOp::Fused(
                                                    FusedOp::MatMulAddRelu {
                                                        a: inputs[0],
                                                        b: inputs[1],
                                                        bias,
                                                        output: next_id,
                                                    },
                                                ));
                                                fused_count += 1;
                                                fused_to_relu = true;
                                            }
                                        }
                                    }

                                    if fused_to_relu {
                                        continue;
                                    }

                                    // Evaluate standard MatMulAdd cost model
                                    let bytes_unfused =
                                        (m * k + k * n + 3 * m * n + bias_len) as f64 * 4.0;
                                    let bytes_fused =
                                        (m * k + k * n + m * n + bias_len) as f64 * 4.0;
                                    let saved_bytes = bytes_unfused - bytes_fused;
                                    let mem_time_saved = saved_bytes / (bandwidth_gbps * 1e9);

                                    let matmul_flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
                                    let math_time_unfused =
                                        matmul_flops / (compute_flops_gflops * 1e9);
                                    let overhead_factor = 0.01;
                                    let math_time_overhead = math_time_unfused * overhead_factor;

                                    if mem_time_saved > math_time_overhead {
                                        fused_set.insert(consumer_id);
                                        schedule.push(ScheduledOp::Fused(FusedOp::MatMulAdd {
                                            a: inputs[0],
                                            b: inputs[1],
                                            bias,
                                            output: consumer_id,
                                        }));
                                        fused_count += 1;
                                        continue;
                                    }
                                    debug!(
                                        "MatMulAdd skipped: cost model rejected {}×{} (bw={:.0} GB/s, flops={:.0} GFLOP/s, mem_saved={:.2} µs, math_overhead={:.2} µs)",
                                        m, n, bandwidth_gbps, compute_flops_gflops, mem_time_saved * 1e6, math_time_overhead * 1e6
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            schedule.push(ScheduledOp::Plain(id, node.op.clone()));
        }

        info!(
            "detected {} fused ops, {} total steps",
            fused_count,
            schedule.len()
        );
        Ok(schedule)
    }

    fn collect_ancestors(
        dag: &petgraph::graph::DiGraph<Node, Edge>,
        start: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
    ) {
        if visited.insert(start) {
            for neighbor in dag.neighbors_directed(start, petgraph::Direction::Incoming) {
                Self::collect_ancestors(dag, neighbor, visited);
            }
        }
    }

    fn get_inputs(
        dag: &petgraph::graph::DiGraph<Node, Edge>,
        node_id: NodeIndex,
    ) -> Result<Vec<NodeIndex>, Error> {
        crate::graph::get_indexed_inputs(dag, node_id)
    }
}
