use crate::graph::{Graph, Op};
use crate::scheduler::ScheduledOp;
use crate::tensor::TensorId;
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};

/// Represents a set of tensor prefetching and eviction schedules for execution.
pub struct PrefetchPlan {
    /// List of tensor IDs to prefetch to GPU before executing the step at index.
    pub prefetch_before: HashMap<usize, Vec<TensorId>>,
    /// List of tensor IDs that can be eagerly evicted from GPU after executing the step at index.
    pub evict_after: HashMap<usize, Vec<TensorId>>,
}

/// Dynamic analysis scheduler to minimize memory footprint and optimize host-device memory transfers.
pub struct PrefetchScheduler;

impl PrefetchScheduler {
    /// Generate a prefetch and eviction plan for a given topological schedule of operations.
    pub fn plan(schedule: &[ScheduledOp], graph: &Graph) -> PrefetchPlan {
        let inner = graph.inner.read().unwrap();
        let dag = &inner.dag;

        let mut first_use = HashMap::new();
        let mut last_use = HashMap::new();
        let mut all_tensors = HashSet::new();

        // Target tensor ID is the output of the last step in the schedule
        let target_tid = if let Some(last_op) = schedule.last() {
            let (_, out_idx) = Self::get_inputs_and_output(last_op, graph);
            Some(dag[out_idx].tensor_id)
        } else {
            None
        };

        for (op_idx, op) in schedule.iter().enumerate() {
            let (inputs, output) = Self::get_inputs_and_output(op, graph);

            let output_tid = dag[output].tensor_id;
            all_tensors.insert(output_tid);

            for idx in inputs {
                let tid = dag[idx].tensor_id;
                all_tensors.insert(tid);
                first_use.entry(tid).or_insert(op_idx);
                last_use.insert(tid, op_idx);
            }
        }

        let mut prefetch_before: HashMap<usize, Vec<TensorId>> = HashMap::new();
        let mut evict_after: HashMap<usize, Vec<TensorId>> = HashMap::new();

        for &tid in &all_tensors {
            let mut node_idx_opt = None;
            for idx in dag.node_indices() {
                if dag[idx].tensor_id == tid {
                    node_idx_opt = Some(idx);
                    break;
                }
            }

            if let Some(node_idx) = node_idx_opt {
                let is_input = matches!(dag[node_idx].op, Op::Input(_));
                if is_input {
                    if let Some(&first_idx) = first_use.get(&tid) {
                        let prefetch_idx = if first_idx == 0 { 0 } else { first_idx - 1 };
                        prefetch_before.entry(prefetch_idx).or_default().push(tid);
                    }
                }

                if Some(tid) != target_tid {
                    if let Some(&last_idx) = last_use.get(&tid) {
                        evict_after.entry(last_idx).or_default().push(tid);
                    }
                }
            }
        }

        PrefetchPlan {
            prefetch_before,
            evict_after,
        }
    }

    fn get_inputs_and_output(op: &ScheduledOp, graph: &Graph) -> (Vec<NodeIndex>, NodeIndex) {
        match op {
            ScheduledOp::Plain(node_id, _) => {
                let inner_graph = graph.inner.read().unwrap();
                let dag = &inner_graph.dag;
                use petgraph::visit::EdgeRef;
                let mut inputs = Vec::new();
                for edge in dag.edges_directed(*node_id, petgraph::Direction::Incoming) {
                    inputs.push(edge.source());
                }
                (inputs, *node_id)
            }
            ScheduledOp::Fused(fused) => match fused {
                crate::scheduler::FusedOp::MatMulRelu { a, b, output } => (vec![*a, *b], *output),
                crate::scheduler::FusedOp::MatMulAdd { a, b, bias, output } => {
                    (vec![*a, *b, *bias], *output)
                }
                crate::scheduler::FusedOp::MatMulAddRelu { a, b, bias, output } => {
                    (vec![*a, *b, *bias], *output)
                }
                crate::scheduler::FusedOp::ElementwiseChain { input, output, .. } => {
                    (vec![*input], *output)
                }
            },
        }
    }
}
