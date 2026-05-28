use crate::graph::{Edge, Node};
use crate::scheduler::ScheduledOp;
use crate::tensor::TensorId;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::{HashMap, HashSet};

/// Maps tensor lifetimes to schedule indices for precise eviction planning.
///
/// After each op executes, all tensors whose `last_use` is that op index
/// can be safely freed from GPU memory.
pub struct LivenessMap {
    /// For each TensorId: the last op index that reads it.
    pub last_use: HashMap<TensorId, usize>,
    /// For each op index: tensors that can be freed after it executes.
    pub free_after: HashMap<usize, Vec<TensorId>>,
}

impl LivenessMap {
    /// Analyze a schedule to compute precise tensor liveness.
    ///
    /// Walks the schedule in order, recording for each tensor the highest
    /// op index that uses it as input.
    pub fn analyze(schedule: &[ScheduledOp], dag: &DiGraph<Node, Edge>) -> Self {
        let mut last_use: HashMap<TensorId, usize> = HashMap::new();
        let mut all_tensors: HashSet<TensorId> = HashSet::new();

        // Target tensor (output of last op) must never be freed early
        let target_tid = schedule.last().map(|op| {
            let node_id = op_output_node(op);
            dag[node_id].tensor_id
        });

        for (op_idx, op) in schedule.iter().enumerate() {
            let node_id = op_output_node(op);
            let output_tid = dag[node_id].tensor_id;
            all_tensors.insert(output_tid);

            // Record last use for each input tensor
            for input_node in op_inputs(op, dag) {
                let tid = dag[input_node].tensor_id;
                all_tensors.insert(tid);
                last_use.insert(tid, op_idx);
            }
        }

        // Build free_after map: for each tensor, schedule it to be freed
        // at its last use op index, unless it's the target tensor
        let mut free_after: HashMap<usize, Vec<TensorId>> = HashMap::new();
        for &tid in &all_tensors {
            if Some(tid) == target_tid {
                continue; // never free the output tensor
            }
            if let Some(&last_idx) = last_use.get(&tid) {
                free_after.entry(last_idx).or_default().push(tid);
            }
        }

        LivenessMap {
            last_use,
            free_after,
        }
    }
}

fn op_output_node(op: &ScheduledOp) -> NodeIndex {
    match op {
        ScheduledOp::Plain(node_id, _) => *node_id,
        ScheduledOp::Fused(fused) => match fused {
            crate::scheduler::FusedOp::MatMulRelu { output, .. } => *output,
            crate::scheduler::FusedOp::MatMulAdd { output, .. } => *output,
            crate::scheduler::FusedOp::MatMulAddRelu { output, .. } => *output,
            crate::scheduler::FusedOp::ElementwiseChain { output, .. } => *output,
        },
    }
}

fn op_inputs(op: &ScheduledOp, dag: &DiGraph<Node, Edge>) -> Vec<NodeIndex> {
    match op {
        ScheduledOp::Plain(node_id, _) => {
            use petgraph::visit::EdgeRef;
            dag.edges_directed(*node_id, petgraph::Direction::Incoming)
                .map(|e| e.source())
                .collect()
        }
        ScheduledOp::Fused(fused) => match fused {
            crate::scheduler::FusedOp::MatMulRelu { a, b, .. } => vec![*a, *b],
            crate::scheduler::FusedOp::MatMulAdd { a, b, bias, .. } => vec![*a, *b, *bias],
            crate::scheduler::FusedOp::MatMulAddRelu { a, b, bias, .. } => vec![*a, *b, *bias],
            crate::scheduler::FusedOp::ElementwiseChain { input, .. } => vec![*input],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::{ScheduledOp, Scheduler, SimpleScheduler};
    use crate::{Device, Graph, Shape};

    #[test]
    fn test_liveness_analysis_basic() {
        let graph = Graph::new();
        let a = graph.tensor(vec![1.0; 4], Shape::new(vec![2, 2]));
        let b = graph.tensor(vec![2.0; 4], Shape::new(vec![2, 2]));
        let c = a.matmul(b).relu().sum_all();

        let scheduler = SimpleScheduler::new();
        let plan = scheduler.schedule(&c, Device::Cpu).unwrap();
        let schedule: Vec<ScheduledOp> = plan
            .steps
            .into_iter()
            .map(|step| ScheduledOp::Plain(step.node_id, step.op))
            .collect();

        let inner = graph.inner.read().unwrap();
        let dag = &inner.dag;
        let liveness = LivenessMap::analyze(&schedule, dag);

        // Input tensors should have last_use > 0 (used by matmul)
        assert!(
            !liveness.last_use.is_empty(),
            "Should have at least some last_use entries"
        );
        // free_after should have entries for intermediate tensors
        assert!(
            !liveness.free_after.is_empty(),
            "Should have at least one free_after entry"
        );
        // The output tensor (sum_all result) should not be in free_after
        let last_node = schedule.last().unwrap();
        let last_tid = dag[op_output_node(last_node)].tensor_id;
        let is_output_freed = liveness.free_after.values().any(|v| v.contains(&last_tid));
        assert!(
            !is_output_freed,
            "Output tensor should never be freed early"
        );
    }

    #[test]
    fn test_liveness_intermediate_tensor_freed_early() {
        let graph = Graph::new();
        let a = graph.tensor(vec![1.0; 4], Shape::new(vec![2, 2]));
        let b = graph.tensor(vec![2.0; 4], Shape::new(vec![2, 2]));
        // a is used in matmul, but not after — should be freed after op 0
        let c = a.matmul(b).relu();

        let scheduler = SimpleScheduler::new();
        let plan = scheduler.schedule(&c, Device::Cpu).unwrap();
        let schedule: Vec<ScheduledOp> = plan
            .steps
            .into_iter()
            .map(|step| ScheduledOp::Plain(step.node_id, step.op))
            .collect();

        let inner = graph.inner.read().unwrap();
        let dag = &inner.dag;
        let liveness = LivenessMap::analyze(&schedule, dag);

        // Tensor `a` should be freed after op 0 (its only use)
        let a_tid = dag[a.id].tensor_id;
        if let Some(&last_idx) = liveness.last_use.get(&a_tid) {
            assert!(
                last_idx < schedule.len().saturating_sub(1),
                "Tensor `a` should be freed before the final op (last_use={}, total_ops={})",
                last_idx,
                schedule.len()
            );
        }
    }
}
