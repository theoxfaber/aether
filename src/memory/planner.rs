use crate::graph::{Edge, Node};
use crate::memory::liveness::LivenessMap;
use crate::scheduler::ScheduledOp;
use crate::tensor::{Dtype, TensorId};
use crate::Device;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

/// Represents the offset and size of a tensor in a contiguous memory arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaAllocation {
    pub offset: usize,
    pub size: usize,
}

/// The planned allocations mapping tensors to their locations in the arena.
#[derive(Debug, Clone)]
pub struct StaticMemoryPlan {
    pub allocations: HashMap<TensorId, ArenaAllocation>,
    pub total_size: usize,
}

pub struct StaticMemoryPlanner;

impl StaticMemoryPlanner {
    /// Generate a static memory allocation plan for a device's intermediate tensors.
    ///
    /// # Arguments
    /// * `schedule` - The topological execution schedule.
    /// * `dag` - The underlying computation graph DAG.
    /// * `liveness` - Pre-calculated tensor liveness map.
    /// * `target_tid` - The tensor ID of the final target output.
    /// * `device_filter` - Only plan for tensors computed on this device.
    /// * `target_device` - The global target device requested for execution.
    /// * `graph` - The graph object to query target devices.
    /// * `alignment` - Alignment requirements for offsets (e.g., 256 bytes for GPU).
    pub fn plan(
        schedule: &[ScheduledOp],
        dag: &DiGraph<Node, Edge>,
        liveness: &LivenessMap,
        target_tid: TensorId,
        device_filter: Device,
        target_device: Device,
        graph: &crate::Graph,
        alignment: usize,
    ) -> StaticMemoryPlan {
        let mut allocations = HashMap::new();
        let mut arena_tensors = Vec::new();

        // 1. Collect all intermediate/target tensors computed on the filtered device
        for (op_idx, op) in schedule.iter().enumerate() {
            let out_node = op_output_node(op);
            let tid = dag[out_node].tensor_id;

            // Determine step execution device
            let step_device = graph.get_device(tid, target_device);
            if step_device != device_filter {
                continue;
            }

            // Calculate size based on dtype and shape
            let element_size = match dag[out_node].dtype {
                Dtype::F32 => 4,
                Dtype::F16 => 2,
                Dtype::BF16 => 2,
            };
            let size = dag[out_node].shape.num_elements() * element_size;

            // Define active lifetime interval [start_step, end_step]
            let start_step = op_idx;
            let mut end_step = start_step;

            if tid == target_tid {
                end_step = schedule.len().saturating_sub(1);
            } else if let Some(&last_idx) = liveness.last_use.get(&tid) {
                end_step = last_idx;
            }

            arena_tensors.push((tid, size, start_step, end_step));
        }

        // 2. Sort by size in descending order to minimize fragmentation (Greedy First-Fit Decreasing)
        arena_tensors.sort_by(|a, b| b.1.cmp(&a.1));

        let mut placed: Vec<(TensorId, usize, usize, usize, usize)> = Vec::new(); // (tid, size, start, end, offset)

        for (tid, size, start, end) in arena_tensors {
            let mut candidate_offset = 0;

            loop {
                let mut conflict = false;
                for &(_p_tid, p_size, p_start, p_end, p_offset) in &placed {
                    // Check if lifetimes overlap: max(start1, start2) <= min(end1, end2)
                    let lifetime_overlap =
                        std::cmp::max(start, p_start) <= std::cmp::min(end, p_end);
                    if lifetime_overlap {
                        // Check if byte ranges overlap
                        let byte_overlap = std::cmp::max(candidate_offset, p_offset)
                            < std::cmp::min(candidate_offset + size, p_offset + p_size);
                        if byte_overlap {
                            conflict = true;
                            // Advance candidate offset past the conflicting tensor, respecting alignment
                            let next_possible = p_offset + p_size;
                            let aligned = (next_possible + alignment - 1) / alignment * alignment;
                            if aligned > candidate_offset {
                                candidate_offset = aligned;
                            } else {
                                candidate_offset += alignment;
                            }
                            break;
                        }
                    }
                }
                if !conflict {
                    break;
                }
            }

            placed.push((tid, size, start, end, candidate_offset));
            allocations.insert(
                tid,
                ArenaAllocation {
                    offset: candidate_offset,
                    size,
                },
            );
        }

        let total_size = placed
            .iter()
            .map(|&(_, size, _, _, offset)| offset + size)
            .max()
            .unwrap_or(0);

        StaticMemoryPlan {
            allocations,
            total_size,
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
