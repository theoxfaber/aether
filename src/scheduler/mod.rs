pub mod scheduler_mod {
    use crate::graph::{Edge, GraphTensor, Node, Op};
    use crate::{Device, Error};
    use petgraph::graph::{DiGraph, NodeIndex};
    use std::collections::HashSet;

    #[derive(Debug, Clone)]
    pub struct ScheduledStep {
        pub node_id: NodeIndex,
        pub op: Op,
        pub inputs: Vec<NodeIndex>,
        pub device: Device,
    }

    #[derive(Debug, Clone)]
    pub struct ExecutionPlan {
        pub steps: Vec<ScheduledStep>,
    }

    pub trait Scheduler {
        fn schedule(&self, target: &GraphTensor, device: Device) -> Result<ExecutionPlan, Error>;
    }

    pub struct SimpleScheduler;

    impl Default for SimpleScheduler {
        fn default() -> Self {
            Self::new()
        }
    }

    impl SimpleScheduler {
        pub fn new() -> Self {
            Self
        }

        fn collect_ancestors(
            dag: &DiGraph<Node, Edge>,
            start: NodeIndex,
            visited: &mut HashSet<NodeIndex>,
        ) {
            if visited.insert(start) {
                for neighbor in dag.neighbors_directed(start, petgraph::Direction::Incoming) {
                    Self::collect_ancestors(dag, neighbor, visited);
                }
            }
        }
    }

    impl Scheduler for SimpleScheduler {
        fn schedule(&self, target: &GraphTensor, device: Device) -> Result<ExecutionPlan, Error> {
            let inner = target.graph.inner.read().unwrap();
            let dag = &inner.dag;

            // 1. Find all ancestors of the target node (including the target node itself)
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

            // 4. Generate scheduled steps
            let mut steps = Vec::new();
            for node_id in filtered_sorted {
                let node = &dag[node_id];

                // Gather inputs ordered by LHS/RHS/Unary
                let inputs = if let Op::Input(_) = &node.op {
                    Vec::new()
                } else {
                    crate::graph::get_indexed_inputs(dag, node_id)?
                };

                steps.push(ScheduledStep {
                    node_id,
                    op: node.op.clone(),
                    inputs,
                    device, // Default all ops to the requested device in this simple scheduler
                });
            }

            Ok(ExecutionPlan { steps })
        }
    }
}
pub mod fusion;
pub mod memory_aware;
pub mod prefetch_overlap;
pub use fusion::{FusedOp, FusionPass, ScheduledOp};
pub use memory_aware::{DynamicScheduler, MemoryAwareScheduler};
pub use prefetch_overlap::AsyncPrefetchScheduler;
pub use scheduler_mod::{ExecutionPlan, ScheduledStep, Scheduler, SimpleScheduler};
