use crate::graph::{Graph, Op};
use crate::Error;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Hash for detecting common subexpressions.
/// Two nodes with the same op and same input tensor IDs are considered identical.
fn node_hash(
    inner: &crate::graph::graph_mod::GraphInner,
    node: petgraph::graph::NodeIndex,
) -> Option<u64> {
    let dag = &inner.dag;
    let op = &dag[node].op;

    if !matches!(
        op,
        Op::Input(_)
            | Op::MatMul
            | Op::Relu
            | Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::Tanh
            | Op::Sigmoid
            | Op::Exp
            | Op::Sqrt
            | Op::Neg
            | Op::BroadcastAdd { .. }
            | Op::BroadcastMul { .. }
            | Op::BroadcastSub { .. }
            | Op::BroadcastDiv { .. }
            | Op::Transpose
            | Op::SumAll
            | Op::SumDim { .. }
            | Op::Reshape { .. }
            | Op::Softmax
            | Op::Concat { .. }
    ) {
        return None;
    }

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    format!("{:?}", op).hash(&mut hasher);

    // Hash input tensor IDs (sorted by source node index for deterministic order)
    let mut inputs: Vec<_> = dag
        .edges_directed(node, petgraph::Direction::Incoming)
        .map(|e| (e.source(), *e.weight()))
        .collect();
    inputs.sort_by_key(|(src, _)| src.index());
    for (src, _) in &inputs {
        let id = dag[*src].tensor_id;
        id.0.hash(&mut hasher);
    }

    Some(hasher.finish())
}

/// Common Subexpression Elimination pass.
/// Replaces duplicate computations with references to the first computed result.
pub fn run_cse_pass(graph: &Graph) -> Result<(), Error> {
    let mut hash_to_node: HashMap<u64, petgraph::graph::NodeIndex> = HashMap::new();
    let mut replacements: Vec<(petgraph::graph::NodeIndex, petgraph::graph::NodeIndex)> =
        Vec::new();

    // Find duplicates
    let nodes: Vec<_> = {
        let inner = graph
            .inner
            .read()
            .expect("graph lock poisoned in CSE node collection");
        inner.dag.node_indices().collect()
    };

    for node in &nodes {
        let h = {
            let inner = graph
                .inner
                .read()
                .expect("graph lock poisoned in CSE node hashing");
            node_hash(&inner, *node)
        };
        if let Some(hash) = h {
            if let Some(&existing) = hash_to_node.get(&hash) {
                if existing != *node {
                    replacements.push((*node, existing));
                }
            } else {
                hash_to_node.insert(hash, *node);
            }
        }
    }

    if replacements.is_empty() {
        return Ok(());
    }

    // Apply replacements: replace all edges pointing to a node with edges to its duplicate
    let mut inner = graph
        .inner
        .write()
        .expect("graph lock poisoned in CSE replacement");
    for (old, new) in &replacements {
        // Find all nodes that reference `old` as input
        let consumers: Vec<_> = inner
            .dag
            .neighbors_directed(*old, petgraph::Direction::Outgoing)
            .map(|n| {
                let edge = inner
                    .dag
                    .find_edge(*old, n)
                    .expect("edge must exist between old node and its consumer");
                (
                    n,
                    *inner
                        .dag
                        .edge_weight(edge)
                        .expect("edge weight must exist on valid edge"),
                )
            })
            .collect();

        for (consumer, edge_weight) in consumers {
            // Remove old edge and add new edge
            let old_edge = inner
                .dag
                .find_edge(*old, consumer)
                .expect("edge must exist between old node and consumer");
            inner.dag.remove_edge(old_edge);
            inner.dag.add_edge(*new, consumer, edge_weight);
        }
    }

    Ok(())
}
