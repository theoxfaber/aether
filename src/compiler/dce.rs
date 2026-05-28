use crate::graph::Graph;
use crate::Error;
use petgraph::visit::EdgeRef;

/// Dead Code Elimination pass.
/// Removes nodes that are not reachable from any output node.
pub fn run_dce_pass(graph: &Graph) -> Result<(), Error> {
    let live_nodes = {
        let inner = graph
            .inner
            .read()
            .expect("graph lock poisoned in DCE live node detection");
        let dag = &inner.dag;

        // Find all output nodes (nodes with no outgoing edges)
        let outputs: Vec<_> = dag
            .node_indices()
            .filter(|&n| {
                dag.neighbors_directed(n, petgraph::Direction::Outgoing)
                    .count()
                    == 0
            })
            .collect();

        if outputs.is_empty() {
            return Ok(());
        }

        // Reverse BFS from all output nodes to find live nodes
        let mut live = std::collections::HashSet::new();
        let mut stack = outputs;
        while let Some(n) = stack.pop() {
            if !live.insert(n) {
                continue;
            }
            for edge in dag.edges_directed(n, petgraph::Direction::Incoming) {
                stack.push(edge.source());
            }
        }
        live
    };

    // Remove nodes that are not live (in reverse order to maintain indices)
    let mut nodes_to_remove: Vec<_> = {
        let inner = graph
            .inner
            .read()
            .expect("graph lock poisoned in DCE node collection");
        inner
            .dag
            .node_indices()
            .filter(|n| !live_nodes.contains(n))
            .collect()
    };
    nodes_to_remove.sort_by(|a, b| b.cmp(a)); // reverse order

    let mut inner = graph
        .inner
        .write()
        .expect("graph lock poisoned in DCE node removal");
    for node in nodes_to_remove {
        inner.dag.remove_node(node);
    }

    Ok(())
}
