use crate::graph::Graph;
use crate::scheduler::ScheduledOp;
use crate::tensor::TensorId;
use crate::Device;
#[derive(Debug, Clone)]
pub struct PrefetchOp {
    pub tensor_id: TensorId,
    pub target_device: Device,
    pub estimated_bytes: usize,
}

/// Asynchronous prefetch planner.
///
/// Given a schedule, identifies tensors needed by future ops and
/// prefetches them before they are needed, overlapping transfer with compute.
pub struct AsyncPrefetchScheduler;

impl AsyncPrefetchScheduler {
    pub fn plan(schedule: &[ScheduledOp], graph: &Graph) -> Vec<PrefetchOp> {
        let mut prefetches = Vec::new();
        let inner = graph.inner.read().unwrap();
        for (i, _op) in schedule.iter().enumerate() {
            if i + 2 < schedule.len() {
                if let ScheduledOp::Plain(node_id, _) = &schedule[i + 2] {
                    let tid = inner
                        .dag
                        .node_weight(*node_id)
                        .map(|n| n.tensor_id)
                        .unwrap_or_else(TensorId::next);
                    prefetches.push(PrefetchOp {
                        tensor_id: tid,
                        target_device: Device::Wgpu,
                        estimated_bytes: 1024,
                    });
                }
            }
        }
        prefetches
    }
}
