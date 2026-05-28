use crate::{Error, Graph, GraphTensor, Shape, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};

/// A serialized tensor structure inside the versioned weight format.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SerializedTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// The current versioned weight schema format.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VersionedWeights {
    pub version: u32,
    pub metadata: HashMap<String, String>,
    pub weights: HashMap<String, SerializedTensor>,
}

/// Save a set of named weights to a file in versioned JSON format.
pub fn save_weights(weights: &HashMap<String, Tensor>, path: &str) -> Result<(), Error> {
    let mut serialized_weights = HashMap::new();
    for (name, tensor) in weights {
        serialized_weights.insert(
            name.clone(),
            SerializedTensor {
                data: tensor.data().to_vec(),
                shape: tensor.shape().dims().to_vec(),
            },
        );
    }

    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "aether-json".to_string());
    metadata.insert("version".to_string(), "2".to_string());

    let versioned = VersionedWeights {
        version: 2,
        metadata,
        weights: serialized_weights,
    };

    let json_str = serde_json::to_string_pretty(&versioned)
        .map_err(|e| Error::ExecutionError(format!("Failed to serialize weights: {:?}", e)))?;
    let mut file = File::create(path)
        .map_err(|e| Error::ExecutionError(format!("Failed to create weights file: {:?}", e)))?;
    file.write_all(json_str.as_bytes())
        .map_err(|e| Error::ExecutionError(format!("Failed to write weights file: {:?}", e)))?;
    Ok(())
}

/// Load a set of named weights from a JSON file, supporting both versioned and legacy formats.
pub fn load_weights(path: &str) -> Result<HashMap<String, Tensor>, Error> {
    let mut file = File::open(path)
        .map_err(|e| Error::ExecutionError(format!("Failed to open weights file: {:?}", e)))?;
    let mut json_str = String::new();
    file.read_to_string(&mut json_str)
        .map_err(|e| Error::ExecutionError(format!("Failed to read weights file: {:?}", e)))?;

    // Try parsing as versioned weights first
    if let Ok(versioned) = serde_json::from_str::<VersionedWeights>(&json_str) {
        let mut weights = HashMap::new();
        for (name, st) in versioned.weights {
            weights.insert(name, Tensor::new(st.data, Shape::new(st.shape)));
        }
        return Ok(weights);
    }

    // Legacy fallback: deserialize as HashMap<String, (Vec<f32>, Vec<usize>)>
    let serialized_map: HashMap<String, (Vec<f32>, Vec<usize>)> = serde_json::from_str(&json_str)
        .map_err(|e| {
        Error::ExecutionError(format!(
            "Failed to deserialize weights (tried versioned and legacy format): {:?}",
            e
        ))
    })?;

    let mut weights = HashMap::new();
    for (name, (data, dims)) in serialized_map {
        weights.insert(name, Tensor::new(data, Shape::new(dims)));
    }
    Ok(weights)
}

/// Load saved weights directly into the corresponding parameter nodes in a Graph.
pub fn load_weights_into_graph(
    graph: &Graph,
    weights: &HashMap<String, Tensor>,
    parameter_nodes: &HashMap<String, GraphTensor>,
) {
    for (name, tensor) in weights {
        if let Some(param) = parameter_nodes.get(name) {
            graph.update_input(param.id(), tensor.data().to_vec());
        }
    }
}

// ─────────────────────────────────────────────
//  Training-state Checkpointing
// ─────────────────────────────────────────────

/// Serialized AdamW optimizer state for a single parameter.
/// Keyed by the string representation of the `TensorId` in `TrainingCheckpoint`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SerializedAdamWState {
    /// First moment estimate (m).
    pub m: Vec<f32>,
    /// Second moment estimate (v).
    pub v: Vec<f32>,
}

/// A complete training checkpoint, bundling model weights, AdamW optimizer
/// state, the optimizer step counter, and the current epoch.
///
/// # File Layout (JSON)
/// ```json
/// {
///   "schema_version": 1,
///   "epoch": 3,
///   "optimizer_step": 300,
///   "weights": { /* VersionedWeights */ },
///   "adamw_states": { "<tensor_id>": { "m": [...], "v": [...] } }
/// }
/// ```
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrainingCheckpoint {
    /// Schema version — bump when the layout changes in a breaking way.
    pub schema_version: u32,
    /// Number of completed training epochs at save time.
    pub epoch: u64,
    /// Total optimizer steps taken at save time.
    pub optimizer_step: u64,
    /// Model weights in `VersionedWeights` format (supports legacy fallback on
    /// individual weight deserialization via `load_weights`).
    pub weights: VersionedWeights,
    /// AdamW per-parameter moment states, keyed by `TensorId` as a string.
    /// Missing entries are silently skipped on restore — safe for partial
    /// checkpoints or when adding new parameters.
    pub adamw_states: HashMap<String, SerializedAdamWState>,
}

/// Save a full training checkpoint to `path`.
///
/// # Arguments
/// * `weights`        – Named tensors (model parameters).
/// * `optimizer`      – The `AdamW` optimizer to snapshot.
/// * `param_nodes`    – Map of parameter name → `GraphTensor`, used to resolve
///   `TensorId`s for optimizer state keys.
/// * `epoch`          – Current epoch number (0-indexed or 1-indexed, caller's choice).
/// * `path`           – Destination file path (written atomically via
///   create-then-rename on POSIX systems).
pub fn save_checkpoint(
    weights: &HashMap<String, Tensor>,
    optimizer: &crate::optimizer::AdamW,
    param_nodes: &HashMap<String, GraphTensor>,
    epoch: u64,
    path: &str,
) -> Result<(), Error> {
    // Serialize weights
    let mut serialized_weights = HashMap::new();
    for (name, tensor) in weights {
        serialized_weights.insert(
            name.clone(),
            SerializedTensor {
                data: tensor.data().to_vec(),
                shape: tensor.shape().dims().to_vec(),
            },
        );
    }
    let mut meta = HashMap::new();
    meta.insert("format".to_string(), "aether-checkpoint".to_string());
    let versioned_weights = VersionedWeights {
        version: 2,
        metadata: meta,
        weights: serialized_weights,
    };

    // Serialize AdamW states
    let mut adamw_states: HashMap<String, SerializedAdamWState> = HashMap::new();
    for (name, param_gt) in param_nodes {
        let tid = param_gt.tensor_id();
        let m = optimizer.get_m(tid).cloned().unwrap_or_default();
        let v = optimizer.get_v(tid).cloned().unwrap_or_default();
        if !m.is_empty() || !v.is_empty() {
            adamw_states.insert(format!("{:?}", tid), SerializedAdamWState { m, v });
        }
        let _ = name; // param name reserved for future human-readable keying
    }

    let checkpoint = TrainingCheckpoint {
        schema_version: 1,
        epoch,
        optimizer_step: optimizer.step_count() as u64,
        weights: versioned_weights,
        adamw_states,
    };

    let json_str = serde_json::to_string_pretty(&checkpoint)
        .map_err(|e| Error::ExecutionError(format!("Failed to serialize checkpoint: {:?}", e)))?;

    // Write to a temp file, then rename for atomicity
    let tmp_path = format!("{}.tmp", path);
    {
        let mut file = File::create(&tmp_path).map_err(|e| {
            Error::ExecutionError(format!("Failed to create checkpoint tmp file: {:?}", e))
        })?;
        file.write_all(json_str.as_bytes())
            .map_err(|e| Error::ExecutionError(format!("Failed to write checkpoint: {:?}", e)))?;
    }
    std::fs::rename(&tmp_path, path)
        .map_err(|e| Error::ExecutionError(format!("Failed to rename checkpoint file: {:?}", e)))?;

    Ok(())
}

/// Load a training checkpoint from `path`.
///
/// Returns the `TrainingCheckpoint` struct. The caller is responsible for:
/// 1. Calling `load_weights_into_graph` to restore parameter tensors.
/// 2. Calling `optimizer.set_moments(tid, m, v)` and `optimizer.set_step_count(n)`
///    to restore the optimizer state.
pub fn load_checkpoint(path: &str) -> Result<TrainingCheckpoint, Error> {
    let mut file = File::open(path)
        .map_err(|e| Error::ExecutionError(format!("Failed to open checkpoint file: {:?}", e)))?;
    let mut json_str = String::new();
    file.read_to_string(&mut json_str)
        .map_err(|e| Error::ExecutionError(format!("Failed to read checkpoint file: {:?}", e)))?;

    let checkpoint: TrainingCheckpoint = serde_json::from_str(&json_str)
        .map_err(|e| Error::ExecutionError(format!("Failed to deserialize checkpoint: {:?}", e)))?;

    Ok(checkpoint)
}

/// Restore model weights from a checkpoint into the graph.
/// Convenience wrapper around `load_weights_into_graph` for checkpoint payloads.
pub fn restore_weights_from_checkpoint(
    graph: &Graph,
    checkpoint: &TrainingCheckpoint,
    parameter_nodes: &HashMap<String, GraphTensor>,
) {
    let weights: HashMap<String, Tensor> = checkpoint
        .weights
        .weights
        .iter()
        .map(|(k, st)| {
            (
                k.clone(),
                Tensor::new(st.data.clone(), Shape::new(st.shape.clone())),
            )
        })
        .collect();
    load_weights_into_graph(graph, &weights, parameter_nodes);
}
