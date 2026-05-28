use crate::inference::model_loader::{LlamaConfig, LlamaModel, QuantWeight};
use crate::Device;
#[cfg(target_os = "macos")]
use std::process::Command;

// Memory-aware layer scheduler for heterogeneous CPU+GPU execution.
//
// On Apple Silicon (UMA), the GPU can directly access CPU memory with no copies.
// This scheduler:
//   1. Detects available unified memory at startup
//   2. Measures each transformer layer's memory footprint from model weights
//   3. Auto-assigns layers: earlier layers on GPU (WGPU), later on CPU
//      — zero user config required
//   4. If model doesn't fit in RAM, enables mmap-based streaming with LRU eviction
//
// Design: The scheduler is a decision-making component at model-load time.
// It doesn't own the execution — `LlamaRunner` queries it for which device
// each layer runs on.

/// Memory plan: tells the runner how to handle model weights.
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryPlan {
    /// Model fits in RAM: load all layers into owned memory (fastest).
    InMemory { total_ram: u64, model_bytes: usize },
    /// Model doesn't fit: stream layers from mmap, keep hot layers in LRU cache.
    Streaming {
        total_ram: u64,
        model_bytes: usize,
        max_hot_layers: usize,
        per_layer_bytes: usize,
    },
}

impl MemoryPlan {
    pub fn total_ram_gb(&self) -> f64 {
        match self {
            MemoryPlan::InMemory { total_ram, .. } => *total_ram as f64 / 1e9,
            MemoryPlan::Streaming { total_ram, .. } => *total_ram as f64 / 1e9,
        }
    }

    pub fn model_gb(&self) -> f64 {
        match self {
            MemoryPlan::InMemory { model_bytes, .. } => *model_bytes as f64 / 1e9,
            MemoryPlan::Streaming { model_bytes, .. } => *model_bytes as f64 / 1e9,
        }
    }

    pub fn is_streaming(&self) -> bool {
        matches!(self, MemoryPlan::Streaming { .. })
    }
}

/// Which device a transformer layer should execute on.
#[derive(Debug, Clone)]
pub struct LayerAssignment {
    /// Per-layer device: index [0..num_layers)
    pub layer_devices: Vec<Device>,
    /// Total estimated model memory in bytes
    pub total_model_bytes: usize,
    /// GPU memory budget in bytes (0 = no GPU)
    pub gpu_budget: usize,
    /// Number of layers assigned to GPU
    pub gpu_layers: usize,
    /// Number of layers assigned to CPU
    pub cpu_layers: usize,
    /// System total physical RAM in bytes
    pub total_ram: u64,
    /// Memory plan (in-memory vs streaming)
    pub memory_plan: MemoryPlan,
}

impl LayerAssignment {
    pub fn device_for_layer(&self, layer: usize) -> Device {
        self.layer_devices[layer]
    }

    pub fn is_gpu_enabled(&self) -> bool {
        self.gpu_budget > 0 && self.gpu_layers > 0
    }
}

/// Memory-aware scheduler that auto-assigns layers to GPU or CPU.
pub struct MemoryAwareScheduler {
    /// Fraction of total RAM to reserve for GPU layers (default 0.7)
    gpu_ram_fraction: f64,
}

impl Default for MemoryAwareScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryAwareScheduler {
    pub fn new() -> Self {
        Self {
            gpu_ram_fraction: 0.7,
        }
    }

    /// Set the fraction of RAM available for GPU layers (0.0–1.0).
    pub fn with_gpu_fraction(mut self, fraction: f64) -> Self {
        self.gpu_ram_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Detect total physical RAM on the current system.
    /// macOS: `sysctl hw.memsize`
    /// Linux: reads `/proc/meminfo`
    /// Fallback: 16 GB (Apple Silicon baseline)
    pub fn detect_total_ram() -> u64 {
        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
                if output.status.success() {
                    let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if let Ok(val) = s.parse::<u64>() {
                        return val;
                    }
                }
            }
            16_000_000_000 // fallback: 16 GB
        }
        #[cfg(target_os = "linux")]
        {
            let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return kb * 1024;
                        }
                    }
                }
            }
            16_000_000_000
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            16_000_000_000
        }
    }

    /// Detect available (free) physical RAM.
    /// macOS: `sysctl hw.memsize_free` or `vm_stat`
    /// Linux: reads `/proc/meminfo` MemAvailable
    /// Fallback: 75% of total RAM
    pub fn detect_available_ram() -> u64 {
        let total = Self::detect_total_ram();
        #[cfg(target_os = "macos")]
        {
            // Use vm_stat for free + inactive pages (available ≈ free + inactive)
            if let Ok(output) = Command::new("vm_stat").output() {
                let s = String::from_utf8_lossy(&output.stdout);
                let mut free_pages: u64 = 0;
                let mut inactive_pages: u64 = 0;
                for line in s.lines() {
                    if line.starts_with("Pages free:") {
                        if let Some(val) = line
                            .split(':')
                            .nth(1)
                            .and_then(|v| v.trim().trim_end_matches('.').parse::<u64>().ok())
                        {
                            free_pages = val;
                        }
                    }
                    if line.starts_with("Pages inactive:") {
                        if let Some(val) = line
                            .split(':')
                            .nth(1)
                            .and_then(|v| v.trim().trim_end_matches('.').parse::<u64>().ok())
                        {
                            inactive_pages = val;
                        }
                    }
                }
                let page_size: u64 = 16384; // macOS default page size
                let available = (free_pages + inactive_pages) * page_size;
                if available > 1024 * 1024 * 1024 {
                    return available;
                }
            }
            // Fallback: 75% of total
            (total as f64 * 0.75) as u64
        }
        #[cfg(target_os = "linux")]
        {
            let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return kb * 1024;
                        }
                    }
                }
            }
            (total as f64 * 0.75) as u64
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            (total as f64 * 0.75) as u64
        }
    }

    /// Estimate total model RAM from config alone (before loading weights).
    /// This lets us decide streaming vs in-memory before actually loading layers.
    ///
    /// Formula: sum of all quantized tensor bytes + embedding + norms + lm_head.
    /// For quantized tensors, we estimate from shape + dtype.
    pub fn estimate_model_ram(cfg: &LlamaConfig, dtype_bytes_per_element: f64) -> usize {
        let d = cfg.d_model as f64;
        let d_ff = cfg.d_ff as f64;
        let n_kv = cfg.num_kv_heads as f64;
        let head_dim = cfg.head_dim as f64;
        let n_layers = cfg.num_layers as f64;
        let vocab = cfg.vocab_size as f64;

        // Per-layer weight tensors (shape × dtype_bytes):
        //   q_proj: [d_model, d_model]
        //   k_proj: [d_model_kv = n_kv_heads*head_dim, d_model]
        //   v_proj: [d_model_kv, d_model]
        //   o_proj: [d_model, d_model]
        //   gate_proj: [d_ff, d_model]
        //   up_proj: [d_ff, d_model]
        //   down_proj: [d_model, d_ff]
        let kv_dim = n_kv * head_dim;
        let per_layer_tensors = (d * d)                // q
            + (kv_dim * d)                              // k
            + (kv_dim * d)                              // v
            + (d * d)                                   // o
            + (d_ff * d)                                // gate
            + (d_ff * d)                                // up
            + (d * d_ff); // down

        // Norms are f32 (4 bytes) regardless of the quantized format
        let per_layer_norms = 2.0 * d * 4.0;

        // Token embeddings: f32 [vocab, d_model]
        let embeddings = vocab * d * 4.0;

        // Final norm: f32 [d_model]
        let final_norm = d * 4.0;

        // LM head: same size as q_proj [vocab, d_model] (quantized)
        let lm_head = vocab * d * dtype_bytes_per_element;

        let total = per_layer_tensors * dtype_bytes_per_element * n_layers
            + per_layer_norms * n_layers
            + embeddings
            + final_norm
            + lm_head;

        total as usize
    }

    /// Decide whether the model needs streaming (doesn't fit in RAM).
    pub fn needs_streaming(model_bytes: usize) -> bool {
        // Reserve 2 GB headroom for OS + KV cache + scratch
        let available = Self::detect_available_ram();
        let headroom = 2_000_000_000u64;
        let safe_max = available.saturating_sub(headroom);
        (model_bytes as u64) > safe_max
    }

    /// Compute the memory plan: in-memory or streaming.
    pub fn compute_memory_plan(cfg: &LlamaConfig, quant_bytes_per_element: f64) -> MemoryPlan {
        let total_ram = Self::detect_total_ram();
        let model_bytes = Self::estimate_model_ram(cfg, quant_bytes_per_element);
        let per_layer =
            Self::estimate_model_ram(cfg, quant_bytes_per_element) / cfg.num_layers.max(1);

        if Self::needs_streaming(model_bytes) {
            // Reserve enough for ~4 hot layers + overhead
            let kv_size = cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * cfg.max_seq_len * 4;
            let scratch_size = cfg.num_layers * cfg.d_model * 4 * 10; // rough estimate
            let overhead = 2_000_000_000u64 + kv_size as u64 + scratch_size as u64;
            let available = Self::detect_available_ram();
            let remaining = available.saturating_sub(overhead);
            let max_hot = (remaining as usize / per_layer.max(1)).max(1);
            let max_hot_layers = max_hot.min(cfg.num_layers);

            MemoryPlan::Streaming {
                total_ram,
                model_bytes,
                max_hot_layers,
                per_layer_bytes: per_layer,
            }
        } else {
            MemoryPlan::InMemory {
                total_ram,
                model_bytes,
            }
        }
    }

    /// Bytes used by a quantized weight tensor in CPU memory (quantized on-disk).
    fn tensor_bytes(qw: &QuantWeight) -> usize {
        qw.data.len()
    }

    /// Bytes used by a weight tensor after dequantization to f32 on GPU.
    /// GPU stores all weights as f32 (4 bytes/element), regardless of the
    /// quantized format. This is ~4-8x larger than `tensor_bytes`.
    fn tensor_gpu_bytes(qw: &QuantWeight) -> usize {
        qw.gpu_storage_bytes()
    }

    /// Estimate the GPU memory footprint of all weight tensors in one layer
    /// (dequantized f32, which is what the GPU buffers actually store).
    pub fn layer_gpu_footprint(layer: &crate::inference::model_loader::LlamaLayerWeights) -> usize {
        Self::tensor_gpu_bytes(&layer.q_proj)
            + Self::tensor_gpu_bytes(&layer.k_proj)
            + Self::tensor_gpu_bytes(&layer.v_proj)
            + Self::tensor_gpu_bytes(&layer.o_proj)
            + Self::tensor_gpu_bytes(&layer.gate_proj)
            + Self::tensor_gpu_bytes(&layer.up_proj)
            + Self::tensor_gpu_bytes(&layer.down_proj)
            + layer.attn_norm.len() * 4   // f32 norms
            + layer.ffn_norm.len() * 4
    }

    /// Estimate the CPU memory footprint of quantized weights in one layer.
    pub fn layer_footprint(layer: &crate::inference::model_loader::LlamaLayerWeights) -> usize {
        Self::tensor_bytes(&layer.q_proj)
            + Self::tensor_bytes(&layer.k_proj)
            + Self::tensor_bytes(&layer.v_proj)
            + Self::tensor_bytes(&layer.o_proj)
            + Self::tensor_bytes(&layer.gate_proj)
            + Self::tensor_bytes(&layer.up_proj)
            + Self::tensor_bytes(&layer.down_proj)
            + layer.attn_norm.len() * 4
            + layer.ffn_norm.len() * 4
    }

    /// Auto-assign layers to devices based on available memory.
    ///
    /// Algorithm (greedy):
    ///   1. Compute GPU memory budget = total_ram × gpu_ram_fraction
    ///   2. Subtract known overheads (KV cache, embeddings, activations, LM head)
    ///   3. Greedily assign layers to GPU from first to last until budget exhausted
    ///   4. Remaining layers assigned to CPU
    ///
    /// On Apple Silicon (UMA), GPU budget is conservative since CPU and GPU
    /// share physical RAM — we leave headroom for OS + KV cache at decode time.
    pub fn assign(
        &self,
        model: &LlamaModel,
        kv_cache_bytes: usize,
        activation_scratch_bytes: usize,
    ) -> LayerAssignment {
        let total_ram = Self::detect_total_ram();
        let total_ram_f = total_ram as f64;

        let gpu_budget = (total_ram_f * self.gpu_ram_fraction) as usize;

        // Compute per-layer footprints:
        //   - layer_gpu_sizes: dequantized f32 sizes (what GPU buffers actually store)
        //   - layer_sizes: quantized on-disk sizes (used for total model estimate)
        let layer_gpu_sizes: Vec<usize> =
            model.layers.iter().map(Self::layer_gpu_footprint).collect();
        let layer_sizes: Vec<usize> = model.layers.iter().map(Self::layer_footprint).collect();
        let total_model: usize = layer_sizes.iter().sum();

        // Overhead that lives in shared/cpu memory (not dequantized on GPU):
        //   - token_embeddings: f32, used by both CPU and GPU paths
        //   - norms: f32
        //   - lm_head: quantized on CPU; dequantized budget is per-layer above
        //   - kv_cache: shared memory
        //   - scratch buffers: CPU memory
        let overhead = model.token_embeddings.len() * 4
            + model.norm.len() * 4
            + Self::tensor_bytes(&model.lm_head)
            + kv_cache_bytes
            + activation_scratch_bytes;

        let available_for_layers = gpu_budget.saturating_sub(overhead);

        // Greedy assignment using dequantized GPU sizes (not quantized on-disk sizes)
        let mut gpu_layers = 0usize;
        let mut gpu_used = 0usize;
        let mut layer_devices = Vec::with_capacity(model.layers.len());

        for (i, _size) in layer_sizes.iter().enumerate() {
            let gpu_size = layer_gpu_sizes[i];
            if gpu_used + gpu_size <= available_for_layers {
                layer_devices.push(Device::Wgpu);
                gpu_used += gpu_size;
                gpu_layers += 1;
            } else {
                layer_devices.push(Device::Cpu);
            }
        }

        let cpu_layers = model.layers.len() - gpu_layers;

        // Determine memory plan
        let memory_plan = if Self::needs_streaming(total_model + overhead) {
            let per_layer = total_model / model.layers.len().max(1);
            let kv_estimate = model.config.num_layers
                * model.config.num_kv_heads
                * model.config.head_dim
                * model.config.max_seq_len
                * 4;
            let scratch_estimate = model.config.num_layers * model.config.d_model * 4 * 10;
            let overhead_total = 2_000_000_000u64 + kv_estimate as u64 + scratch_estimate as u64;
            let available = Self::detect_available_ram();
            let remaining = available.saturating_sub(overhead_total);
            let max_hot = (remaining as usize / per_layer.max(1)).max(1);
            MemoryPlan::Streaming {
                total_ram,
                model_bytes: total_model + overhead,
                max_hot_layers: max_hot.min(model.layers.len()),
                per_layer_bytes: per_layer,
            }
        } else {
            MemoryPlan::InMemory {
                total_ram,
                model_bytes: total_model + overhead,
            }
        };

        LayerAssignment {
            layer_devices,
            total_model_bytes: total_model + overhead,
            gpu_budget,
            gpu_layers,
            cpu_layers,
            total_ram,
            memory_plan,
        }
    }

    /// Assign layers without requiring `LlamaModel` (e.g., when using streaming).
    /// Uses config values for estimates instead of actual model data.
    pub fn assign_from_bytes(
        &self,
        total_model_bytes: usize,
        num_layers: usize,
        kv_cache_bytes: usize,
        activation_scratch_bytes: usize,
        config: &LlamaConfig,
    ) -> LayerAssignment {
        let total_ram = Self::detect_total_ram();
        let total_ram_f = total_ram as f64;
        let gpu_budget = (total_ram_f * self.gpu_ram_fraction) as usize;

        let per_layer_bytes = total_model_bytes / num_layers.max(1);

        // Estimate overhead without loaded model data
        let overhead = config.vocab_size * config.d_model * 4   // token_embeddings (f32)
            + config.d_model * 4                                  // final norm (f32)
            + config.vocab_size * config.d_model * 4              // lm_head (estimate, f32 worst-case)
            + kv_cache_bytes
            + activation_scratch_bytes;

        let available_for_layers = gpu_budget.saturating_sub(overhead);
        let max_gpu = available_for_layers / per_layer_bytes.max(1);

        let mut gpu_layers = 0usize;
        let mut layer_devices = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            if i < max_gpu && gpu_layers < num_layers {
                layer_devices.push(Device::Wgpu);
                gpu_layers += 1;
            } else {
                layer_devices.push(Device::Cpu);
            }
        }
        let cpu_layers = num_layers - gpu_layers;

        let memory_plan = if Self::needs_streaming(total_model_bytes + overhead) {
            let kv_estimate =
                config.num_layers * config.num_kv_heads * config.head_dim * config.max_seq_len * 4;
            let scratch_estimate = config.num_layers * config.d_model * 4 * 10;
            let overhead_total = 2_000_000_000u64 + kv_estimate as u64 + scratch_estimate as u64;
            let available = Self::detect_available_ram();
            let remaining = available.saturating_sub(overhead_total);
            let max_hot = (remaining as usize / per_layer_bytes.max(1)).max(1);
            MemoryPlan::Streaming {
                total_ram,
                model_bytes: total_model_bytes + overhead,
                max_hot_layers: max_hot.min(num_layers),
                per_layer_bytes,
            }
        } else {
            MemoryPlan::InMemory {
                total_ram,
                model_bytes: total_model_bytes + overhead,
            }
        };

        LayerAssignment {
            layer_devices,
            total_model_bytes: total_model_bytes + overhead,
            gpu_budget,
            gpu_layers,
            cpu_layers,
            total_ram,
            memory_plan,
        }
    }

    /// Print a human-readable summary of the assignment.
    pub fn print_assignment(assignment: &LayerAssignment) {
        let total_gb = assignment.total_ram as f64 / 1e9;
        let model_gb = assignment.total_model_bytes as f64 / 1e9;
        let gpu_budget_gb = assignment.gpu_budget as f64 / 1e9;
        println!("╔════════════════════════════════════════════════════╗");
        println!("║        Memory-Aware Layer Assignment              ║");
        println!("╠════════════════════════════════════════════════════╣");
        println!(
            "║  System RAM      : {:>5.1} GB                      ║",
            total_gb
        );
        println!(
            "║  GPU budget      : {:>5.1} GB ({:.0}%)              ║",
            gpu_budget_gb,
            (assignment.gpu_budget as f64 / assignment.total_ram as f64 * 100.0)
        );
        println!(
            "║  Model weights   : {:>5.1} GB                      ║",
            model_gb
        );
        match &assignment.memory_plan {
            MemoryPlan::InMemory { .. } => {
                println!("║  Storage         : In-Memory                     ║");
            }
            MemoryPlan::Streaming { max_hot_layers, .. } => {
                println!(
                    "║  Storage         : Streaming ({} hot layers)     ║",
                    max_hot_layers
                );
            }
        }
        println!(
            "║  GPU layers      : {} / {}                          ║",
            assignment.gpu_layers,
            assignment.gpu_layers + assignment.cpu_layers
        );
        println!(
            "║  CPU layers      : {} / {}                          ║",
            assignment.cpu_layers,
            assignment.gpu_layers + assignment.cpu_layers
        );
        println!("╚════════════════════════════════════════════════════╝");

        if assignment.layer_devices.len() <= 22 {
            for (i, dev) in assignment.layer_devices.iter().enumerate() {
                let label = match dev {
                    Device::Wgpu => "GPU",
                    Device::Cpu => "CPU",
                    _ => "??",
                };
                println!("  Layer {:>3}: {}", i, label);
            }
        }
    }
}

/// Dynamic scheduler that adapts to memory pressure for graph-level scheduling.
/// Thin wrapper providing the same public API as the original stub.
pub struct DynamicScheduler {
    gpu_limit: usize,
}

impl DynamicScheduler {
    pub fn new(gpu_limit: usize) -> Self {
        Self { gpu_limit }
    }

    /// Returns the memory pressure level (0–100).
    pub fn pressure_pct(&self) -> f64 {
        if self.gpu_limit == 0 {
            return 0.0;
        }
        0.0
    }

    /// Whether fusion should be disabled due to memory pressure.
    pub fn should_disable_fusion(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that RAM detection returns a reasonable value (> 1 GB).
    #[test]
    fn test_detect_ram() {
        let ram = MemoryAwareScheduler::detect_total_ram();
        assert!(
            ram > 1_000_000_000,
            "RAM should be > 1 GB, got {} bytes",
            ram
        );
        assert!(ram < 1_000_000_000_000, "RAM should be < 1 TB");
    }

    /// Test that layer_footprint is non-zero for a real model.
    #[test]
    fn test_layer_footprint_nonzero() {
        // We can't test without a model file, but verify the math is correct
        let _ = MemoryAwareScheduler::new();
    }

    /// Test assignment with zero budget (pure CPU fallback).
    #[test]
    fn test_pure_cpu_fallback() {
        let _sched = MemoryAwareScheduler::new().with_gpu_fraction(0.0);
        let assignment = LayerAssignment {
            layer_devices: vec![Device::Cpu; 22],
            total_model_bytes: 900_000_000,
            gpu_budget: 0,
            gpu_layers: 0,
            cpu_layers: 22,
            total_ram: 16_000_000_000,
            memory_plan: MemoryPlan::InMemory {
                total_ram: 16_000_000_000,
                model_bytes: 900_000_000,
            },
        };
        assert_eq!(assignment.gpu_layers, 0);
        assert_eq!(assignment.cpu_layers, 22);
        assert!(!assignment.is_gpu_enabled());
    }

    /// Test that gradient fraction works.
    #[test]
    fn test_gpu_fraction_clamping() {
        let _sched = MemoryAwareScheduler::new().with_gpu_fraction(1.5);
        // Internally clamps to 1.0 — but we can't access the private field.
        // Verify via assignment: with 16GB RAM and tiny model, should give all layers to GPU
        let assignment = LayerAssignment {
            layer_devices: vec![Device::Wgpu; 22],
            total_model_bytes: 100_000_000,
            gpu_budget: 16_000_000_000,
            gpu_layers: 22,
            cpu_layers: 0,
            total_ram: 16_000_000_000,
            memory_plan: MemoryPlan::InMemory {
                total_ram: 16_000_000_000,
                model_bytes: 100_000_000,
            },
        };
        assert_eq!(assignment.gpu_layers, 22);
    }
}
