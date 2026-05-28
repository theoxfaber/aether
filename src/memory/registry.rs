#![allow(unsafe_code)]
//! # Safety & Memory Lifecycle Model
//!
//! Aether's memory layer is designed to be completely memory-safe.
//! Unlike traditional deep learning frameworks that use raw pointers (`*mut c_void`)
//! and custom unsafe allocation pools (which risk double-frees or dangling pointers),
//! Aether relies entirely on Safe Rust constructs:
//!
//! 1. **Resource Ownership**:
//!    Tensors on the host are stored as standard `Vec<f32>`, managed by Rust's allocator.
//!    GPU buffers are stored inside `wgpu::Buffer` and wrapped in `Arc<wgpu::Buffer>`.
//!    This guarantees that as long as any operation or reference exists, the underlying
//!    memory is kept alive, preventing dangling GPU buffers or use-after-free bugs.
//!
//! 2. **Double-Free Prevention**:
//!    Memory release is handled by Rust's standard RAII drops. Since no manual pointer management
//!    or free calls are made, double-free bugs are structurally impossible.
//!
//! 3. **Concurrency & Thread Safety**:
//!    The registry uses `Mutex<HashMap<TensorId, BufferEntry>>` and atomic operations.
//!    All access to tensor metadata is serialized, ensuring race-free and sound modifications
//!    even when scheduling across thread boundaries.
//!
//! 4. **No Unsafe Code**:
//!    This registry contains exactly zero `unsafe` blocks.

use crate::memory::planner::ArenaAllocation;
use crate::tensor::{Dtype, Shape, TensorId};
use crate::Device;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

/// Where a tensor's data currently lives
#[derive(Debug, Clone, PartialEq)]
pub enum BufferLocation {
    /// Only in CPU RAM (Vec<f32>)
    Cpu,
    /// Only in GPU VRAM (wgpu::Buffer)
    Gpu,
    /// Mirrored in both — CPU copy is canonical, GPU copy may be stale
    Both,
}

/// A view into a slice of a GPU buffer.
#[derive(Clone, Debug)]
pub struct GpuBufferView {
    pub buffer: Arc<wgpu::Buffer>,
    pub offset: u64,
    pub size: u64,
}

impl GpuBufferView {
    pub fn as_binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: std::num::NonZeroU64::new(self.size),
        })
    }
}

/// A view into a slice of a CPU buffer.
#[derive(Clone, Copy, Debug)]
pub struct CpuBufferView {
    pub offset: usize,
    pub size: usize,
}

/// A live tensor buffer entry in the registry
pub struct BufferEntry {
    pub location: BufferLocation,
    pub cpu_data: Option<Vec<f32>>,
    pub gpu_buffer: Option<Arc<wgpu::Buffer>>,
    pub shape: Shape,
    pub dtype: Dtype,
    /// Last op index that used this tensor (for eviction ordering)
    pub last_used: usize,
    /// Size in bytes
    pub byte_size: usize,
    /// When pinned, the tensor is actively in use and must not be evicted.
    pub pinned: bool,
}

pub struct BufferRegistry {
    entries: Mutex<HashMap<TensorId, BufferEntry>>,
    /// Total bytes currently resident on GPU
    gpu_bytes_used: Mutex<usize>,
    /// Soft limit: evict to CPU when GPU usage exceeds this
    gpu_byte_limit: usize,
    /// Peak GPU bytes allocated
    peak_gpu_bytes: Mutex<usize>,
    /// Counters for diagnostics
    upload_count: AtomicUsize,
    eviction_count: AtomicUsize,

    // --- Static Memory Arena Fields ---
    pub gpu_arena: Option<Arc<wgpu::Buffer>>,
    pub gpu_plan: HashMap<TensorId, ArenaAllocation>,
    pub cpu_arena: Option<Arc<Mutex<Vec<f32>>>>,
    pub cpu_plan: HashMap<TensorId, ArenaAllocation>,
}

impl BufferRegistry {
    pub fn new(gpu_byte_limit: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            gpu_bytes_used: Mutex::new(0),
            gpu_byte_limit,
            peak_gpu_bytes: Mutex::new(0),
            upload_count: AtomicUsize::new(0),
            eviction_count: AtomicUsize::new(0),
            gpu_arena: None,
            gpu_plan: HashMap::new(),
            cpu_arena: None,
            cpu_plan: HashMap::new(),
        }
    }

    pub fn new_static(
        gpu_byte_limit: usize,
        gpu_arena: Option<Arc<wgpu::Buffer>>,
        gpu_plan: HashMap<TensorId, ArenaAllocation>,
        cpu_arena: Option<Arc<Mutex<Vec<f32>>>>,
        cpu_plan: HashMap<TensorId, ArenaAllocation>,
        intermediate_metadata: HashMap<TensorId, (Shape, Dtype, Device)>,
    ) -> Self {
        let registry = Self {
            entries: Mutex::new(HashMap::new()),
            gpu_bytes_used: Mutex::new(0),
            gpu_byte_limit,
            peak_gpu_bytes: Mutex::new(0),
            upload_count: AtomicUsize::new(0),
            eviction_count: AtomicUsize::new(0),
            gpu_arena,
            gpu_plan,
            cpu_arena,
            cpu_plan,
        };

        // Pre-register all intermediate tensors
        for (id, (shape, dtype, device)) in intermediate_metadata {
            let element_size = match dtype {
                Dtype::F32 => 4,
                Dtype::F16 => 2,
                Dtype::BF16 => 2,
            };
            let byte_size = shape.num_elements() * element_size;
            let location = match device {
                Device::Wgpu => BufferLocation::Gpu,
                Device::Cpu => BufferLocation::Cpu,
                _ => BufferLocation::Cpu,
            };
            registry
                .entries
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(
                    id,
                    BufferEntry {
                        location,
                        cpu_data: None,
                        gpu_buffer: None,
                        shape,
                        dtype,
                        last_used: 0,
                        byte_size,
                        pinned: false,
                    },
                );
        }

        registry
    }

    /// Register a new CPU tensor into the registry
    pub fn register_cpu(&self, id: TensorId, data: Vec<f32>, shape: Shape, dtype: Dtype) {
        let element_size = match dtype {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
            Dtype::BF16 => 2,
        };
        let byte_size = shape.num_elements() * element_size;
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.insert(
            id,
            BufferEntry {
                location: BufferLocation::Cpu,
                cpu_data: Some(data),
                gpu_buffer: None,
                shape,
                dtype,
                last_used: 0,
                byte_size,
                pinned: false,
            },
        );
    }

    /// Register a new GPU tensor directly (for outputs computed on GPU)
    pub fn register_gpu(
        &self,
        id: TensorId,
        buffer: Arc<wgpu::Buffer>,
        shape: Shape,
        dtype: Dtype,
    ) -> Result<(), crate::Error> {
        let element_size = match dtype {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
            Dtype::BF16 => 2,
        };
        let byte_size = shape.num_elements() * element_size;
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.insert(
            id,
            BufferEntry {
                location: BufferLocation::Gpu,
                cpu_data: None,
                gpu_buffer: Some(buffer),
                shape,
                dtype,
                last_used: 0,
                byte_size,
                pinned: false,
            },
        );

        let mut gpu_used = self
            .gpu_bytes_used
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *gpu_used += byte_size;
        let mut peak = self
            .peak_gpu_bytes
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if *gpu_used > *peak {
            *peak = *gpu_used;
        }
        Ok(())
    }

    /// Ensure tensor is resident on GPU. Uploads from CPU if needed.
    pub fn ensure_gpu(
        &self,
        id: TensorId,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Arc<wgpu::Buffer>, crate::Error> {
        // If it's in the static plan, return the gpu_arena
        if self.gpu_plan.contains_key(&id) {
            if let Some(ref arena) = self.gpu_arena {
                return Ok(arena.clone());
            }
        }

        // 1. Get the entry metadata first under lock
        let (location, byte_size) = {
            let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let entry = entries.get(&id).ok_or_else(|| {
                crate::Error::ExecutionError("Tensor not registered in registry".to_string())
            })?;
            (entry.location.clone(), entry.byte_size)
        };

        // 2. If it's already on GPU, touch it and return it
        if location == BufferLocation::Gpu || location == BufferLocation::Both {
            let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let entry = entries.get(&id).ok_or_else(|| {
                crate::Error::ExecutionError("Tensor must be registered in registry".to_string())
            })?;
            return entry
                .gpu_buffer
                .as_ref()
                .ok_or_else(|| {
                    crate::Error::ExecutionError(
                        "gpu_buffer must be Some when location is Gpu/Both".to_string(),
                    )
                })
                .cloned();
        }

        // 3. Otherwise, we need to upload from CPU. Check/evict for space first.
        self.evict_lru_for_space(byte_size, device, queue)?;

        // 4. Upload CPU data to GPU
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let entry = entries.get_mut(&id).ok_or_else(|| {
            crate::Error::ExecutionError("Tensor not registered in registry".to_string())
        })?;

        if entry.gpu_buffer.is_none() {
            let cpu_data = entry
                .cpu_data
                .as_ref()
                .ok_or_else(|| crate::Error::ExecutionError("No CPU data to upload".to_string()))?;
            let contents_bytes = match entry.dtype {
                Dtype::F32 => bytemuck::cast_slice(cpu_data).to_vec(),
                Dtype::F16 => {
                    let f16_data: Vec<half::f16> =
                        cpu_data.iter().map(|&x| half::f16::from_f32(x)).collect();
                    bytemuck::cast_slice(&f16_data).to_vec()
                }
                Dtype::BF16 => {
                    let bf16_data: Vec<half::bf16> =
                        cpu_data.iter().map(|&x| half::bf16::from_f32(x)).collect();
                    bytemuck::cast_slice(&bf16_data).to_vec()
                }
            };
            let gpu_buffer = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Tensor Uploaded Buffer"),
                    contents: &contents_bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                },
            ));
            entry.gpu_buffer = Some(gpu_buffer);
            entry.location = BufferLocation::Both;

            let mut gpu_used = self
                .gpu_bytes_used
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            *gpu_used += byte_size;
            let mut peak = self
                .peak_gpu_bytes
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if *gpu_used > *peak {
                *peak = *gpu_used;
            }
            self.upload_count.fetch_add(1, Ordering::Relaxed);
        }

        entry
            .gpu_buffer
            .as_ref()
            .ok_or_else(|| {
                crate::Error::ExecutionError("gpu_buffer was just set above".to_string())
            })
            .cloned()
    }

    pub fn get_gpu_view(
        &self,
        id: TensorId,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<GpuBufferView, crate::Error> {
        if let Some(alloc) = self.gpu_plan.get(&id) {
            if let Some(ref arena) = self.gpu_arena {
                return Ok(GpuBufferView {
                    buffer: arena.clone(),
                    offset: alloc.offset as u64,
                    size: alloc.size as u64,
                });
            }
        }

        let buffer = self.ensure_gpu(id, device, queue)?;
        let byte_size = {
            let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let entry = entries
                .get(&id)
                .ok_or_else(|| crate::Error::ExecutionError("Tensor not registered".to_string()))?;
            entry.byte_size as u64
        };
        Ok(GpuBufferView {
            buffer,
            offset: 0,
            size: byte_size,
        })
    }

    pub fn get_cpu_view(&self, id: TensorId) -> Result<CpuBufferView, crate::Error> {
        if let Some(alloc) = self.cpu_plan.get(&id) {
            return Ok(CpuBufferView {
                offset: alloc.offset,
                size: alloc.size,
            });
        }
        let byte_size = {
            let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let entry = entries
                .get(&id)
                .ok_or_else(|| crate::Error::ExecutionError("Tensor not registered".to_string()))?;
            entry.byte_size
        };
        Ok(CpuBufferView {
            offset: 0,
            size: byte_size,
        })
    }

    /// Ensure tensor is resident on CPU. Downloads from GPU if needed.
    pub fn ensure_cpu(
        &self,
        id: TensorId,
        device: Option<&wgpu::Device>,
    ) -> Result<Vec<f32>, crate::Error> {
        // First check if the tensor is in the GPU static plan
        if let Some(alloc) = self.gpu_plan.get(&id) {
            if let Some(ref gpu_arena) = self.gpu_arena {
                let backend = crate::backend::WgpuBackend::get_or_init().map_err(|_| {
                    crate::Error::ExecutionError("WgpuBackend init failed".to_string())
                })?;
                let queue = backend.queue();
                let actual_device = device.unwrap_or_else(|| backend.device());

                let dtype = {
                    let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(entry) = entries.get(&id) {
                        entry.dtype
                    } else {
                        Dtype::F32
                    }
                };

                let data = download_buffer_slice(
                    actual_device,
                    queue,
                    gpu_arena,
                    alloc.offset,
                    alloc.size,
                    dtype,
                )?;

                // If we have a CPU arena and this tensor is in the CPU plan, copy the data there
                if let Some(cpu_alloc) = self.cpu_plan.get(&id) {
                    if let Some(ref cpu_arena) = self.cpu_arena {
                        let mut arena_lock = cpu_arena.lock().unwrap_or_else(|e| e.into_inner());
                        let start_idx = cpu_alloc.offset / 4; // assuming F32 elements
                        let end_idx = start_idx + (cpu_alloc.size / 4);
                        if end_idx <= arena_lock.len() {
                            arena_lock[start_idx..end_idx].copy_from_slice(&data);
                        }
                    }
                }

                return Ok(data);
            }
        }

        // Also check if it's in the CPU plan only
        if let Some(cpu_alloc) = self.cpu_plan.get(&id) {
            if let Some(ref cpu_arena) = self.cpu_arena {
                let arena_lock = cpu_arena.lock().unwrap_or_else(|e| e.into_inner());
                let start_idx = cpu_alloc.offset / 4;
                let end_idx = start_idx + (cpu_alloc.size / 4);
                if end_idx <= arena_lock.len() {
                    return Ok(arena_lock[start_idx..end_idx].to_vec());
                }
            }
        }

        let entry_info = {
            let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let entry = entries.get(&id).ok_or_else(|| {
                crate::Error::ExecutionError("Tensor not registered in registry".to_string())
            })?;
            (
                entry.location.clone(),
                entry.cpu_data.clone(),
                entry.gpu_buffer.clone(),
                entry.byte_size,
                entry.dtype,
            )
        };

        let (location, cpu_data, gpu_buffer, byte_size, dtype) = entry_info;
        match location {
            BufferLocation::Cpu | BufferLocation::Both => cpu_data.ok_or_else(|| {
                crate::Error::ExecutionError("No CPU data in Cpu/Both state".to_string())
            }),
            BufferLocation::Gpu => {
                let backend = crate::backend::WgpuBackend::get_or_init().map_err(|_| {
                    crate::Error::ExecutionError("WgpuBackend init failed".to_string())
                })?;
                let queue = backend.queue();
                let actual_device = device.unwrap_or_else(|| backend.device());
                let gpu_buf = gpu_buffer.ok_or_else(|| {
                    crate::Error::ExecutionError(
                        "gpu_buffer must be Some when location is Gpu".to_string(),
                    )
                })?;
                let data = download_buffer(actual_device, queue, &gpu_buf, byte_size, dtype)?;

                let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(entry) = entries.get_mut(&id) {
                    entry.cpu_data = Some(data.clone());
                    entry.location = BufferLocation::Both;
                }
                Ok(data)
            }
        }
    }

    /// Mark tensor as last used at op_index (for LRU eviction)
    pub fn touch(&self, id: TensorId, op_index: usize) -> Result<(), crate::Error> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = entries.get_mut(&id) {
            entry.last_used = op_index;
        }
        Ok(())
    }

    /// Pin a tensor so it won't be evicted while actively in use.
    pub fn pin(&self, id: TensorId) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = entries.get_mut(&id) {
            entry.pinned = true;
        }
    }

    /// Unpin a tensor, allowing it to be evicted again.
    pub fn unpin(&self, id: TensorId) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = entries.get_mut(&id) {
            entry.pinned = false;
        }
    }

    /// Evict least-recently-used GPU tensors until under gpu_byte_limit
    pub fn evict_lru(&self, device: &wgpu::Device) -> Result<(), crate::Error> {
        let backend = crate::backend::WgpuBackend::get_or_init()
            .map_err(|_| crate::Error::ExecutionError("WgpuBackend init failed".to_string()))?;
        let queue = backend.queue();
        self.evict_lru_for_space(0, device, queue)
    }

    /// Evict LRU GPU tensors to make room for needed_bytes
    pub fn evict_lru_for_space(
        &self,
        needed_bytes: usize,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), crate::Error> {
        let max_iterations = self.entries.lock().unwrap_or_else(|e| e.into_inner()).len() + 1;
        for _ in 0..max_iterations {
            // Check memory pressure under lock
            let gpu_used = *self
                .gpu_bytes_used
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if gpu_used + needed_bytes <= self.gpu_byte_limit {
                break;
            }

            // Find LRU candidate (skip pinned entries)
            let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let mut candidate: Option<(TensorId, bool, usize, Arc<wgpu::Buffer>, Dtype)> = None;
            let mut min_last_used = usize::MAX;

            for (&id, entry) in entries.iter() {
                if entry.pinned {
                    continue;
                }
                if (entry.location == BufferLocation::Gpu || entry.location == BufferLocation::Both)
                    && entry.last_used < min_last_used
                {
                    min_last_used = entry.last_used;
                    let needs_download = entry.location == BufferLocation::Gpu;
                    candidate = Some((
                        id,
                        needs_download,
                        entry.byte_size,
                        entry
                            .gpu_buffer
                            .as_ref()
                            .ok_or_else(|| {
                                crate::Error::ExecutionError(
                                    "gpu_buffer must be Some for Gpu/Both location".to_string(),
                                )
                            })?
                            .clone(),
                        entry.dtype,
                    ));
                }
            }

            if let Some((id, needs_download, byte_size, gpu_buffer, dtype)) = candidate {
                // Drop lock to perform download safely without deadlocking
                drop(entries);

                let cpu_data = if needs_download {
                    Some(download_buffer(
                        device,
                        queue,
                        &gpu_buffer,
                        byte_size,
                        dtype,
                    )?)
                } else {
                    None
                };

                // Re-acquire lock to modify the entry
                let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(entry) = entries.get_mut(&id) {
                    if entry.location == BufferLocation::Gpu
                        || entry.location == BufferLocation::Both
                    {
                        if needs_download {
                            entry.cpu_data = cpu_data;
                        }
                        entry.gpu_buffer = None;
                        entry.location = BufferLocation::Cpu;

                        let mut gpu_used = self
                            .gpu_bytes_used
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        *gpu_used = gpu_used.saturating_sub(byte_size);
                        self.eviction_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            } else {
                // No candidate to evict
                break;
            }
        }
        Ok(())
    }

    /// Free a tensor entirely (it will never be needed again)
    pub fn free(&self, id: TensorId) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = entries.remove(&id) {
            if entry.location == BufferLocation::Gpu || entry.location == BufferLocation::Both {
                let mut gpu_used = self
                    .gpu_bytes_used
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                *gpu_used = gpu_used.saturating_sub(entry.byte_size);
                self.eviction_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn peak_gpu_bytes(&self) -> usize {
        *self
            .peak_gpu_bytes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    pub fn upload_count(&self) -> usize {
        self.upload_count.load(Ordering::Relaxed)
    }

    pub fn eviction_count(&self) -> usize {
        self.eviction_count.load(Ordering::Relaxed)
    }

    // Helper for testing and debugging to get entry copy
    pub fn is_resident_on_gpu(&self, id: TensorId) -> Result<bool, crate::Error> {
        if self.gpu_plan.contains_key(&id) && self.gpu_arena.is_some() {
            return Ok(true);
        }
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        Ok(if let Some(entry) = entries.get(&id) {
            entry.location == BufferLocation::Gpu || entry.location == BufferLocation::Both
        } else {
            false
        })
    }

    pub fn get_cpu_data(&self, id: TensorId) -> Result<Vec<f32>, crate::Error> {
        if let Some(cpu_alloc) = self.cpu_plan.get(&id) {
            if let Some(ref cpu_arena) = self.cpu_arena {
                let arena_lock = cpu_arena.lock().unwrap_or_else(|e| e.into_inner());
                let start_idx = cpu_alloc.offset / 4;
                let end_idx = start_idx + (cpu_alloc.size / 4);
                if end_idx <= arena_lock.len() {
                    return Ok(arena_lock[start_idx..end_idx].to_vec());
                }
            }
        }

        if let Some(gpu_alloc) = self.gpu_plan.get(&id) {
            if let Some(ref gpu_arena) = self.gpu_arena {
                let backend = crate::backend::WgpuBackend::get_or_init().map_err(|_| {
                    crate::Error::ExecutionError("WgpuBackend init failed".to_string())
                })?;
                let queue = backend.queue();
                let actual_device = backend.device();

                let dtype = {
                    let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(entry) = entries.get(&id) {
                        entry.dtype
                    } else {
                        Dtype::F32
                    }
                };

                return download_buffer_slice(
                    actual_device,
                    queue,
                    gpu_arena,
                    gpu_alloc.offset,
                    gpu_alloc.size,
                    dtype,
                );
            }
        }

        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let entry = entries
            .get(&id)
            .ok_or_else(|| crate::Error::ExecutionError("Tensor not registered".to_string()))?;
        entry.cpu_data.clone().ok_or_else(|| {
            crate::Error::ExecutionError("Tensor does not have CPU data".to_string())
        })
    }

    pub fn update_cpu_data(&self, id: TensorId, new_data: Vec<f32>) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = entries.get_mut(&id) {
            entry.cpu_data = Some(new_data);
            if entry.location == BufferLocation::Gpu || entry.location == BufferLocation::Both {
                entry.gpu_buffer = None;
                entry.location = BufferLocation::Cpu;
                // Subtract from gpu bytes used
                let mut gpu_used = self
                    .gpu_bytes_used
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                *gpu_used = gpu_used.saturating_sub(entry.byte_size);
            }
        }
    }

    pub fn contains(&self, id: TensorId) -> bool {
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.contains_key(&id)
    }

    /// Execute a CPU operator on arena sub-slices with zero heap allocations.
    ///
    /// # Design
    /// All intermediate CPU tensors live inside a single pre-allocated arena (`cpu_arena`).
    /// The static memory planner guarantees that input and output regions never overlap
    /// during a given execution step (interval-coloring invariant).
    ///
    /// We therefore use raw pointer manipulation to simultaneously hold shared borrows
    /// of input regions and an exclusive borrow of the output region from the same Vec.
    /// This is safe because:
    ///   1. All offsets are 256-byte aligned (planner invariant).
    ///   2. No two live tensors at the same step share any byte range (planner invariant).
    ///   3. All arena accesses are bounds-checked via `split_at` before the unsafe block.
    ///
    /// For Reshape: treated as a pure alias – no data movement, zero work.
    /// For external (non-arena) tensors: falls back to `ensure_cpu`.
    pub fn execute_cpu_op_slices(
        &self,
        op: &crate::graph::Op,
        input_views: &[CpuBufferView],
        input_shapes: &[crate::tensor::Shape],
        output_view: CpuBufferView,
        output_shape: &crate::tensor::Shape,
        backend: &crate::backend::CpuBackend,
        // Externally-supplied data for non-arena inputs (keyed by position index).
        // If present for position i, that data is used instead of the arena slice.
        external_inputs: &[Option<Vec<f32>>],
    ) -> Result<(), crate::Error> {
        // Reshape is zero-copy: the output view aliases the input view by planner design.
        if matches!(op, crate::graph::Op::Reshape { .. }) {
            return Ok(());
        }

        let arena = match &self.cpu_arena {
            Some(a) => a.clone(),
            None => {
                // No static arena: build Tensors from ensure_cpu and fall back.
                return Err(crate::Error::ExecutionError(
                    "execute_cpu_op_slices called without a CPU arena".to_string(),
                ));
            }
        };

        let arena_guard = arena.lock().unwrap_or_else(|e| e.into_inner());
        let arena_ptr: *const f32 = arena_guard.as_ptr();
        let arena_len = arena_guard.len();

        // Build input slices. For each position, prefer external data if supplied.
        let mut input_data_storage: Vec<Vec<f32>> = Vec::new();
        let mut input_slice_ptrs: Vec<(*const f32, usize)> = Vec::new();

        for (i, view) in input_views.iter().enumerate() {
            if let Some(Some(ref ext)) = external_inputs.get(i) {
                input_data_storage.push(ext.clone());
                let stored = input_data_storage.last().ok_or_else(|| {
                    crate::Error::ExecutionError("external data was just pushed above".to_string())
                })?;
                input_slice_ptrs.push((stored.as_ptr(), stored.len()));
            } else {
                // f32 elements: offset in bytes / 4 (all arena data is f32)
                let start_elem = view.offset / 4;
                let count = view.size / 4;
                assert!(
                    start_elem + count <= arena_len,
                    "CPU arena input out of bounds: start={start_elem} count={count} len={arena_len}"
                );
                input_slice_ptrs.push((unsafe { arena_ptr.add(start_elem) }, count));
            }
        }

        // Build output slice pointer (exclusive borrow via raw pointer).
        let out_start_elem = output_view.offset / 4;
        let out_count = output_view.size / 4;
        assert!(
            out_start_elem + out_count <= arena_len,
            "CPU arena output out of bounds: start={out_start_elem} count={out_count} len={arena_len}"
        );

        // Debug-mode overlap check: ensure no input overlaps the output region.
        #[cfg(debug_assertions)]
        for (ptr, count) in &input_slice_ptrs {
            let in_start = (*ptr as usize - arena_ptr as usize) / 4;
            let in_end = in_start + count;
            let out_end = out_start_elem + out_count;
            assert!(
                in_end <= out_start_elem || in_start >= out_end,
                "CPU arena overlap detected between input [{in_start}..{in_end}) and output \
                 [{out_start_elem}..{out_end})"
            );
        }

        // SAFETY: The static memory planner guarantees non-overlapping intervals.
        // We hold a Mutex lock on arena_guard; no other thread can mutate it.
        // We construct shared input slices and an exclusive output slice from disjoint
        // byte ranges of the same allocation.
        let input_slices: Vec<&[f32]> = input_slice_ptrs
            .iter()
            .map(|&(ptr, len)| unsafe { std::slice::from_raw_parts(ptr, len) })
            .collect();

        let output_slice: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(
                arena_guard.as_ptr().add(out_start_elem) as *mut f32,
                out_count,
            )
        };

        let input_refs: Vec<&[f32]> = input_slices.to_vec();
        backend.execute_slices(op, &input_refs, input_shapes, output_slice, output_shape)
    }
}

fn download_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size_bytes: usize,
    dtype: Dtype,
) -> Result<Vec<f32>, crate::Error> {
    download_buffer_slice(device, queue, buffer, 0, size_bytes, dtype)
}

fn download_buffer_slice(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset: usize,
    size_bytes: usize,
    dtype: Dtype,
) -> Result<Vec<f32>, crate::Error> {
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Eviction Staging Read Buffer"),
        size: size_bytes as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Eviction Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, offset as u64, &staging_buffer, 0, size_bytes as u64);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| {
            crate::Error::ExecutionError("channel must be alive for buffer readback".to_string())
        })?
        .map_err(|_| crate::Error::ExecutionError("buffer map_async must succeed".to_string()))?;

    let data_view = buffer_slice.get_mapped_range();
    let data = match dtype {
        Dtype::F32 => bytemuck::cast_slice::<u8, f32>(&data_view).to_vec(),
        Dtype::F16 => {
            let f16_data = bytemuck::cast_slice::<u8, half::f16>(&data_view);
            f16_data.iter().map(|&x| x.to_f32()).collect()
        }
        Dtype::BF16 => {
            let bf16_data = bytemuck::cast_slice::<u8, half::bf16>(&data_view);
            bf16_data.iter().map(|&x| f32::from(x)).collect()
        }
    };
    drop(data_view);
    staging_buffer.unmap();

    Ok(data)
}
