use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

pub struct PipelineCache {
    cache: RwLock<HashMap<String, CacheEntry>>,
}

#[derive(Clone)]
enum CacheEntry {
    Ready(Arc<wgpu::ComputePipeline>),
    Pending,
}

static CACHE: OnceLock<PipelineCache> = OnceLock::new();

impl PipelineCache {
    pub fn global() -> &'static Self {
        CACHE.get_or_init(|| Self {
            cache: RwLock::new(HashMap::new()),
        })
    }

    pub fn get_or_compile(
        &self,
        key: &str,
        wgsl: &str,
        device: &wgpu::Device,
    ) -> Arc<wgpu::ComputePipeline> {
        loop {
            // Scope for read lock
            {
                let read_guard = self.cache.read().unwrap();
                if let Some(entry) = read_guard.get(key) {
                    match entry {
                        CacheEntry::Ready(pipeline) => return pipeline.clone(),
                        CacheEntry::Pending => {
                            // Drop lock and spin-wait
                            drop(read_guard);
                            std::thread::yield_now();
                            continue;
                        }
                    }
                }
            }

            // Cache miss: let's try to insert `Pending` using write lock
            let mut write_guard = self.cache.write().unwrap();
            // Double-check under write lock to avoid race
            if let Some(entry) = write_guard.get(key) {
                match entry {
                    CacheEntry::Ready(pipeline) => return pipeline.clone(),
                    CacheEntry::Pending => {
                        drop(write_guard);
                        std::thread::yield_now();
                        continue;
                    }
                }
            }

            write_guard.insert(key.to_string(), CacheEntry::Pending);
            drop(write_guard); // Release write lock during compilation

            // Compile the pipeline
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("AST Shader: {}", key)),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("AST Pipeline: {}", key)),
                layout: None,
                module: &shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

            let pipeline_arc = Arc::new(pipeline);

            // Update to Ready
            let mut write_guard = self.cache.write().unwrap();
            write_guard.insert(key.to_string(), CacheEntry::Ready(pipeline_arc.clone()));
            return pipeline_arc;
        }
    }
}
