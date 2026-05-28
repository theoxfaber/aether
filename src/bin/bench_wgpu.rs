use aether::backend::wgpu_backend::WgpuBackend;
use std::time::Instant;

fn main() {
    eprintln!("Testing WgpuBackend init...");
    let t0 = Instant::now();
    match WgpuBackend::get_or_init() {
        Ok(_) => eprintln!(
            "WgpuBackend initialized in {:.2}s",
            t0.elapsed().as_secs_f32()
        ),
        Err(e) => eprintln!("WgpuBackend init FAILED: {:?}", e),
    }
}
