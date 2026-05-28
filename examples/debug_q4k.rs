/// Find blocks with NaN/Inf d/dmin
use aether::loader::gguf::GGUFLoader;
use half::f16;

fn main() {
    let model = GGUFLoader::load("tinyllama-q4.gguf").unwrap();
    if let Some(tensor) = model.tensors.get("blk.0.attn_q.weight") {
        let data = &tensor.data;
        let n_blocks = data.len() / 72;
        let mut nan_blocks = Vec::new();
        for bi in 0..n_blocks {
            let bo = bi * 72;
            let d = f16::from_le_bytes([data[bo], data[bo + 1]]).to_f32();
            let dmin = f16::from_le_bytes([data[bo + 2], data[bo + 3]]).to_f32();
            if d.is_nan() || d.is_infinite() || dmin.is_nan() || dmin.is_infinite() {
                nan_blocks.push((bi, d, dmin));
            }
            // Also check for extreme values
            if d.abs() > 1000.0 || dmin.abs() > 1000.0 {
                if nan_blocks.is_empty() || nan_blocks.last().map_or(true, |&(i, _, _)| i != bi) {
                    // Don't add duplicate from both checks
                }
            }
        }
        println!("Total blocks: {}", n_blocks);
        println!("Blocks with non-finite d/dmin: {}", nan_blocks.len());
        for (i, d, dmin) in nan_blocks.iter().take(10) {
            println!("  Block {}: d={:?} dmin={:?}", i, d, dmin);
        }

        // Also count blocks where d or dmin is very large
        let mut large_count = 0;
        for bi in 0..n_blocks {
            let bo = bi * 72;
            let d = f16::from_le_bytes([data[bo], data[bo + 1]]).to_f32();
            let dmin = f16::from_le_bytes([data[bo + 2], data[bo + 3]]).to_f32();
            if d.abs() > 100.0 || dmin.abs() > 100.0 {
                large_count += 1;
            }
        }
        println!("Blocks with |d| > 100 or |dmin| > 100: {}", large_count);
    }
}
