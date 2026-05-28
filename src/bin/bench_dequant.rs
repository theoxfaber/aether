use aether::loader::dequant::dequantize_transpose_f32;
use aether::loader::gguf::GGUFLoader;
use std::time::Instant;

fn main() {
    let path = "mistral-7b-q4k.gguf";
    eprintln!("Loading GGUF...");
    let t0 = Instant::now();
    let gguf = GGUFLoader::load(path).unwrap();
    eprintln!("GGUF loaded in {:.2}s", t0.elapsed().as_secs_f32());

    // Find gate_proj weight (asymmetric: [4096, 14336]) in layer 0
    for (name, tensor) in &gguf.tensors {
        if name == "blk.0.ffn_gate.weight" {
            eprintln!(
                "Found {}: shape={:?}, dtype={:?}, data_len={}",
                name,
                tensor.shape,
                tensor.dtype,
                tensor.data.len()
            );
            let t1 = Instant::now();
            let result = dequantize_transpose_f32(
                &tensor.data,
                tensor.dtype,
                tensor.shape[0],
                tensor.shape[1],
            );
            eprintln!(
                "dequantize_transpose_f32: {:.2}s for {} elements -> {} f32 values",
                t1.elapsed().as_secs_f32(),
                tensor.shape[0] * tensor.shape[1],
                result.len()
            );
            break;
        }
    }
}
