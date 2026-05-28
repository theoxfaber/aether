#![allow(unused)]

use aether::loader::dequant::dequantize;
use aether::loader::gguf::GGUFDtype;
use aether::loader::gguf::GGUFLoader;
use std::time::Instant;

fn main() {
    let gguf = GGUFLoader::load("mistral-7b-q4k.gguf").unwrap();
    for (name, tensor) in &gguf.tensors {
        if name == "blk.0.ffn_gate.weight" {
            let elems = tensor.shape[0] * tensor.shape[1];
            eprintln!(
                "{}: shape={:?}, elements={}, dtype={:?}",
                name, tensor.shape, elems, tensor.dtype
            );
            let t0 = Instant::now();
            let flat = dequantize(&tensor.data, tensor.dtype, &tensor.shape);
            eprintln!(
                "  dequantize only: {:.3}s (no transpose)",
                t0.elapsed().as_secs_f64()
            );
            let t1 = Instant::now();
            let flat2 = dequantize(&tensor.data, tensor.dtype, &tensor.shape);
            eprintln!("  dequantize repeated: {:.3}s", t1.elapsed().as_secs_f64());
            break;
        }
    }
    // Also time symmetric weight
    for (name, tensor) in &gguf.tensors {
        if name == "blk.0.attn_q.weight" {
            let elems = tensor.shape[0] * tensor.shape[1];
            eprintln!(
                "{}: shape={:?}, elements={}, dtype={:?}",
                name, tensor.shape, elems, tensor.dtype
            );
            let t0 = Instant::now();
            let flat = dequantize(&tensor.data, tensor.dtype, &tensor.shape);
            eprintln!("  dequantize only: {:.3}s", t0.elapsed().as_secs_f64());
            break;
        }
    }
}
