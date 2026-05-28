use aether::loader::dequant::dequantize;
use aether::loader::gguf::GGUFLoader;

fn main() {
    let gguf = GGUFLoader::load("tinyllama-q4.gguf").unwrap();

    let tensors = &gguf.tensors;

    // Check token embeddings shape
    if let Some(t) = tensors.get("token_embd.weight") {
        eprintln!(
            "token_embd.weight: shape={:?} dtype={:?} len={}",
            t.shape,
            t.dtype,
            t.data.len()
        );
        let f32_data = dequantize(&t.data, t.dtype, &t.shape);
        eprintln!("f32 len: {} first 5: {:?}", f32_data.len(), &f32_data[..5]);

        // Check specific token embeddings
        // Token 0 = BOS, check first few tokens
        let d_model = 2048usize;
        for tok in [0u32, 1, 2, 100, 1000, 18107u32] {
            let start = tok as usize * d_model;
            if start + d_model <= f32_data.len() {
                let emb = &f32_data[start..start + d_model];
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!("token {}: norm={:.6} first_5={:?}", tok, norm, &emb[..5]);
            }
        }
    }

    // Check output_norm and output
    if let Some(t) = tensors.get("output_norm.weight") {
        eprintln!(
            "output_norm.weight: shape={:?} dtype={:?} len={}",
            t.shape,
            t.dtype,
            t.data.len()
        );
    }
    if let Some(t) = tensors.get("output.weight") {
        eprintln!(
            "output.weight: shape={:?} dtype={:?} len={}",
            t.shape,
            t.dtype,
            t.data.len()
        );
    }
}
