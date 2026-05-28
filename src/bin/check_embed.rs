use aether::loader::gguf::GGUFLoader;
fn main() {
    let gguf = GGUFLoader::load("tinyllama-q4.gguf").unwrap();
    let t = &gguf.tensors["token_embd.weight"];
    eprintln!("token_embd.weight shape: {:?}", t.shape);
    let t2 = &gguf.tensors["output.weight"];
    eprintln!("output.weight shape: {:?}", t2.shape);
    for name in ["blk.0.attn_norm.weight", "blk.0.ffn_norm.weight"] {
        let w = &gguf.tensors[name];
        eprintln!("{} shape: {:?}", name, w.shape);
    }
    // Check metadata for architecture
    for (k, v) in &gguf.metadata {
        eprintln!("meta: {} = {:?}", k, v);
    }
}
