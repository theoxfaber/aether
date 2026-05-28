use aether::loader::gguf::GGUFLoader;
fn main() {
    let gguf = GGUFLoader::load("tinyllama-q4.gguf").unwrap();
    let tensors = &gguf.tensors;
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ] {
        let t = &tensors[name];
        eprintln!("{:<30} shape={:?} dtype={:?}", name, t.shape, t.dtype);
    }
}
