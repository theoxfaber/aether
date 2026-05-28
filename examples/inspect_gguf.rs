/// Just test GGUF loading
use aether::loader::gguf::GGUFLoader;

fn main() {
    eprintln!("Loading...");
    let model = match GGUFLoader::load("tinyllama-q4.gguf") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: {}", e);
            return;
        }
    };
    eprintln!("Loaded {} tensors", model.tensors.len());
    eprintln!("Metadata:");
    for (k, v) in &model.metadata {
        if k == "tokenizer.ggml.tokens" {
            eprintln!(
                "  {}: Array(len={})",
                k,
                match v {
                    aether::loader::gguf::GGUFValue::Array(a) => a.len(),
                    _ => 0,
                }
            );
        } else {
            eprintln!("  {}: {:?}", k, v);
        }
    }
}
