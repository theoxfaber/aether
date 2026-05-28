use aether::loader::gguf::GGUFLoader;
use aether::tokenizer::Tokenizer;

fn main() {
    let gguf = GGUFLoader::load("tinyllama-q4.gguf").unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();

    // Check what token 18107 is
    eprintln!("token 18107: {:?}", tokenizer.decode_one(18107));

    // Check first few tokens
    for t in [0u32, 1, 2, 3, 29871, 13, 32, 18107] {
        eprintln!("token {}: {:?}", t, tokenizer.decode_one(t));
    }

    // Encode various strings
    for s in &[
        "Hello",
        "Hello world",
        "The capital of France is",
        "уче",
        "Paris",
    ] {
        let encoded = tokenizer.encode(s, false);
        let decoded = tokenizer.decode(&encoded);
        eprintln!("'{}' -> tokens {:?} -> '{}'", s, encoded, decoded);
    }
}
