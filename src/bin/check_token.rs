use aether::loader::gguf::GGUFLoader;
use aether::tokenizer::Tokenizer;

fn main() {
    let gguf = GGUFLoader::load("tinyllama-q4.gguf").unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
    for t in [24155u32, 31192, 16489, 16040, 4014, 20067] {
        eprintln!("token {}: {:?}", t, tokenizer.decode_one(t));
    }
}
