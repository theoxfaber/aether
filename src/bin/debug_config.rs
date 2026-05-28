use aether::inference::model_loader::LlamaConfig;
use aether::loader::gguf::GGUFLoader;

fn main() {
    let gguf = GGUFLoader::load("tinyllama-q4.gguf").unwrap();
    let cfg = LlamaConfig::from_gguf(&gguf).unwrap();

    eprintln!("Config: {:?}", cfg);
    eprintln!("d_model={} n_layers={} n_heads={} n_kv={} head_dim={} d_ff={} vocab={} max_seq={} rope_base={} eps={} rope_dim={}",
        cfg.d_model, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
        cfg.d_ff, cfg.vocab_size, cfg.max_seq_len, cfg.rope_base, cfg.rms_norm_eps, cfg.rope_dim);
    eprintln!("arch: {:?}", cfg.arch);
}
