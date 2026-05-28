use std::io::Write;

use aether::inference::runner::LlamaRunner;

// ── GGUF binary writer helpers (GGUF v3) ──

fn write_u32(w: &mut impl Write, v: u32) {
    w.write_all(&v.to_le_bytes()).unwrap();
}
fn write_u64(w: &mut impl Write, v: u64) {
    w.write_all(&v.to_le_bytes()).unwrap();
}
fn write_f32_le(w: &mut impl Write, v: f32) {
    w.write_all(&v.to_le_bytes()).unwrap();
}
fn write_string(w: &mut impl Write, s: &str) {
    write_u64(w, s.len() as u64);
    w.write_all(s.as_bytes()).unwrap();
}
fn pad_to(w: &mut Vec<u8>, align: usize) {
    while w.len() % align != 0 {
        w.push(0);
    }
}

// ── Synthetic model dimensions (Llama 3.2 1B scaled down) ──

const D_MODEL: usize = 32;
const N_LAYERS: usize = 2;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const D_FF: usize = 64;
const VOCAB_SIZE: usize = 256;
const MAX_SEQ: usize = 64;
const ROPE_DIM: usize = 16;

const fn kv_dim() -> usize {
    D_MODEL * N_KV_HEADS / N_HEADS
}

/// Generate a minimal but valid GGUF file containing a Llama 3.2 1B
/// configuration with all required tensors filled with deterministic f32 data.
fn generate_test_gguf() -> Vec<u8> {
    // ── Define tensor descriptors ──
    struct TDesc {
        name: String,
        shape: Vec<usize>,
        n_dims: u32,
        offset: u64,
    }

    let mut tensors: Vec<TDesc> = Vec::new();

    macro_rules! t2 {
        ($name:expr, $d0:expr, $d1:expr) => {
            tensors.push(TDesc {
                name: $name.to_string(),
                shape: vec![$d0, $d1],
                n_dims: 2,
                offset: 0,
            });
        };
    }
    macro_rules! t1 {
        ($name:expr, $d0:expr) => {
            tensors.push(TDesc {
                name: $name.to_string(),
                shape: vec![$d0],
                n_dims: 1,
                offset: 0,
            });
        };
    }

    // Embedding + LM head
    t2!("token_embd.weight", D_MODEL, VOCAB_SIZE);
    t2!("output.weight", D_MODEL, VOCAB_SIZE);

    // Transformer layers
    for l in 0..N_LAYERS {
        t2!(&format!("blk.{}.attn_q.weight", l), D_MODEL, D_MODEL);
        t2!(&format!("blk.{}.attn_k.weight", l), D_MODEL, kv_dim());
        t2!(&format!("blk.{}.attn_v.weight", l), D_MODEL, kv_dim());
        t2!(&format!("blk.{}.attn_output.weight", l), D_MODEL, D_MODEL);
        t2!(&format!("blk.{}.ffn_gate.weight", l), D_MODEL, D_FF);
        t2!(&format!("blk.{}.ffn_up.weight", l), D_MODEL, D_FF);
        t2!(&format!("blk.{}.ffn_down.weight", l), D_FF, D_MODEL);
        t1!(&format!("blk.{}.attn_norm.weight", l), D_MODEL);
        t1!(&format!("blk.{}.ffn_norm.weight", l), D_MODEL);
    }

    // Final norm
    t1!("output_norm.weight", D_MODEL);

    // Compute data offsets (relative to start of tensor data section)
    let mut data_offset = 0u64;
    for t in &mut tensors {
        let n: usize = t.shape.iter().product();
        t.offset = data_offset;
        data_offset += (n * 4) as u64; // F32 = 4 bytes per element
    }

    // ── Build metadata entries ──
    struct MEntry {
        key: String,
        value_type: u32, // GGUFValueType
        value_bytes: Vec<u8>,
    }

    let mut meta: Vec<MEntry> = Vec::new();

    // Helper: push into `meta`
    macro_rules! meta_str {
        ($key:expr, $val:expr) => {{
            let mut vb = Vec::new();
            write_string(&mut vb, $val);
            meta.push(MEntry {
                key: $key.to_string(),
                value_type: 8,
                value_bytes: vb,
            });
        }};
    }
    macro_rules! meta_u32 {
        ($key:expr, $val:expr) => {{
            meta.push(MEntry {
                key: $key.to_string(),
                value_type: 4,
                value_bytes: ($val as u32).to_le_bytes().to_vec(),
            });
        }};
    }
    macro_rules! meta_f32 {
        ($key:expr, $val:expr) => {{
            meta.push(MEntry {
                key: $key.to_string(),
                value_type: 6,
                value_bytes: ($val as f32).to_le_bytes().to_vec(),
            });
        }};
    }
    macro_rules! meta_str_array {
        ($key:expr, $vals:expr) => {{
            let mut vb = Vec::new();
            vb.extend_from_slice(&8u32.to_le_bytes()); // element type = String
            vb.extend_from_slice(&($vals.len() as u64).to_le_bytes());
            for v in $vals {
                write_string(&mut vb, v);
            }
            meta.push(MEntry {
                key: $key.to_string(),
                value_type: 9,
                value_bytes: vb,
            });
        }};
    }

    meta_str!("general.architecture", "llama");
    meta_u32!("general.alignment", 32);
    meta_u32!("embedding_length", D_MODEL);
    meta_u32!("block_count", N_LAYERS);
    meta_u32!("attention.head_count", N_HEADS);
    meta_u32!("attention.head_count_kv", N_KV_HEADS);
    meta_u32!("feed_forward_length", D_FF);
    meta_u32!("context_length", MAX_SEQ);
    meta_f32!("rope.freq_base", 10000.0);
    meta_f32!("attention.layer_norm_rms_epsilon", 1e-5);
    meta_u32!("rope.dimension_count", ROPE_DIM);

    // Tokenizer: byte-fallback tokens for full byte coverage
    let tokens: Vec<String> = (0..VOCAB_SIZE)
        .map(|i| format!("<0x{:02X}>", i))
        .collect();
    meta_str_array!("tokenizer.ggml.tokens", &tokens);
    meta_u32!("tokenizer.ggml.bos_token_id", 1);
    meta_u32!("tokenizer.ggml.eos_token_id", 2);

    // ── Serialize ──
    let mut buf = Vec::new();

    // Header
    write_u32(&mut buf, 0x46554747); // GGUF magic
    write_u32(&mut buf, 3); // version 3
    write_u64(&mut buf, tensors.len() as u64);
    write_u64(&mut buf, meta.len() as u64);

    // Metadata KVs
    for m in &meta {
        write_string(&mut buf, &m.key);
        write_u32(&mut buf, m.value_type);
        buf.extend_from_slice(&m.value_bytes);
    }

    // Tensor info entries
    for t in &tensors {
        write_string(&mut buf, &t.name);
        write_u32(&mut buf, t.n_dims);
        for &dim in &t.shape {
            write_u64(&mut buf, dim as u64);
        }
        write_u32(&mut buf, 0); // GGUFDtype::F32 = 0
        write_u64(&mut buf, t.offset);
    }

    // Pad to alignment (default GGUF alignment = 32)
    pad_to(&mut buf, 32);

    // Tensor data: deterministic f32 values based on cumulative position
    let mut counter = 0u32;
    for t in &tensors {
        let n: usize = t.shape.iter().product();
        for _ in 0..n {
            let val = (counter % 100) as f32 * 0.01 - 0.5;
            write_f32_le(&mut buf, val);
            counter += 1;
        }
    }

    buf
}

#[test]
fn test_e2e_synthetic_model() {
    let gguf_bytes = generate_test_gguf();
    let model_path = std::env::temp_dir().join("aether_e2e_synthetic.gguf");
    std::fs::write(&model_path, &gguf_bytes).unwrap();

    // Load with LlamaRunner
    let mut runner =
        LlamaRunner::from_gguf(model_path.to_str().unwrap()).expect("Failed to load synthetic GGUF");

    // Generate with greedy decoding (temperature=0) for reproducibility
    let output1 = runner
        .generate("test", 16, 0.0, 0.9, 1.0)
        .expect("First generation failed");

    // Drop and reload to reset KV cache
    drop(runner);
    let mut runner2 =
        LlamaRunner::from_gguf(model_path.to_str().unwrap()).expect("Failed to load synthetic GGUF (second load)");

    let output2 = runner2
        .generate("test", 16, 0.0, 0.9, 1.0)
        .expect("Second generation failed");

    assert_eq!(
        output1, output2,
        "Deterministic generation must produce identical output for the same input"
    );

    std::fs::remove_file(&model_path).ok();
}
