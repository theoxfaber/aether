use std::collections::HashSet;
use std::io::Write;

use aether::inference::runner::{self, LlamaRunner};

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

fn build_synthetic_model() -> Vec<u8> {
    let d_model = 8usize;
    let d_ff = 32usize;
    let num_layers = 1usize;
    let vocab_size = 8usize;
    let num_heads = 2usize;
    let num_kv_heads = 2usize;

    let mut buf = Vec::new();

    // ── Header ────────────────────────────────────────────────────────────
    write_u32(&mut buf, 0x46554747); // magic
    write_u32(&mut buf, 3);          // version
    write_u64(&mut buf, 0);          // tensor_count (patched later)
    write_u64(&mut buf, 11);         // metadata_kv_count

    // ── Metadata ──────────────────────────────────────────────────────────
    write_string(&mut buf, "general.architecture");
    write_u32(&mut buf, 8);
    write_string(&mut buf, "llama");

    write_string(&mut buf, "llama.embedding_length");
    write_u32(&mut buf, 4);
    write_u32(&mut buf, d_model as u32);

    write_string(&mut buf, "llama.block_count");
    write_u32(&mut buf, 4);
    write_u32(&mut buf, num_layers as u32);

    write_string(&mut buf, "llama.feed_forward_length");
    write_u32(&mut buf, 4);
    write_u32(&mut buf, d_ff as u32);

    write_string(&mut buf, "llama.attention.head_count");
    write_u32(&mut buf, 4);
    write_u32(&mut buf, num_heads as u32);

    write_string(&mut buf, "llama.attention.head_count_kv");
    write_u32(&mut buf, 4);
    write_u32(&mut buf, num_kv_heads as u32);

    write_string(&mut buf, "llama.attention.layer_norm_rms_epsilon");
    write_u32(&mut buf, 6);
    write_f32_le(&mut buf, 1e-6);

    write_string(&mut buf, "llama.context_length");
    write_u32(&mut buf, 4);
    write_u32(&mut buf, 128);

    write_string(&mut buf, "llama.rope.freq_base");
    write_u32(&mut buf, 6);
    write_f32_le(&mut buf, 10000.0);

    write_string(&mut buf, "tokenizer.ggml.bos_token_id");
    write_u32(&mut buf, 4);
    write_u32(&mut buf, 1);

    write_string(&mut buf, "tokenizer.ggml.tokens");
    write_u32(&mut buf, 9);  // Array
    write_u32(&mut buf, 8);  // element type = String
    write_u64(&mut buf, 8);  // count
    for t in &["<unk>", "<s>", "</s>", "a", "b", "c", "d", "e"] {
        write_string(&mut buf, t);
    }

    // ── Tensor info ──────────────────────────────────────────────────────
    // Build list of (name, shape_in_gguf, byte_offset)
    struct TensorEntry {
        name: String,
        shape: Vec<u64>,
        offset: u64,
    }

    let byte_size = |shape: &[u64]| -> u64 { shape.iter().product::<u64>() * 4 };

    let mut tensors = Vec::new();
    let mut cur = 0u64;

    // Global tensors (GGUF shape convention: [in_features, out_features])
    tensors.push(TensorEntry {
        name: "token_embd.weight".into(),
        shape: vec![d_model as u64, vocab_size as u64],
        offset: cur,
    });
    cur += byte_size(&[d_model as u64, vocab_size as u64]);

    tensors.push(TensorEntry {
        name: "output_norm.weight".into(),
        shape: vec![d_model as u64],
        offset: cur,
    });
    cur += byte_size(&[d_model as u64]);

    tensors.push(TensorEntry {
        name: "output.weight".into(),
        shape: vec![d_model as u64, vocab_size as u64],
        offset: cur,
    });
    cur += byte_size(&[d_model as u64, vocab_size as u64]);

    // Per-layer tensors
    for li in 0..num_layers {
        let p = format!("blk.{}.", li);

        // attn_norm.weight [d_model]
        tensors.push(TensorEntry { name: format!("{}attn_norm.weight", p), shape: vec![d_model as u64], offset: cur });
        cur += byte_size(&[d_model as u64]);

        // attn_q.weight [d_model, d_model]
        tensors.push(TensorEntry { name: format!("{}attn_q.weight", p), shape: vec![d_model as u64, d_model as u64], offset: cur });
        cur += byte_size(&[d_model as u64, d_model as u64]);

        // attn_k.weight [d_model, d_model]
        tensors.push(TensorEntry { name: format!("{}attn_k.weight", p), shape: vec![d_model as u64, d_model as u64], offset: cur });
        cur += byte_size(&[d_model as u64, d_model as u64]);

        // attn_v.weight [d_model, d_model]
        tensors.push(TensorEntry { name: format!("{}attn_v.weight", p), shape: vec![d_model as u64, d_model as u64], offset: cur });
        cur += byte_size(&[d_model as u64, d_model as u64]);

        // attn_output.weight [d_model, d_model]
        tensors.push(TensorEntry { name: format!("{}attn_output.weight", p), shape: vec![d_model as u64, d_model as u64], offset: cur });
        cur += byte_size(&[d_model as u64, d_model as u64]);

        // ffn_norm.weight [d_model]
        tensors.push(TensorEntry { name: format!("{}ffn_norm.weight", p), shape: vec![d_model as u64], offset: cur });
        cur += byte_size(&[d_model as u64]);

        // ffn_gate.weight [d_model, d_ff]
        tensors.push(TensorEntry { name: format!("{}ffn_gate.weight", p), shape: vec![d_model as u64, d_ff as u64], offset: cur });
        cur += byte_size(&[d_model as u64, d_ff as u64]);

        // ffn_up.weight [d_model, d_ff]
        tensors.push(TensorEntry { name: format!("{}ffn_up.weight", p), shape: vec![d_model as u64, d_ff as u64], offset: cur });
        cur += byte_size(&[d_model as u64, d_ff as u64]);

        // ffn_down.weight [d_ff, d_model]
        tensors.push(TensorEntry { name: format!("{}ffn_down.weight", p), shape: vec![d_ff as u64, d_model as u64], offset: cur });
        cur += byte_size(&[d_ff as u64, d_model as u64]);
    }

    // Write tensor info entries
    for t in &tensors {
        write_string(&mut buf, &t.name);
        write_u32(&mut buf, t.shape.len() as u32);
        for &d in &t.shape {
            write_u64(&mut buf, d);
        }
        write_u32(&mut buf, 0); // dtype = F32
        write_u64(&mut buf, t.offset);
    }

    // Patch tensor_count at offset 8
    let tensor_count = tensors.len() as u64;
    buf[8..16].copy_from_slice(&tensor_count.to_le_bytes());

    // Align to 32 bytes
    pad_to(&mut buf, 32);

    // ── Tensor data ───────────────────────────────────────────────────────
    // Deterministic small non-zero values
    for t in &tensors {
        let n: usize = t.shape.iter().map(|&d| d as usize).product();
        for e in 0..n {
            let val = ((e % 17) as f32 - 8.0) * 0.01;
            write_f32_le(&mut buf, val);
        }
    }

    buf
}

#[test]
fn test_synthetic_model_load_and_prefill() {
    let gguf_bytes = build_synthetic_model();
    let tmp = std::env::temp_dir().join("aether_synthetic_e2e.gguf");
    std::fs::write(&tmp, &gguf_bytes).unwrap();

    let mut runner = LlamaRunner::from_gguf(tmp.to_str().unwrap()).unwrap();

    // Encode a short prompt
    let prompt = "abc";
    let tokens = runner.tokenizer.encode(prompt, true);
    assert!(!tokens.is_empty(), "Should have at least one token");

    // Prefill runs the full forward pass on all tokens
    let logits = runner.prefill(&tokens).unwrap();
    assert_eq!(logits.len(), 8, "Logits should be vocab_size = 8");
    for &v in &logits {
        assert!(v.is_finite(), "All logits should be finite");
    }

    // Decode one token
    let pos = tokens.len();
    let prev_set: HashSet<u32> = tokens.iter().copied().collect();
    let next_token = runner::sample(&logits, 0.0, 0.9, &prev_set, 1.0);

    let num_layers = runner.layer_assignment.layer_devices.len();
    let mut layer_tel = vec![aether::inference::telemetry::LayerTelemetry::default(); num_layers];
    let next_logits = runner.decode_step(next_token, pos, &mut layer_tel).unwrap();
    assert_eq!(next_logits.len(), 8);
    for &v in &next_logits {
        assert!(v.is_finite(), "Decode logits should be finite");
    }

    std::fs::remove_file(&tmp).ok();
}
