use std::collections::HashMap;
use std::io::Write;

use aether::loader::dequant::dequantize;
use aether::loader::gguf::{GGUFDtype, GGUFLoader, GGUFModel, GGUFValue};

// ── Helpers to build synthetic GGUF v3 binaries ──

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
    while !w.len().is_multiple_of(align) {
        w.push(0);
    }
}

fn build_simple_gguf() -> Vec<u8> {
    let mut buf = Vec::new();

    let magic: u32 = 0x46554747;
    write_u32(&mut buf, magic);
    write_u32(&mut buf, 3);
    write_u64(&mut buf, 2);
    write_u64(&mut buf, 2);

    write_string(&mut buf, "general.name");
    write_u32(&mut buf, 8);
    write_string(&mut buf, "test-model");

    write_string(&mut buf, "test.int");
    write_u32(&mut buf, 5);
    write_u32(&mut buf, 42i32 as u32);

    let weight_offset: u64 = 0;
    write_string(&mut buf, "weight");
    write_u32(&mut buf, 2);
    write_u64(&mut buf, 3);
    write_u64(&mut buf, 2);
    write_u32(&mut buf, 0);
    write_u64(&mut buf, weight_offset);

    let bias_offset: u64 = (6 * 4) as u64; // 24 bytes for 6 F32 values
    write_string(&mut buf, "bias");
    write_u32(&mut buf, 1);
    write_u64(&mut buf, 4);
    write_u32(&mut buf, 1);
    write_u64(&mut buf, bias_offset);

    pad_to(&mut buf, 32);

    let weight_data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for v in &weight_data {
        write_f32_le(&mut buf, *v);
    }

    let bias_data: [half::f16; 4] = [
        half::f16::from_f32(0.5),
        half::f16::from_f32(1.0),
        half::f16::from_f32(1.5),
        half::f16::from_f32(2.0),
    ];
    for v in &bias_data {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    buf
}

fn build_q8_0_gguf() -> Vec<u8> {
    let mut buf = Vec::new();

    write_u32(&mut buf, 0x46554747);
    write_u32(&mut buf, 3);
    write_u64(&mut buf, 1);
    write_u64(&mut buf, 0);

    let offset: u64 = 0;
    write_string(&mut buf, "qweight");
    write_u32(&mut buf, 1);
    write_u64(&mut buf, 32);
    write_u32(&mut buf, 8);
    write_u64(&mut buf, offset);
    pad_to(&mut buf, 32);

    // Q8_0 block: 2 (f16 scale) + 32 (int8 quants) = 34 bytes
    let d = half::f16::from_f32(0.5);
    buf.extend_from_slice(&d.to_le_bytes());
    for i in 0..32i8 {
        buf.push(i as u8);
    }

    buf
}

#[test]
fn test_gguf_load_metadata_and_tensors() {
    let buf = build_simple_gguf();
    let tmp = std::env::temp_dir().join("aether_test_simple.gguf");
    std::fs::write(&tmp, &buf).unwrap();

    let model = GGUFLoader::load(tmp.to_str().unwrap()).unwrap();

    assert_eq!(model.metadata.len(), 2);
    match model.metadata.get("general.name").unwrap() {
        GGUFValue::String(s) => assert_eq!(s, "test-model"),
        _ => panic!("Expected String"),
    }
    match model.metadata.get("test.int").unwrap() {
        GGUFValue::Int32(v) => assert_eq!(*v, 42),
        _ => panic!("Expected Int32"),
    }

    assert_eq!(model.tensors.len(), 2);

    let weight = model.tensors.get("weight").unwrap();
    assert_eq!(weight.name, "weight");
    assert_eq!(weight.shape, vec![3, 2]);
    assert_eq!(weight.dtype, GGUFDtype::F32);
    let w_deq = dequantize(&weight.data, weight.dtype, &weight.shape);
    assert_eq!(w_deq, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let bias = model.tensors.get("bias").unwrap();
    assert_eq!(bias.name, "bias");
    assert_eq!(bias.shape, vec![4]);
    assert_eq!(bias.dtype, GGUFDtype::F16);
    let b_deq = dequantize(&bias.data, bias.dtype, &bias.shape);
    assert_eq!(b_deq, vec![0.5, 1.0, 1.5, 2.0]);

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_gguf_invalid_magic() {
    let mut buf = build_simple_gguf();
    buf[0] = 0x00;
    let tmp = std::env::temp_dir().join("aether_test_bad_magic.gguf");
    std::fs::write(&tmp, &buf).unwrap();
    let result = GGUFLoader::load(tmp.to_str().unwrap());
    assert!(result.is_err());
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_gguf_file_not_found() {
    let result = GGUFLoader::load("/nonexistent/path/to/model.gguf");
    assert!(result.is_err());
}

// ── Malformed-input hardening: every case below must return Err ─────────────
// (previously: OOM abort, unbounded CPU burn, `% 0` panic, or wrapping
// arithmetic that bypassed the bounds check in release builds).

static MALFORMED_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn load_bytes(buf: &[u8]) -> Result<GGUFModel, aether::Error> {
    let n = MALFORMED_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = std::env::temp_dir().join(format!("aether_test_malformed_{n}.gguf"));
    std::fs::write(&tmp, buf).unwrap();
    let result = GGUFLoader::load(tmp.to_str().unwrap());
    std::fs::remove_file(&tmp).ok();
    result
}

fn header_with_counts(tensor_count: u64, metadata_kv_count: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    write_u32(&mut buf, 0x46554747);
    write_u32(&mut buf, 3);
    write_u64(&mut buf, tensor_count);
    write_u64(&mut buf, metadata_kv_count);
    buf
}

#[test]
fn test_gguf_rejects_huge_string_len() {
    // First metadata key claims a 4 GiB length but the file is 32 bytes.
    let mut buf = header_with_counts(0, 1);
    write_u64(&mut buf, 0x1_0000_0000);
    buf.extend_from_slice(b"short");
    assert!(load_bytes(&buf).is_err());
}

#[test]
fn test_gguf_rejects_huge_metadata_count() {
    let buf = header_with_counts(0, u64::MAX);
    assert!(load_bytes(&buf).is_err());
}

#[test]
fn test_gguf_rejects_huge_tensor_count() {
    let buf = header_with_counts(u64::MAX, 0);
    assert!(load_bytes(&buf).is_err());
}

#[test]
fn test_gguf_rejects_huge_array_len() {
    let mut buf = header_with_counts(0, 1);
    write_string(&mut buf, "key");
    write_u32(&mut buf, 9); // Array
    write_u32(&mut buf, 8); // element type String
    write_u64(&mut buf, u64::MAX);
    assert!(load_bytes(&buf).is_err());
}

#[test]
fn test_gguf_rejects_absurd_tensor_dims() {
    let mut buf = header_with_counts(1, 0);
    write_string(&mut buf, "w");
    write_u32(&mut buf, 1000); // n_dims
    assert!(load_bytes(&buf).is_err());
}

#[test]
fn test_gguf_rejects_tensor_offset_oob() {
    // Valid header/shape, but the data offset points past EOF.
    let mut buf = header_with_counts(1, 0);
    write_string(&mut buf, "w");
    write_u32(&mut buf, 1); // n_dims
    write_u64(&mut buf, 4); // dim
    write_u32(&mut buf, 0); // F32
    write_u64(&mut buf, u64::MAX); // offset
    while !buf.len().is_multiple_of(32) {
        buf.push(0);
    }
    assert!(load_bytes(&buf).is_err());
}

#[test]
fn test_gguf_zero_alignment_falls_back_to_default() {
    // general.alignment = 0 previously panicked with `% 0`.
    let mut buf = header_with_counts(0, 1);
    write_string(&mut buf, "general.alignment");
    write_u32(&mut buf, 4); // Uint32
    write_u32(&mut buf, 0);
    let model = load_bytes(&buf).expect("alignment=0 must fall back to 32");
    assert_eq!(model.tensors.len(), 0);
}

#[test]
fn test_gguf_negative_alignment_ignored() {
    // Int32(-8) previously wrapped to u64::MAX via `as` cast.
    let mut buf = header_with_counts(0, 1);
    write_string(&mut buf, "general.alignment");
    write_u32(&mut buf, 5); // Int32
    write_u32(&mut buf, (-8i32) as u32);
    let model = load_bytes(&buf).expect("negative alignment must fall back to 32");
    assert_eq!(model.tensors.len(), 0);
}

#[test]
fn test_gguf_truncated_file() {
    let mut buf = build_simple_gguf();
    buf.truncate(buf.len() / 2);
    assert!(load_bytes(&buf).is_err());
}

#[test]
fn test_gguf_empty_file() {
    assert!(load_bytes(&[]).is_err());
    assert!(load_bytes(&[0u8; 3]).is_err());
}

// ── Config validation ───────────────────────────────────────────────────────

fn minimal_config_meta(heads: i64, layers: i64, d_model: i64) -> HashMap<String, GGUFValue> {
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".to_string(),
        GGUFValue::String("llama".to_string()),
    );
    meta.insert(
        "llama.embedding_length".to_string(),
        GGUFValue::Int64(d_model),
    );
    meta.insert("llama.block_count".to_string(), GGUFValue::Int64(layers));
    meta.insert(
        "llama.attention.head_count".to_string(),
        GGUFValue::Int64(heads),
    );
    meta.insert(
        "llama.feed_forward_length".to_string(),
        GGUFValue::Int64(64),
    );
    meta
}

fn config_from_meta(meta: HashMap<String, GGUFValue>) -> Result<(), aether::Error> {
    use aether::loader::gguf::{GGUFTensor, SharedBytes};
    let mut tensors = HashMap::new();
    // Minimal 2-D token embedding so vocab size resolves.
    tensors.insert(
        "token_embd.weight".to_string(),
        GGUFTensor {
            name: "token_embd.weight".to_string(),
            shape: vec![8, 4],
            dtype: GGUFDtype::F32,
            data: SharedBytes::new_owned(vec![0u8; 8 * 4 * 4]),
        },
    );
    let model = GGUFModel {
        metadata: meta,
        tensors,
    };
    aether::inference::model_loader::LlamaConfig::from_gguf(&model).map(|_| ())
}

#[test]
fn test_config_rejects_zero_heads() {
    // head_count = 0 previously panicked with divide-by-zero.
    assert!(config_from_meta(minimal_config_meta(0, 2, 32)).is_err());
}

#[test]
fn test_config_rejects_zero_layers() {
    assert!(config_from_meta(minimal_config_meta(4, 0, 32)).is_err());
}

#[test]
fn test_config_rejects_negative_dims() {
    // Negative Int64 previously wrapped to huge usize via `as` casts.
    assert!(config_from_meta(minimal_config_meta(-1, 2, 32)).is_err());
    assert!(config_from_meta(minimal_config_meta(4, 2, -64)).is_err());
}

#[test]
fn test_config_rejects_incompatible_head_dim() {
    // d_model=30 is not divisible by 4 heads.
    assert!(config_from_meta(minimal_config_meta(4, 2, 30)).is_err());
}

#[test]
fn test_config_rejects_more_kv_heads_than_heads() {
    let mut meta = minimal_config_meta(4, 2, 32);
    meta.insert(
        "llama.attention.head_count_kv".to_string(),
        GGUFValue::Int64(8),
    );
    assert!(config_from_meta(meta).is_err());
}

#[test]
fn test_dequant_f32() {
    let data = bytemuck::cast_slice::<f32, u8>(&[1.0, 2.0, 3.0, 4.0]).to_vec();
    let result = dequantize(&data, GGUFDtype::F32, &[4]);
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_dequant_f16() {
    let f16_vals: Vec<half::f16> = vec![
        half::f16::from_f32(0.5),
        half::f16::from_f32(-1.0),
        half::f16::from_f32(2.5),
    ];
    let data: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let result = dequantize(&data, GGUFDtype::F16, &[3]);
    assert!((result[0] - 0.5).abs() < 1e-3);
    assert!((result[1] - (-1.0)).abs() < 1e-3);
    assert!((result[2] - 2.5).abs() < 1e-3);
}

#[test]
fn test_dequant_q8_0() {
    // Q8_0 block: 2 bytes f16 scale + 32 bytes int8 quants = 34 bytes per 32-element block
    let d = half::f16::from_f32(0.5);
    let mut data = Vec::new();
    data.extend_from_slice(&d.to_le_bytes());
    for i in 0..32i8 {
        data.push(i as u8);
    }
    let result = dequantize(&data, GGUFDtype::Q8_0, &[32]);
    assert_eq!(result.len(), 32);
    for i in 0..32i8 {
        let expected = (i as f32) * 0.5;
        assert!((result[i as usize] - expected).abs() < 1e-3);
    }
}

#[test]
fn test_dequant_q8_0_multiblock() {
    let d = half::f16::from_f32(1.0);
    let mut data = Vec::new();
    data.extend_from_slice(&d.to_le_bytes());
    for i in 0..32i8 {
        data.push(i as u8);
    }
    // Second block: scale=2.0
    let d2 = half::f16::from_f32(2.0);
    data.extend_from_slice(&d2.to_le_bytes());
    for i in 0..32i8 {
        data.push((i + 10) as u8);
    }

    let result = dequantize(&data, GGUFDtype::Q8_0, &[64]);
    assert_eq!(result.len(), 64);
    for i in 0..32i8 {
        assert!((result[i as usize] - (i as f32)).abs() < 1e-3);
    }
    for i in 0..32i8 {
        let expected = ((i + 10) as f32) * 2.0;
        assert!((result[32 + i as usize] - expected).abs() < 1e-3);
    }
}

#[test]
fn test_dequant_i8() {
    let data: Vec<u8> = vec![0u8, 255, 128, 1];
    let result = dequantize(&data, GGUFDtype::I8, &[4]);
    assert_eq!(result, vec![0.0, -1.0, -128.0, 1.0]);
}

#[test]
fn test_dequant_i16() {
    let i16_vals: [i16; 3] = [-100, 0, 255];
    let data = bytemuck::cast_slice::<i16, u8>(&i16_vals).to_vec();
    let result = dequantize(&data, GGUFDtype::I16, &[3]);
    assert_eq!(result, vec![-100.0, 0.0, 255.0]);
}

#[test]
fn test_dequant_i32() {
    let i32_vals: [i32; 3] = [-100000, 0, 99999];
    let data = bytemuck::cast_slice::<i32, u8>(&i32_vals).to_vec();
    let result = dequantize(&data, GGUFDtype::I32, &[3]);
    assert_eq!(result, vec![-100000.0, 0.0, 99999.0]);
}

#[test]
fn test_dequant_unsupported_fallback() {
    let data = vec![1, 2, 3, 4];
    let result = dequantize(&data, GGUFDtype::Q4_1, &[4]);
    assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_load_with_q8_0_tensor() {
    let buf = build_q8_0_gguf();
    let tmp = std::env::temp_dir().join("aether_test_q8_0.gguf");
    std::fs::write(&tmp, &buf).unwrap();

    let model = GGUFLoader::load(tmp.to_str().unwrap()).unwrap();
    assert_eq!(model.tensors.len(), 1);

    let t = model.tensors.get("qweight").unwrap();
    assert_eq!(t.shape, vec![32]);
    assert_eq!(t.dtype, GGUFDtype::Q8_0);

    let deq = dequantize(&t.data, t.dtype, &t.shape);
    assert_eq!(deq.len(), 32);
    for (i, &v) in deq.iter().enumerate() {
        assert!((v - (i as f32) * 0.5).abs() < 1e-3);
    }

    std::fs::remove_file(&tmp).ok();
}
