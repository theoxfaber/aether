use std::io::Write;

use aether::loader::dequant::dequantize;
use aether::loader::gguf::{GGUFDtype, GGUFLoader, GGUFValue};

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
    while w.len() % align != 0 {
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
    for i in 0..32 {
        assert!((deq[i] - (i as f32) * 0.5).abs() < 1e-3);
    }

    std::fs::remove_file(&tmp).ok();
}
