#![allow(clippy::erasing_op, clippy::identity_op)]
use aether::loader::dequant::dequantize;
/// Verify Q6_K block layout by comparing dequantized vs expected f32 values.
use aether::loader::gguf::GGUFLoader;
use aether::Error;

const Q6K_BLOCK_SIZE: usize = 256;
const Q6K_BLOCK_BYTES: usize = 210;

fn decode_d(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    let f = half::f16::from_bits(bits).to_f32();
    if f.is_nan() || f.is_infinite() {
        bits as f32 / 65536.0
    } else {
        f
    }
}

/// Dequantize one Q6_K block at byte offset `bo` from `data`, return 256 f32 values.
fn dequant_q6k_one_block(data: &[u8], bo: usize) -> Vec<f32> {
    let d = decode_d(data[bo + 208], data[bo + 209]);
    let ql = &data[bo..bo + 128];
    let qh = &data[bo + 128..bo + 192];
    let sc = &data[bo + 192..bo + 208];
    let mut out = vec![0.0f32; Q6K_BLOCK_SIZE];

    for half in 0..2 {
        let ql_off = half * 64;
        let qh_off = half * 32;
        let sc_off = half * 8;
        let hbase = half * 128;

        for l in 0..32 {
            let is = l / 16;
            let ql_l = ql[ql_off + l] as u32;
            let ql_l32 = ql[ql_off + l + 32] as u32;
            let qh_l = qh[qh_off + l] as u32;

            let q1 = ((ql_l & 0x0F) | ((qh_l & 0x03) << 4)) as i32 - 32;
            let q2 = ((ql_l32 & 0x0F) | ((qh_l & 0x0C) << 2)) as i32 - 32;
            let q3 = ((ql_l >> 4) | (qh_l & 0x30)) as i32 - 32;
            let q4 = ((ql_l32 >> 4) | ((qh_l & 0xC0) >> 2)) as i32 - 32;

            let s1 = sc[sc_off + is ] as i8 as f32;
            let s2 = sc[sc_off + is + 2] as i8 as f32;
            let s3 = sc[sc_off + is + 4] as i8 as f32;
            let s4 = sc[sc_off + is + 6] as i8 as f32;

            out[hbase + l] = d * s1 * q1 as f32;
            out[hbase + l + 32] = d * s2 * q2 as f32;
            out[hbase + l + 64] = d * s3 * q3 as f32;
            out[hbase + l + 96] = d * s4 * q4 as f32;
        }
    }
    out
}

fn main() -> Result<(), Error> {
    let gguf = GGUFLoader::load("mistral-7b-q4k.gguf")?;
    let output_tensor = gguf.tensors.get("output.weight").expect("output.weight");
    let emb_tensor = gguf
        .tensors
        .get("token_embd.weight")
        .expect("token_embd.weight");
    let d_model = 4096usize;
    let vocab_size = emb_tensor.shape[1]; // 32000

    eprintln!(
        "output.weight: shape={:?} dtype={:?}",
        output_tensor.shape, output_tensor.dtype
    );
    eprintln!("output.weight data len: {}", output_tensor.data.len());

    // Reference: dequantize entire tensor to f32 using the library's dequantize function
    let lm_f32_ref = dequantize(
        &output_tensor.data,
        output_tensor.dtype,
        &output_tensor.shape,
    );
    eprintln!("Dequantized ref: {} f32 values", lm_f32_ref.len());

    // The GGUF shape is [d_model, vocab_size] = [4096, 32000]
    // The ref is in row-major with shape [4096, 32000]:
    //   ref[d * vocab_size + v] = weight for dimension d, token v
    // We need: expected[v * d_model + d] = weight for token v, dimension d

    // Now manually dequantize one block and compare
    // The Q6_K data stores blocks in row-major order
    // For each row (dimension d of the [d_model, vocab_size] matrix), the data has
    // blocks_per_row = d_model / 256 = 16 blocks
    // For row d, block b: offset = d * 16 * 210 + b * 210

    let blocks_per_row = d_model / Q6K_BLOCK_SIZE; // 16

    // Let's check block 0 of row 0 (first 256 weights of dimension 0)
    let bo_row0_block0 = 0 * blocks_per_row * Q6K_BLOCK_BYTES + 0 * Q6K_BLOCK_BYTES;
    let block0 = dequant_q6k_one_block(&output_tensor.data, bo_row0_block0);

    // Compare with reference: ref[d=0, v=0..255] = lm_f32_ref[0 * vocab_size + 0..0 * vocab_size + 256]
    eprintln!("\n=== Comparing block 0 (first 256 values of dimension 0) ===");
    let mut max_diff = 0.0f64;
    for i in 0..Q6K_BLOCK_SIZE {
        let ref_val = lm_f32_ref[0 * vocab_size + i];
        let block_val = block0[i];
        let diff = (ref_val - block_val).abs() as f64;
        if diff > max_diff {
            max_diff = diff;
        }
        if i < 10 {
            eprintln!(
                "  idx={}: ref={:.6} block={:.6} diff={:.6}",
                i, ref_val, block_val, diff
            );
        }
    }
    eprintln!("  max_diff = {:.10}", max_diff);

    // Now check a later block
    eprintln!("\n=== Checking block at row=100, block_idx=0 ===");
    let row = 100;
    let bo = row * blocks_per_row * Q6K_BLOCK_BYTES + 0 * Q6K_BLOCK_BYTES;
    let block_r100 = dequant_q6k_one_block(&output_tensor.data, bo);
    max_diff = 0.0f64;
    for i in 0..Q6K_BLOCK_SIZE.min(10) {
        let ref_val = lm_f32_ref[row * vocab_size + i];
        let diff = (ref_val - block_r100[i]).abs() as f64;
        if diff > max_diff {
            max_diff = diff;
        }
        eprintln!(
            "  idx={}: ref={:.6} block={:.6} diff={:.6}",
            i, ref_val, block_r100[i], diff
        );
    }
    eprintln!("  first 10 max_diff = {:.10}", max_diff);

    // Also check block 0 of row 0 using the load_quant convention
    // load_quant reverses shape to [vocab_size, d_model] = [32000, 4096]
    // In this convention, row v has: v * blocks_per_row_g = v * (4096/256) = v * 16 blocks
    // Block data offset for row v, block b: v * 16 * 210 + b * 210
    //
    // The data layout in the output_tensor is [d_model, vocab_size] in row-major
    // Row d has blocks: d * 16 * 210 bytes
    // Column v of row d is at position v in that row
    //
    // In [vocab_size, d_model] convention:
    // Row v = vocab element v, columns = d_model dimensions
    // This is the TRANSPOSE: element (v, d) = element (d, v) of original
    // Block v*16+b in [vocab_size, d_model] = block d*16+b in [d_model, vocab_size]
    // But the BLOCK content is the same in both transposed layouts!
    // Actually no — in [d_model, vocab_size], block row d covers dimensions d*256..d*256+255
    // In [vocab_size, d_model], block row v covers vocabulary tokens v*256..v*256+255
    // These are different blocks with different data!

    // So the data layout in [vocab_size, d_model] is DIFFERENT
    // Let me verify with the load_quant convention:
    let _n_vocab = vocab_size; // 32000
    let n_dim = d_model; // 4096
    let blocks_per_row_q = n_dim / Q6K_BLOCK_SIZE; // 16 for each vocab row

    // For vocab row 0, block 0: offset = 0 * 16 * 210 + 0 * 210 = 0
    // This is the same physical offset as dimension row 0, block 0
    // But the MEANING is different:
    //   [d_model, vocab_size] row 0, block 0: dimensions 0..255 of vocabulary tokens 0..255
    //   [vocab_size, d_model] row 0, block 0: vocabulary token 0, dimensions 0..255
    // These are TRANSPOSED!

    // Let me verify: for vocab row 0, block 0 in [vocab_size, d_model] layout,
    // the values should be: lm_f32_ref[dim=0..255, v=0] = lm_f32_ref[0*vocab_size+0, ..., 255*vocab_size+0]
    eprintln!("\n=== Testing [vocab_size, d_model] convention ===");
    let vocab_row = 0;
    let bo_v = vocab_row * blocks_per_row_q * Q6K_BLOCK_BYTES + 0 * Q6K_BLOCK_BYTES;
    let block_v0 = dequant_q6k_one_block(&output_tensor.data, bo_v);

    // Expected: ref[dim=0..255][vocab=0] = lm_f32_ref[d * vocab_size + 0] for d=0..255
    eprintln!("First 10 values for vocab token 0:");
    max_diff = 0.0f64;
    for i in 0..Q6K_BLOCK_SIZE.min(10) {
        let ref_val = lm_f32_ref[i * vocab_size + vocab_row]; // dim=i, vocab=0
        let diff = (ref_val - block_v0[i]).abs() as f64;
        if diff > max_diff {
            max_diff = diff;
        }
        eprintln!(
            "  dim={}: ref={:.6} block={:.6} diff={:.6}",
            i, ref_val, block_v0[i], diff
        );
    }
    eprintln!("  max_diff = {:.10}", max_diff);

    // If max_diff is ~0, the [vocab_size, d_model] interpretation is correct
    // If max_diff is large, the [d_model, vocab_size] interpretation is correct

    Ok(())
}
