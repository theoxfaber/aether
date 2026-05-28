/// Dequantization routines for GGUF quantized tensor formats.
///
/// Converts quantized types to F32 for execution in Aether's graph runtime.
use crate::loader::gguf::GGUFDtype;

/// Dequantize and transpose a quantized tensor from GGUF [in, out] row-major
/// order to [out, in] order (what the matmul kernels expect).
///
/// For symmetric tensors (in == out) this is effectively a no-op — callers
/// should skip it. The transpose uses blocking for cache efficiency.
/// Parallelized with rayon across output rows.
pub fn dequantize_transpose_f32(
    data: &[u8],
    dtype: GGUFDtype,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let total = in_features * out_features;
    // Dequantize using GGUF shape [in, out] → flat f32 in GGUF row-major order
    let flat = dequantize(data, dtype, &[in_features, out_features]);
    let mut transposed = vec![0.0f32; total];
    // Block transpose for cache efficiency
    let block = 64usize;
    for i0 in (0..in_features).step_by(block) {
        let i_end = (i0 + block).min(in_features);
        for j0 in (0..out_features).step_by(block) {
            let j_end = (j0 + block).min(out_features);
            for i in i0..i_end {
                let flat_row = &flat[i * out_features..];
                let tgt_base = &mut transposed[j0 * in_features + i..];
                let j_len = j_end - j0;
                for j in 0..j_len {
                    tgt_base[j * in_features] = flat_row[j0 + j];
                }
            }
        }
    }
    transposed
}

/// Dequantize a tensor from any supported format to F32.
pub fn dequantize(data: &[u8], dtype: GGUFDtype, shape: &[usize]) -> Vec<f32> {
    match dtype {
        GGUFDtype::F32 => dequant_f32(data, shape),
        GGUFDtype::F16 => dequant_f16(data, shape),
        GGUFDtype::Q8_0 => dequant_q8_0(data, shape),
        GGUFDtype::Q4_0 => dequant_q4_0(data, shape),
        GGUFDtype::Q4_K => dequant_q4_k(data, shape),
        GGUFDtype::Q5_K => dequant_q5_k(data, shape),
        GGUFDtype::Q6_K => dequant_q6_k(data, shape),
        GGUFDtype::Q2_K => dequant_q2_k(data, shape),
        GGUFDtype::Q3_K => dequant_q3_k(data, shape),
        GGUFDtype::Q8_K => dequant_q8_k(data, shape),
        GGUFDtype::I8 => dequant_i8(data, shape),
        GGUFDtype::I16 => dequant_i16(data, shape),
        GGUFDtype::I32 => dequant_i32(data, shape),
        _ => {
            // Fallback: return zeros for unsupported quant types
            let n: usize = shape.iter().product();
            vec![0.0; n]
        }
    }
}

pub fn dequant_f32(data: &[u8], shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut out = vec![0.0f32; n];
    if (data.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()) {
        let src = bytemuck::cast_slice::<u8, f32>(data);
        let copy_len = n.min(src.len());
        out[..copy_len].copy_from_slice(&src[..copy_len]);
    } else {
        let copy_len = n.min(data.len() / 4);
        for i in 0..copy_len {
            let offset = i * 4;
            out[i] = f32::from_ne_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
        }
    }
    out
}

pub fn dequant_f16(data: &[u8], _shape: &[usize]) -> Vec<f32> {
    if (data.as_ptr() as usize).is_multiple_of(std::mem::align_of::<half::f16>()) {
        let src = bytemuck::cast_slice::<u8, half::f16>(data);
        src.iter().map(|&x| x.to_f32()).collect()
    } else {
        let src_len = data.len() / 2;
        let mut out = Vec::with_capacity(src_len);
        for i in 0..src_len {
            let offset = i * 2;
            let bits = u16::from_ne_bytes([data[offset], data[offset + 1]]);
            out.push(half::f16::from_bits(bits).to_f32());
        }
        out
    }
}

pub fn dequant_q8_0(data: &[u8], shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let block_size = 32usize;
    let num_blocks = n.div_ceil(block_size);
    let mut out = vec![0.0f32; n];

    for block_idx in 0..num_blocks {
        let offset = block_idx * 34; // 2 (d as f16) + 32 (quants as int8)
        if offset + 34 > data.len() {
            break;
        }
        // Read scale (f16)
        let d = half::f16::from_le_bytes([data[offset], data[offset + 1]]).to_f32();
        let quants_offset = offset + 2;
        for i in 0..block_size {
            let out_idx = block_idx * block_size + i;
            if out_idx >= n {
                break;
            }
            let q = data[quants_offset + i] as i8 as f32;
            out[out_idx] = q * d;
        }
    }
    out
}

pub fn dequant_q4_0(data: &[u8], shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let block_size = 32usize;
    let num_blocks = n.div_ceil(block_size);
    let mut out = vec![0.0f32; n];

    for block_idx in 0..num_blocks {
        let offset = block_idx * 18;
        if offset + 18 > data.len() {
            break;
        }
        let d = half::f16::from_le_bytes([data[offset], data[offset + 1]]).to_f32();
        for i in 0..block_size {
            let out_idx = block_idx * block_size + i;
            if out_idx >= n {
                break;
            }
            let byte = data[offset + 2 + i / 2];
            let q = if i % 2 == 0 {
                (byte & 0x0F) as i8 - 8
            } else {
                ((byte >> 4) & 0x0F) as i8 - 8
            };
            out[out_idx] = (q as f32) * d;
        }
    }
    out
}

const BLOCK_SIZE: usize = 256;

/// Helper: decode f16 with NaN/Inf fallback to u16/65536 fraction.
fn decode_d(val: u16) -> f32 {
    let f = half::f16::from_bits(val).to_f32();
    if f.is_nan() || f.is_infinite() {
        val as f32 / 65536.0
    } else {
        f
    }
}

/// Q4_K dequantization (QK_K=256, matching ggml's modern Q4_K).
///
/// Block (144 bytes):
///   [0..1]    d        f16 super-block scale
///   [2..3]    dmin     f16 super-block min
///   [4..15]   scales[12], 6-bit packed (see get_scale_min_k4)
///   [16..143] qs[128], 256 × 4-bit quants packed 2 per byte
///
/// scales[12] layout:
///   scales[0..3]: low 6 bits = sc[0..3], high 2 bits = sc[4..7] high bits
///   scales[4..7]: low 6 bits = mm[0..3], high 2 bits = mm[4..7] high bits
///   scales[8..11]: low nibble = sc[4..7] low 4 bits, high nibble = mm[4..7] low 4 bits
const Q4K_BLOCK_SIZE: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144;

fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let mm = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, mm)
    }
}

pub fn dequant_q4_k(data: &[u8], shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let num_blocks = n / Q4K_BLOCK_SIZE;
    let mut out = vec![0.0f32; n];

    for bi in 0..num_blocks {
        let bo = bi * Q4K_BLOCK_BYTES;
        let d = decode_d(u16::from_le_bytes([data[bo], data[bo + 1]]));
        let dmin = decode_d(u16::from_le_bytes([data[bo + 2], data[bo + 3]]));

        let scales = &data[bo + 4..bo + 16];
        let mut qs = &data[bo + 16..bo + 144];

        let mut is = 0usize;
        for _j in (0..Q4K_BLOCK_SIZE).step_by(64) {
            let (sc0, mm0) = get_scale_min_k4(is, scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * mm0 as f32;
            let (sc1, mm1) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * mm1 as f32;

            for l in 0..32 {
                out[bi * Q4K_BLOCK_SIZE + is * 32 + l] = d1 * (qs[l] & 0x0F) as f32 - m1;
                out[bi * Q4K_BLOCK_SIZE + (is + 1) * 32 + l] =
                    d2 * ((qs[l] >> 4) & 0x0F) as f32 - m2;
            }
            qs = &qs[32..];
            is += 2;
        }
    }
    out
}

pub fn dequant_q5_k(data: &[u8], shape: &[usize]) -> Vec<f32> {
    const Q5K_BLOCK_SIZE: usize = 256;
    const Q5K_BLOCK_BYTES: usize = 176;
    let n: usize = shape.iter().product();
    let num_blocks = n / Q5K_BLOCK_SIZE;
    let mut out = vec![0.0f32; n];

    for bi in 0..num_blocks {
        let bo = bi * Q5K_BLOCK_BYTES;
        if bo + Q5K_BLOCK_BYTES > data.len() {
            break;
        }
        let d = decode_d(u16::from_le_bytes([data[bo], data[bo + 1]]));
        let dmin = decode_d(u16::from_le_bytes([data[bo + 2], data[bo + 3]]));

        let scales = &data[bo + 4..bo + 16];
        let qh = &data[bo + 16..bo + 48];
        let qs = &data[bo + 48..bo + 176];

        let mut qs_offset = 0usize;
        let mut is = 0usize;
        for _j in (0..256).step_by(64) {
            let (sc0, mm0) = get_scale_min_k4(is, scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * mm0 as f32;
            let (sc1, mm1) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * mm1 as f32;

            for l in 0..32 {
                let byte_l = qs[qs_offset + l];
                let ql_lo = byte_l & 0x0F;
                let ql_hi = (byte_l >> 4) & 0x0F;
                let qh_byte_lo = qh[(is * 32 + l) / 8];
                let qh_bit_lo = (qh_byte_lo >> ((is * 32 + l) % 8)) & 1;

                let qh_byte_hi = qh[((is + 1) * 32 + l) / 8];
                let qh_bit_hi = (qh_byte_hi >> (((is + 1) * 32 + l) % 8)) & 1;

                let q_lo = (ql_lo | (qh_bit_lo << 4)) as f32;
                let q_hi = (ql_hi | (qh_bit_hi << 4)) as f32;

                let idx_lo = bi * 256 + is * 32 + l;
                let idx_hi = idx_lo + 32;
                if idx_lo < n {
                    out[idx_lo] = d1 * q_lo - m1;
                }
                if idx_hi < n {
                    out[idx_hi] = d2 * q_hi - m2;
                }
            }
            qs_offset += 32;
            is += 2;
        }
    }
    out
}

/// Q6_K dequantization.
/// QK_K=256, 210 bytes/block.
///
/// This GGUF file uses the Python gguf library's Q6_K block layout:
///   [0..127]  ql[128]    256 × 4 low bits, packed 2 per byte
///   [128..191] qh[64]    256 × 2 high bits, packed 4 per byte
///   [192..207] scales[16] int8, one per 16-element sub-block
///   [208..209] d          f16 super-block scale
///
/// Dequantization follows ggml's shuffle pattern (two 128-element halves,
/// each grouped as 4 interleaved 32-element stripes sharing one qh byte).
const Q6K_BLOCK_SIZE: usize = 256;
const Q6K_BLOCK_BYTES: usize = 210;

pub fn dequant_q6_k(data: &[u8], shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let num_blocks = n / Q6K_BLOCK_SIZE;
    let mut out = vec![0.0f32; n];

    for bi in 0..num_blocks {
        let bo = bi * Q6K_BLOCK_BYTES;
        // d is at the END of the block (Python library layout)
        let d = decode_d(u16::from_le_bytes([data[bo + 208], data[bo + 209]]));

        let base = bi * Q6K_BLOCK_SIZE;
        let ql = &data[bo..bo + 128];
        let qh = &data[bo + 128..bo + 192];
        let sc = &data[bo + 192..bo + 208];

        // two halves of 128 elements each
        for half in 0..2 {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let sc_off = half * 8;

            for l in 0..32 {
                let is = l / 16;

                let ql_l = ql[ql_off + l] as u32;
                let ql_l32 = ql[ql_off + l + 32] as u32;
                let qh_l = qh[qh_off + l] as u32;

                let q1 = ((ql_l & 0x0F) | ((qh_l & 0x03) << 4)) as i32 - 32;
                let q2 = ((ql_l32 & 0x0F) | ((qh_l & 0x0C) << 2)) as i32 - 32;
                let q3 = ((ql_l >> 4) | (qh_l & 0x30)) as i32 - 32;
                let q4 = ((ql_l32 >> 4) | ((qh_l & 0xC0) >> 2)) as i32 - 32;

                let s1 = sc[sc_off + is] as i8 as f32;
                let s2 = sc[sc_off + is + 2] as i8 as f32;
                let s3 = sc[sc_off + is + 4] as i8 as f32;
                let s4 = sc[sc_off + is + 6] as i8 as f32;

                let hbase = base + half * 128;
                out[hbase + l] = d * s1 * q1 as f32;
                out[hbase + l + 32] = d * s2 * q2 as f32;
                out[hbase + l + 64] = d * s3 * q3 as f32;
                out[hbase + l + 96] = d * s4 * q4 as f32;
            }
        }
    }
    out
}

pub fn dequant_q2_k(data: &[u8], shape: &[usize]) -> Vec<f32> {
    const BLOCK_BYTES: usize = 84;
    const BLOCK_SIZE: usize = 256;
    let n: usize = shape.iter().product();
    let mut out = vec![0.0f32; n];
    let num_blocks = n.div_ceil(BLOCK_SIZE);
    for bi in 0..num_blocks {
        let bo = bi * BLOCK_BYTES;
        if bo + BLOCK_BYTES > data.len() {
            break;
        }
        let d = decode_d(u16::from_le_bytes([data[bo], data[bo + 1]]));
        let dmin = decode_d(u16::from_le_bytes([data[bo + 2], data[bo + 3]]));
        let scales = &data[bo + 4..bo + 20];
        let qs = &data[bo + 20..bo + 84];
        let mut out_off = bi * BLOCK_SIZE;
        // 16 sub-blocks of 16 elements, in 2 halves of 128
        for half in 0..2usize {
            let q_off = half * 32;
            let mut is = half * 8;
            for shift in [0i32, 2, 4, 6] {
                let sc = scales[is];
                let dl = d * (sc & 0xF) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16usize {
                    let q = ((qs[q_off + l] >> shift) & 3) as i8;
                    out[out_off] = dl * q as f32 - ml;
                    out_off += 1;
                }
                is += 1;
                let sc = scales[is];
                let dl = d * (sc & 0xF) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16usize {
                    let q = ((qs[q_off + 16 + l] >> shift) & 3) as i8;
                    out[out_off] = dl * q as f32 - ml;
                    out_off += 1;
                }
                is += 1;
            }
        }
    }
    out
}

pub fn dequant_q3_k(data: &[u8], shape: &[usize]) -> Vec<f32> {
    const BLOCK_BYTES: usize = 110;
    const BLOCK_SIZE: usize = 256;
    let n: usize = shape.iter().product();
    let mut out = vec![0.0f32; n];
    let num_blocks = n.div_ceil(BLOCK_SIZE);
    for bi in 0..num_blocks {
        let bo = bi * BLOCK_BYTES;
        if bo + BLOCK_BYTES > data.len() {
            break;
        }
        let hmask = &data[bo..bo + 32];
        let qs = &data[bo + 32..bo + 96];
        let sr = &data[bo + 96..bo + 108];
        let d = decode_d(u16::from_le_bytes([data[bo + 108], data[bo + 109]]));

        // Unpack 16 × 6-bit scales → centered [-32, 31]
        let mut sc = [0i16; 16];
        for i in 0..8 {
            sc[i] = (sr[i] & 0x0F) as i16;
            sc[i + 8] = (sr[i] >> 4) as i16;
        }
        for i in 0..4 {
            let b = sr[i + 8];
            sc[i] |= (b as i16 & 0x03) << 4;
            sc[i + 4] |= ((b >> 2) as i16 & 0x03) << 4;
            sc[i + 8] |= ((b >> 4) as i16 & 0x03) << 4;
            sc[i + 12] |= ((b >> 6) as i16 & 0x03) << 4;
        }
        for s in &mut sc {
            *s -= 32;
        }

        let mut out_off = bi * BLOCK_SIZE;
        // 16 sub-blocks of 16 elements each (s=0..15)
        // qs: 64 bytes, split into 2 halves of 32 (for s=0..7 and s=8..15)
        // hmask: 32 bytes total (256 bits), indexed as:
        //   byte = (s % 2) * 16 + elem  (always 0..31)
        //   bit  = s / 2                 (0..7)
        for s in 0..16usize {
            let half = s / 8;
            let q_off = half * 32;
            let shift = (s as i32 / 2 % 4) * 2; // 0,2,4,6 per sub-block pair
            let dl = d * sc[s] as f32;
            let hm_byte_base = (s % 2) * 16;
            let hm_bit = s as u8 / 2;
            for e in 0..16usize {
                let ql = (qs[q_off + hm_byte_base + e] >> shift) & 3;
                let qh = ((hmask[hm_byte_base + e] >> hm_bit) & 1) ^ 1;
                let q = (ql as i8) - ((qh as i8) << 2);
                out[out_off] = dl * q as f32;
                out_off += 1;
            }
        }
    }
    out
}

pub fn dequant_q8_k(data: &[u8], shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut out = vec![0.0f32; n];
    let num_blocks = n.div_ceil(BLOCK_SIZE);
    for block_idx in 0..num_blocks {
        let bo = block_idx * 136;
        if bo + 136 > data.len() {
            break;
        }
        for j in 0..BLOCK_SIZE {
            let out_idx = block_idx * BLOCK_SIZE + j;
            if out_idx >= n {
                break;
            }
            let q = data[bo + 8 + j] as i8;
            let d = f32::from(half::f16::from_le_bytes([data[bo], data[bo + 1]]));
            out[out_idx] = (q as f32) * d;
        }
    }
    out
}

pub fn dequant_i8(data: &[u8], _shape: &[usize]) -> Vec<f32> {
    data.iter().map(|&x| x as i8 as f32).collect()
}

pub fn dequant_i16(data: &[u8], _shape: &[usize]) -> Vec<f32> {
    if (data.as_ptr() as usize).is_multiple_of(std::mem::align_of::<i16>()) {
        let src = bytemuck::cast_slice::<u8, i16>(data);
        src.iter().map(|&x| x as f32).collect()
    } else {
        let src_len = data.len() / 2;
        let mut out = Vec::with_capacity(src_len);
        for i in 0..src_len {
            let offset = i * 2;
            let val = i16::from_ne_bytes([data[offset], data[offset + 1]]);
            out.push(val as f32);
        }
        out
    }
}

pub fn dequant_i32(data: &[u8], _shape: &[usize]) -> Vec<f32> {
    if (data.as_ptr() as usize).is_multiple_of(std::mem::align_of::<i32>()) {
        let src = bytemuck::cast_slice::<u8, i32>(data);
        src.iter().map(|&x| x as f32).collect()
    } else {
        let src_len = data.len() / 4;
        let mut out = Vec::with_capacity(src_len);
        for i in 0..src_len {
            let offset = i * 4;
            let val = i32::from_ne_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            out.push(val as f32);
        }
        out
    }
}

pub fn dequant_q4_k_block(data: &[u8]) -> [f32; 256] {
    let mut out = [0.0f32; 256];
    let d = decode_d(u16::from_le_bytes([data[0], data[1]]));
    let dmin = decode_d(u16::from_le_bytes([data[2], data[3]]));

    let scales = &data[4..16];
    let mut qs = &data[16..144];

    let mut is = 0usize;
    for _j in (0..256).step_by(64) {
        let (sc0, mm0) = get_scale_min_k4(is, scales);
        let d1 = d * sc0 as f32;
        let m1 = dmin * mm0 as f32;
        let (sc1, mm1) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc1 as f32;
        let m2 = dmin * mm1 as f32;

        for l in 0..32 {
            out[is * 32 + l] = d1 * (qs[l] & 0x0F) as f32 - m1;
            out[(is + 1) * 32 + l] = d2 * ((qs[l] >> 4) & 0x0F) as f32 - m2;
        }
        qs = &qs[32..];
        is += 2;
    }
    out
}

pub fn dequant_q5_k_block(data: &[u8]) -> [f32; 256] {
    let mut out = [0.0f32; 256];
    let d = decode_d(u16::from_le_bytes([data[0], data[1]]));
    let dmin = decode_d(u16::from_le_bytes([data[2], data[3]]));

    let scales = &data[4..16];
    let qh = &data[16..48];
    let qs = &data[48..176];

    let mut qs_offset = 0usize;
    let mut is = 0usize;
    for _j in (0..256).step_by(64) {
        let (sc0, mm0) = get_scale_min_k4(is, scales);
        let d1 = d * sc0 as f32;
        let m1 = dmin * mm0 as f32;
        let (sc1, mm1) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc1 as f32;
        let m2 = dmin * mm1 as f32;

        for l in 0..32 {
            let byte_l = qs[qs_offset + l];
            let ql_lo = byte_l & 0x0F;
            let ql_hi = (byte_l >> 4) & 0x0F;

            let qh_byte_lo = qh[(is * 32 + l) / 8];
            let qh_bit_lo = (qh_byte_lo >> ((is * 32 + l) % 8)) & 1;

            let qh_byte_hi = qh[((is + 1) * 32 + l) / 8];
            let qh_bit_hi = (qh_byte_hi >> (((is + 1) * 32 + l) % 8)) & 1;

            let q_lo = (ql_lo | (qh_bit_lo << 4)) as f32;
            let q_hi = (ql_hi | (qh_bit_hi << 4)) as f32;

            out[is * 32 + l] = d1 * q_lo - m1;
            out[(is + 1) * 32 + l] = d2 * q_hi - m2;
        }
        qs_offset += 32;
        is += 2;
    }
    out
}

pub fn dequant_q6_k_block(data: &[u8]) -> [f32; 256] {
    let mut out = [0.0f32; 256];
    let d = decode_d(u16::from_le_bytes([data[208], data[209]]));

    let ql = &data[0..128];
    let qh = &data[128..192];
    let sc = &data[192..208];

    for half in 0..2 {
        let ql_off = half * 64;
        let qh_off = half * 32;
        let sc_off = half * 8;

        for l in 0..32 {
            let is = l / 16;

            let ql_l = ql[ql_off + l] as u32;
            let ql_l32 = ql[ql_off + l + 32] as u32;
            let qh_l = qh[qh_off + l] as u32;

            let q1 = ((ql_l & 0x0F) | ((qh_l & 0x03) << 4)) as i32 - 32;
            let q2 = ((ql_l32 & 0x0F) | ((qh_l & 0x0C) << 2)) as i32 - 32;
            let q3 = ((ql_l >> 4) | (qh_l & 0x30)) as i32 - 32;
            let q4 = ((ql_l32 >> 4) | ((qh_l & 0xC0) >> 2)) as i32 - 32;

            let s1 = sc[sc_off + is] as i8 as f32;
            let s2 = sc[sc_off + is + 2] as i8 as f32;
            let s3 = sc[sc_off + is + 4] as i8 as f32;
            let s4 = sc[sc_off + is + 6] as i8 as f32;

            let hbase = half * 128;
            out[hbase + l] = d * s1 * q1 as f32;
            out[hbase + l + 32] = d * s2 * q2 as f32;
            out[hbase + l + 64] = d * s3 * q3 as f32;
            out[hbase + l + 96] = d * s4 * q4 as f32;
        }
    }
    out
}
