use std::fs::File;
/// Analyze Q4_K blocks to determine d/dmin encoding.
use std::io::{Read, Seek, SeekFrom};

fn read_u32<R: Read>(r: &mut R) -> u32 {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).unwrap();
    u32::from_le_bytes(b)
}
fn read_u64<R: Read>(r: &mut R) -> u64 {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).unwrap();
    u64::from_le_bytes(b)
}
fn read_string<R: Read>(r: &mut R) -> String {
    let l = read_u64(r) as usize;
    let mut b = vec![0u8; l];
    r.read_exact(&mut b).unwrap();
    String::from_utf8(b).unwrap()
}

fn main() {
    let mut f = File::open("tinyllama-q4.gguf").unwrap();
    let _file_len = f.metadata().unwrap().len();
    let _magic = read_u32(&mut f);
    let _version = read_u32(&mut f);
    let tensor_count = read_u64(&mut f);
    let metadata_kv_count = read_u64(&mut f);

    for _ in 0..metadata_kv_count {
        let _key = read_string(&mut f);
        let val_type = read_u32(&mut f);
        match val_type {
            0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 => {
                let mut b = [0u8; 4];
                f.read_exact(&mut b).unwrap();
            }
            8 => {
                let slen = read_u64(&mut f) as usize;
                let mut s = vec![0u8; slen];
                f.read_exact(&mut s).unwrap();
            }
            9 => {
                let elem_type = read_u32(&mut f);
                let alen = read_u64(&mut f) as usize;
                for _ in 0..alen {
                    match elem_type {
                        0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 => {
                            let mut b = [0u8; 4];
                            f.read_exact(&mut b).unwrap();
                        }
                        8 => {
                            let slen = read_u64(&mut f) as usize;
                            let mut s = vec![0u8; slen];
                            f.read_exact(&mut s).unwrap();
                        }
                        10 | 11 | 12 => {
                            let mut b = [0u8; 8];
                            f.read_exact(&mut b).unwrap();
                        }
                        _ => panic!("Unknown array elem type {}", elem_type),
                    }
                }
            }
            10 | 11 | 12 => {
                let mut b = [0u8; 8];
                f.read_exact(&mut b).unwrap();
            }
            _ => panic!("Unknown val_type {}", val_type),
        }
    }

    struct TInfo {
        name: String,
        dtype: u32,
        offset: u64,
    }
    let mut infos = Vec::new();
    for _ in 0..tensor_count {
        let name = read_string(&mut f);
        let n_dims = read_u32(&mut f) as usize;
        let mut _shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            _shape.push(read_u64(&mut f) as usize);
        }
        let dtype = read_u32(&mut f);
        let offset = read_u64(&mut f);
        infos.push(TInfo {
            name,
            dtype,
            offset,
        });
    }

    let info_end = f.stream_position().unwrap();
    let data_start = (info_end + 255) & !255;

    // Find Q weight (Q4_K, dtype=12) and read first few blocks
    for info in &infos {
        if info.name == "blk.0.attn_q.weight" && info.dtype == 12 {
            let abs_off = data_start + info.offset;
            f.seek(SeekFrom::Start(abs_off)).unwrap();
            let mut buf = vec![0u8; 72 * 10]; // first 10 blocks
            let n = f.read(&mut buf).unwrap();
            let n_blocks = n / 72;

            println!("output.weight: examining {} blocks\n", n_blocks);

            for bi in 0..n_blocks.min(10) {
                let bo = bi * 72;
                let d_bytes = [buf[bo], buf[bo + 1]];
                let dmin_bytes = [buf[bo + 2], buf[bo + 3]];
                let d_u16 = u16::from_le_bytes(d_bytes) as f32 / 65536.0;
                let dmin_u16 = u16::from_le_bytes(dmin_bytes) as f32 / 65536.0;

                // Also try as i16: i16 / 65536.0
                let d_i16 = i16::from_le_bytes(d_bytes) as f32 / 65536.0;
                let dmin_i16 = i16::from_le_bytes(dmin_bytes) as f32 / 65536.0;

                // Try "u16 but with different divisor" - maybe 256?
                let d_div256 = u16::from_le_bytes(d_bytes) as f32 / 256.0;
                let dmin_div256 = u16::from_le_bytes(dmin_bytes) as f32 / 256.0;

                // Scales
                let sc = &buf[bo + 4..bo + 8];
                let qs = &buf[bo + 8..bo + 72];

                // Count zero quants (2-bit values)
                let mut hist = [0u32; 4];
                for j in 0..8 {
                    let shift1 = 2 * (j % 4);
                    let shift2 = 2 * (j / 4);
                    for k in 0..32 {
                        let first = (qs[k] >> shift1) & 3;
                        let second = (qs[32 + k] >> shift2) & 3;
                        hist[first as usize] += 1;
                        hist[second as usize] += 1;
                    }
                }

                // Dequant with different interpretations
                let mut deq_u16 = Vec::new();
                let mut deq_i16 = Vec::new();
                let mut deq_div256 = Vec::new();

                for j in 0..8 {
                    let sc_byte = sc[j / 2];
                    let (sc_l, sc_h) = if j % 2 == 0 {
                        ((sc_byte & 0x0F) as f32, (sc_byte >> 4) as f32)
                    } else {
                        ((sc_byte >> 4) as f32, (sc_byte & 0x0F) as f32)
                    };

                    let shift1 = 2 * (j % 4);
                    let shift2 = 2 * (j / 4);
                    for k in 0..32 {
                        let first = ((qs[k] >> shift1) & 3) as i8;
                        let second = ((qs[32 + k] >> shift2) & 3) as i8;

                        deq_u16.push(
                            d_u16 * sc_l * ((first - 2) as f32)
                                + dmin_u16 * sc_h * ((second - 8) as f32),
                        );
                        deq_i16.push(
                            d_i16 * sc_l * ((first - 2) as f32)
                                + dmin_i16 * sc_h * ((second - 8) as f32),
                        );
                        deq_div256.push(
                            d_div256 * sc_l * ((first - 2) as f32)
                                + dmin_div256 * sc_h * ((second - 8) as f32),
                        );
                    }
                }

                let m_u16 = deq_u16.iter().sum::<f32>() / deq_u16.len() as f32;
                let m_i16 = deq_i16.iter().sum::<f32>() / deq_i16.len() as f32;
                let m_d256 = deq_div256.iter().sum::<f32>() / deq_div256.len() as f32;

                let min_u16 = deq_u16.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_u16 = deq_u16.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min_i16 = deq_i16.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_i16 = deq_i16.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min_d256 = deq_div256.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_d256 = deq_div256.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                println!("Block {}: d={}:{} dmin={}:{} sc=[{:02x} {:02x} {:02x} {:02x}] quants_hist={:?}", 
                    bi,
                    d_u16, format_f32(d_u16), dmin_u16, format_f32(dmin_u16),
                    sc[0], sc[1], sc[2], sc[3], hist);
                println!(
                    "  u16/65536: mean={:.4} range=[{:.4}, {:.4}]",
                    m_u16, min_u16, max_u16
                );
                println!(
                    "  i16/65536: mean={:.4} range=[{:.4}, {:.4}]",
                    m_i16, min_i16, max_i16
                );
                println!(
                    "  /256:      mean={:.4} range=[{:.4}, {:.4}]",
                    m_d256, min_d256, max_d256
                );
                println!();
            }
        }
    }
}

fn format_f32(v: f32) -> String {
    if v.abs() < 1e-6 {
        format!("{:.2e}", v)
    } else if v.abs() < 0.001 {
        format!("{:.6}", v)
    } else if v.abs() < 1.0 {
        format!("{:.4}", v)
    } else {
        format!("{:.2}", v)
    }
}
