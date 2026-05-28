use std::fs::File;
use std::io::{Read, Seek};

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

    #[derive(Clone)]
    struct TInfo {
        name: String,
        dtype: u32,
        offset: u64,
        shape: Vec<usize>,
    }
    let mut infos = Vec::new();
    for _ in 0..tensor_count {
        let name = read_string(&mut f);
        let n_dims = read_u32(&mut f) as usize;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(read_u64(&mut f) as usize);
        }
        let dtype = read_u32(&mut f);
        let offset = read_u64(&mut f);
        infos.push(TInfo {
            name,
            dtype,
            offset,
            shape,
        });
    }

    let info_end = f.stream_position().unwrap();
    let file_len = f.metadata().unwrap().len();
    println!("File size: {} bytes", file_len);
    println!("DIAGNOSTIC - info_end = {}", info_end);

    // Let's print all tensor names and offsets to see what's in the file
    let mut sorted_infos = infos.clone();
    sorted_infos.sort_by_key(|info| info.offset);

    println!("Total tensors: {}", sorted_infos.len());
    println!("First 10 tensors by offset:");
    for i in 0..10.min(sorted_infos.len()) {
        println!(
            "  {}: offset={}, dtype={}, shape={:?}",
            sorted_infos[i].name,
            sorted_infos[i].offset,
            sorted_infos[i].dtype,
            sorted_infos[i].shape
        );
    }

    println!("Last 10 tensors by offset:");
    let len = sorted_infos.len();
    for i in (len - 10.min(len)..len).rev() {
        println!(
            "  {}: offset={}, dtype={}, shape={:?}",
            sorted_infos[i].name,
            sorted_infos[i].offset,
            sorted_infos[i].dtype,
            sorted_infos[i].shape
        );
    }

    // Specific tensors
    let targets = [
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_q_a.weight",
        "output_norm.weight",
        "model.norm.weight",
        "output.weight",
    ];
    for t in targets {
        if let Some(info) = infos.iter().find(|i| i.name == t || i.name.contains(t)) {
            println!(
                "Target tensor '{}' (actual name '{}'): offset={}",
                t, info.name, info.offset
            );
        }
    }
}
