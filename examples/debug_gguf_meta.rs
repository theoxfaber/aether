use std::fs::File;
use std::io::Read;

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
fn read_f32<R: Read>(r: &mut R) -> f32 {
    f32::from_le_bytes({
        let mut b = [0u8; 4];
        r.read_exact(&mut b).unwrap();
        b
    })
}

fn main() {
    let mut f = File::open("tinyllama-q4.gguf").unwrap();
    let magic = read_u32(&mut f);
    let version = read_u32(&mut f);
    let _tensor_count = read_u64(&mut f);
    let metadata_kv_count = read_u64(&mut f);

    println!("GGUF magic=0x{:08x} version={}", magic, version);

    for _ in 0..metadata_kv_count {
        let key = read_string(&mut f);
        let val_type = read_u32(&mut f);
        match val_type {
            0 => {
                let v = read_u32(&mut f);
                println!("  {}: bool({})", key, v);
            }
            1 => {
                let v = read_f32(&mut f);
                println!("  {}: f32({})", key, v);
            }
            2 => {
                let v = read_u32(&mut f);
                println!("  {}: u8({})", key, v);
            }
            3 => {
                let v = read_u32(&mut f) as i32;
                println!("  {}: i8({})", key, v);
            }
            4 => {
                let v = read_u32(&mut f);
                println!("  {}: u32({})", key, v);
            }
            5 => {
                let v = read_u32(&mut f) as i32;
                println!("  {}: i32({})", key, v);
            }
            6 => {
                let v = read_f32(&mut f);
                println!("  {}: f32({})", key, v);
            }
            7 => {
                let v = read_u32(&mut f);
                println!("  {}: u32({})", key, v);
            }
            8 => {
                let s = read_string(&mut f);
                println!("  {}: '{}'", key, s);
            }
            9 => {
                let elem_type = read_u32(&mut f);
                let alen = read_u64(&mut f);
                println!("  {}: array[{}]", key, alen);
                // Read but discard array elements
                for _ in 0..(alen.min(50) as usize) {
                    match elem_type {
                        0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 => {
                            let mut b = [0u8; 4];
                            f.read_exact(&mut b).unwrap();
                        }
                        8 => {
                            let _s = read_string(&mut f);
                        }
                        10 | 11 | 12 => {
                            let mut b = [0u8; 8];
                            f.read_exact(&mut b).unwrap();
                        }
                        _ => {}
                    }
                }
            }
            10 => {
                let v = read_u64(&mut f);
                println!("  {}: u64({})", key, v);
            }
            11 => {
                let v = read_u64(&mut f) as i64;
                println!("  {}: i64({})", key, v);
            }
            12 => {
                let v = f64::from_le_bytes({
                    let mut b = [0u8; 8];
                    f.read_exact(&mut b).unwrap();
                    b
                });
                println!("  {}: f64({})", key, v);
            }
            _ => panic!("Unknown type {} for key {}", val_type, key),
        };
    }
}
