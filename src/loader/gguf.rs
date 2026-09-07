#![allow(unsafe_code)]
use crate::Error;
/// GGUF v3 format loader.
///
/// Spec: https://github.com/philpax/ggml/blob/master/docs/gguf.md
/// GGUF files pack quantized model weights with metadata headers.
use std::collections::HashMap;
use std::io::{Read, Seek};
use std::ops::Deref;
use std::sync::Arc;
use tracing::warn;

/// Compute SHA-256 hex digest of a file.
/// Can be called before `GGUFModel::load` to verify model integrity.
pub fn sha256_hex(path: &str) -> Result<String, Error> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)
        .map_err(|e| Error::ExecutionError(format!("Failed to open file for checksum: {e}")))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| Error::ExecutionError(format!("Failed to read file for checksum: {e}")))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian

/// Upper bounds for untrusted GGUF headers.
///
/// A corrupt or malicious file can claim petabyte string lengths, billions
/// of metadata entries, or wrapping tensor offsets. Without caps the parser
/// would abort on giant allocations (`vec![0u8; len]`), burn CPU looping to
/// EOF, or — in release builds — wrap `offset + size` past the bounds check.
/// Real-world models stay far below these limits (a few thousand tensors,
/// KiB-scale metadata strings, ≤256k-entry tokenizer arrays).
const MAX_STRING_LEN: usize = 16 * 1024 * 1024; // 16 MiB
const MAX_METADATA_KV: u64 = 100_000;
const MAX_TENSOR_COUNT: u64 = 100_000;
const MAX_N_DIMS: usize = 32;
const MAX_ARRAY_LEN: u64 = 32_000_000;
const MAX_ALIGNMENT: u64 = 1 << 20; // 1 MiB
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GGUFValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GGUFValueType {
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub enum GGUFValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum GGUFDtype {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
}

impl GGUFDtype {
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::I8,
            17 => Self::I16,
            18 => Self::I32,
            _ => return None,
        })
    }

    /// Block size in number of weights for quantized types.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::I8 | Self::I16 | Self::I32 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K => 256,
            Self::Q3_K => 256,
            Self::Q4_K => 256, // QK_K=256: 8 sub-blocks of 32 values
            Self::Q5_K => 256,
            Self::Q6_K => 256,
            Self::Q8_K => 256,
        }
    }

    /// Size of one block in bytes for quantized types.
    pub fn block_byte_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::Q4_0 => 18,  // 2 (d) + 16 (quants)
            Self::Q4_1 => 20,  // 2 (d) + 2 (m) + 16 (quants)
            Self::Q5_0 => 22,  // 2 (d) + 4 (qh) + 16 (ql)
            Self::Q5_1 => 24,  // 2 (d) + 2 (m) + 4 (qh) + 16 (ql)
            Self::Q8_0 => 34,  // 2 (d) + 32 (quants as int8)
            Self::Q8_1 => 34,  // 2 (d) + 32 (quants as int8)
            Self::Q2_K => 84,  // d(2)+dmin(2)+scales(16)+qs(64)
            Self::Q3_K => 110, // hmask(32)+qs(64)+scales(12)+d(2)
            Self::Q4_K => 144, // QK_K=256: d(2)+dmin(2)+sc(12)+qs(128)=144
            Self::Q5_K => 176,
            Self::Q6_K => 210, // QK_K=256: d(2)+ql(128)+qh(64)+sc(16)=210
            Self::Q8_K => 136,
        }
    }
}

#[derive(Clone, Debug)]
pub enum SharedBytes {
    Mmap {
        mmap: Arc<memmap2::Mmap>,
        offset: usize,
        len: usize,
    },
    Owned {
        vec: Arc<Vec<u8>>,
        offset: usize,
        len: usize,
    },
}

impl SharedBytes {
    pub fn new_mmap(mmap: Arc<memmap2::Mmap>, offset: usize, len: usize) -> Self {
        SharedBytes::Mmap { mmap, offset, len }
    }

    pub fn new_owned(vec: Vec<u8>) -> Self {
        let len = vec.len();
        SharedBytes::Owned {
            vec: Arc::new(vec),
            offset: 0,
            len,
        }
    }

    /// Return the offset of this slice within the underlying storage.
    pub fn offset(&self) -> usize {
        match self {
            SharedBytes::Mmap { offset, .. } | SharedBytes::Owned { offset, .. } => *offset,
        }
    }

    /// Return the byte length of this slice.
    pub fn len(&self) -> usize {
        match self {
            SharedBytes::Mmap { len, .. } | SharedBytes::Owned { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// If this is backed by an mmap, return the shared Arc reference.
    pub fn mmap_arc(&self) -> Option<Arc<memmap2::Mmap>> {
        match self {
            SharedBytes::Mmap { mmap, .. } => Some(mmap.clone()),
            SharedBytes::Owned { .. } => None,
        }
    }

    /// If this is backed by an mmap, return the backing (start, end) byte range.
    pub fn mmap_range(&self) -> Option<(usize, usize)> {
        match self {
            SharedBytes::Mmap { offset, len, .. } => Some((*offset, *offset + *len)),
            SharedBytes::Owned { .. } => None,
        }
    }

    pub fn slice(&self, offset: usize, len: usize) -> Self {
        match self {
            SharedBytes::Mmap {
                mmap,
                offset: base_offset,
                len: base_len,
            } => {
                assert!(offset + len <= *base_len);
                SharedBytes::Mmap {
                    mmap: mmap.clone(),
                    offset: base_offset + offset,
                    len,
                }
            }
            SharedBytes::Owned {
                vec,
                offset: base_offset,
                len: base_len,
            } => {
                assert!(offset + len <= *base_len);
                SharedBytes::Owned {
                    vec: vec.clone(),
                    offset: base_offset + offset,
                    len,
                }
            }
        }
    }
}

impl Deref for SharedBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            SharedBytes::Mmap { mmap, offset, len } => &mmap[*offset..*offset + *len],
            SharedBytes::Owned { vec, offset, len } => &vec[*offset..*offset + *len],
        }
    }
}

#[derive(Debug, Clone)]
pub struct GGUFTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GGUFDtype,
    pub data: SharedBytes,
}

#[derive(Debug, Clone)]
pub struct GGUFModel {
    pub metadata: HashMap<String, GGUFValue>,
    pub tensors: HashMap<String, GGUFTensor>,
}

pub struct GGUFLoader;

impl GGUFLoader {
    /// Load a GGUF file from disk.
    pub fn load(path: &str) -> Result<GGUFModel, Error> {
        let file = std::fs::File::open(path)
            .map_err(|e| Error::ExecutionError(format!("Failed to open GGUF file: {}", e)))?;
        // SAFETY: `memmap2::Mmap::map` only requires a valid open file
        // descriptor at map time. The returned mapping keeps its own
        // OS-level reference (fd duplication on Unix, mapping handle on
        // Windows), so it stays valid after `file` is closed at the end of
        // this function. Callers must still avoid externally truncating or
        // mutating the file while mapped (that raises SIGBUS on access).
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| Error::ExecutionError(format!("Failed to mmap GGUF file: {}", e)))?;
        let mmap = Arc::new(mmap);

        // Wrap the mmap slice in a Cursor so we can use existing seek/read logic
        let mut cursor = std::io::Cursor::new(&mmap[..]);

        // Read header
        let magic = read_u32(&mut cursor)?;
        if magic != GGUF_MAGIC {
            return Err(Error::ExecutionError(format!(
                "Invalid GGUF magic: {:#x}",
                magic
            )));
        }

        let version = read_u32(&mut cursor)?;
        if version != 3 {
            warn!(
                "GGUF version {} detected — only v3 is fully supported. Parsing may be incorrect.",
                version
            );
        }
        let tensor_count = read_u64(&mut cursor)?;
        let metadata_kv_count = read_u64(&mut cursor)?;

        // Reject absurd header counts before looping: a corrupt file claiming
        // u64::MAX entries would otherwise burn CPU until EOF.
        if metadata_kv_count > MAX_METADATA_KV {
            return Err(Error::ExecutionError(format!(
                "GGUF metadata count {} exceeds limit {}",
                metadata_kv_count, MAX_METADATA_KV
            )));
        }
        if tensor_count > MAX_TENSOR_COUNT {
            return Err(Error::ExecutionError(format!(
                "GGUF tensor count {} exceeds limit {}",
                tensor_count, MAX_TENSOR_COUNT
            )));
        }

        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let (key, value) = read_metadata_kv(&mut cursor)?;
            metadata.insert(key, value);
        }

        // Read tensor info
        let mut tensor_infos = Vec::new();
        for _ in 0..tensor_count {
            let info = read_tensor_info(&mut cursor)?;
            tensor_infos.push(info);
        }

        // Only unsigned values are valid; negative Int* values wrap to huge
        // u64 with `as` casts, so validate sign explicitly. Zero would cause
        // a `% 0` panic below, so fall back to the spec default instead.
        let alignment = match metadata.get("general.alignment") {
            Some(GGUFValue::Uint32(v)) => *v as u64,
            Some(GGUFValue::Uint64(v)) => *v,
            Some(GGUFValue::Uint16(v)) => *v as u64,
            Some(GGUFValue::Uint8(v)) => *v as u64,
            Some(GGUFValue::Int32(v)) if *v > 0 => *v as u64,
            Some(GGUFValue::Int64(v)) if *v > 0 => *v as u64,
            Some(GGUFValue::Int16(v)) if *v > 0 => *v as u64,
            Some(GGUFValue::Int8(v)) if *v > 0 => *v as u64,
            _ => DEFAULT_ALIGNMENT,
        };
        let alignment = if alignment == 0 {
            DEFAULT_ALIGNMENT
        } else {
            alignment
        };
        if alignment > MAX_ALIGNMENT {
            return Err(Error::ExecutionError(format!(
                "GGUF alignment {} exceeds limit {}",
                alignment, MAX_ALIGNMENT
            )));
        }

        // Alignment padding (tensor data starts at page-aligned offset)
        let pos = cursor
            .stream_position()
            .map_err(|e| Error::ExecutionError(format!("Failed to get position: {}", e)))?;
        let remainder = pos % alignment;
        let padding = if remainder == 0 {
            0
        } else {
            alignment - remainder
        };
        let tensor_data_start_pos = pos
            .checked_add(padding)
            .ok_or_else(|| Error::ExecutionError("GGUF data offset overflow".into()))?;

        // Slice tensor data zero-copy
        let mut tensors = HashMap::new();
        for info in tensor_infos {
            let byte_size = compute_tensor_byte_size(&info.dtype, &info.shape)?;
            // checked_add: in release builds plain `+` wraps, which would
            // turn a huge offset into a small one and bypass the bounds check.
            let offset = tensor_data_start_pos
                .checked_add(info.offset)
                .ok_or_else(|| {
                    Error::ExecutionError(format!("Tensor '{}' offset overflow", info.name))
                })?;
            let end = offset.checked_add(byte_size as u64).ok_or_else(|| {
                Error::ExecutionError(format!("Tensor '{}' end offset overflow", info.name))
            })?;

            // Validate that we don't read out of bounds
            if end > mmap.len() as u64 {
                return Err(Error::ExecutionError(format!(
                    "Tensor '{}' offset out of bounds",
                    info.name
                )));
            }

            let data = SharedBytes::new_mmap(mmap.clone(), offset as usize, byte_size);

            tensors.insert(
                info.name.clone(),
                GGUFTensor {
                    name: info.name,
                    shape: info.shape,
                    dtype: info.dtype,
                    data,
                },
            );
        }

        Ok(GGUFModel { metadata, tensors })
    }
}

struct TensorInfo {
    name: String,
    shape: Vec<usize>,
    dtype: GGUFDtype,
    offset: u64,
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, Error> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| Error::ExecutionError(format!("Read error: {}", e)))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, Error> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)
        .map_err(|e| Error::ExecutionError(format!("Read error: {}", e)))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64, Error> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)
        .map_err(|e| Error::ExecutionError(format!("Read error: {}", e)))?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32, Error> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| Error::ExecutionError(format!("Read error: {}", e)))?;
    Ok(f32::from_le_bytes(buf))
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8, Error> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)
        .map_err(|e| Error::ExecutionError(format!("Read error: {}", e)))?;
    Ok(u8::from_le_bytes(buf))
}

fn read_i8<R: Read>(r: &mut R) -> Result<i8, Error> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)
        .map_err(|e| Error::ExecutionError(format!("Read error: {}", e)))?;
    Ok(i8::from_le_bytes(buf))
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16, Error> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)
        .map_err(|e| Error::ExecutionError(format!("Read error: {}", e)))?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16<R: Read>(r: &mut R) -> Result<i16, Error> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)
        .map_err(|e| Error::ExecutionError(format!("Read error: {}", e)))?;
    Ok(i16::from_le_bytes(buf))
}

fn read_f64<R: Read>(r: &mut R) -> Result<f64, Error> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)
        .map_err(|e| Error::ExecutionError(format!("Read error: {}", e)))?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string<R: Read>(r: &mut R) -> Result<String, Error> {
    let len = read_u64(r)? as usize;
    // A corrupt file can claim a petabyte string: `vec![0u8; len]` would
    // abort the process on allocation failure. Cap before allocating.
    if len > MAX_STRING_LEN {
        return Err(Error::ExecutionError(format!(
            "GGUF string length {} exceeds limit {}",
            len, MAX_STRING_LEN
        )));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)
        .map_err(|e| Error::ExecutionError(format!("Read string error: {}", e)))?;
    String::from_utf8(buf).map_err(|e| Error::ExecutionError(format!("Invalid UTF-8: {}", e)))
}

fn read_metadata_kv<R: Read + Seek>(r: &mut R) -> Result<(String, GGUFValue), Error> {
    let key = read_string(r)?;
    let value_type = read_u32(r)?;
    let vt = GGUFValueType::from_u32(value_type).ok_or_else(|| {
        Error::ExecutionError(format!("Unknown metadata value type: {}", value_type))
    })?;

    let value = read_value(r, vt)?;

    Ok((key, value))
}

fn read_value<R: Read>(r: &mut R, vt: GGUFValueType) -> Result<GGUFValue, Error> {
    match vt {
        GGUFValueType::Uint8 => Ok(GGUFValue::Uint8(read_u8(r)?)),
        GGUFValueType::Int8 => Ok(GGUFValue::Int8(read_i8(r)?)),
        GGUFValueType::Uint16 => Ok(GGUFValue::Uint16(read_u16(r)?)),
        GGUFValueType::Int16 => Ok(GGUFValue::Int16(read_i16(r)?)),
        GGUFValueType::Uint32 => Ok(GGUFValue::Uint32(read_u32(r)?)),
        GGUFValueType::Int32 => Ok(GGUFValue::Int32(read_u32(r)? as i32)),
        GGUFValueType::Float32 => Ok(GGUFValue::Float32(read_f32(r)?)),
        GGUFValueType::Bool => Ok(GGUFValue::Bool(read_u8(r)? != 0)),
        GGUFValueType::String => Ok(GGUFValue::String(read_string(r)?)),
        GGUFValueType::Array => {
            let array_type = read_u32(r)?;
            let array_len = read_u64(r)?;
            if array_len > MAX_ARRAY_LEN {
                return Err(Error::ExecutionError(format!(
                    "GGUF array length {} exceeds limit {}",
                    array_len, MAX_ARRAY_LEN
                )));
            }
            let elem_vt = GGUFValueType::from_u32(array_type).ok_or_else(|| {
                Error::ExecutionError(format!("Unknown array element type: {}", array_type))
            })?;
            // Reserve incrementally: `with_capacity(array_len)` on a
            // hostile length would over-commit gigabytes of RSS.
            let mut elems = Vec::with_capacity((array_len.min(4096)) as usize);
            for _ in 0..array_len {
                elems.push(read_value(r, elem_vt)?);
            }
            Ok(GGUFValue::Array(elems))
        }
        GGUFValueType::Uint64 => Ok(GGUFValue::Uint64(read_u64(r)?)),
        GGUFValueType::Int64 => Ok(GGUFValue::Int64(read_i64(r)?)),
        GGUFValueType::Float64 => Ok(GGUFValue::Float64(read_f64(r)?)),
    }
}

fn read_tensor_info<R: Read + Seek>(r: &mut R) -> Result<TensorInfo, Error> {
    let name = read_string(r)?;
    let n_dims = read_u32(r)? as usize;
    if n_dims > MAX_N_DIMS {
        return Err(Error::ExecutionError(format!(
            "GGUF tensor '{}' has {} dims, limit is {}",
            name, n_dims, MAX_N_DIMS
        )));
    }
    let mut shape = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        shape.push(read_u64(r)? as usize);
    }
    let dtype_val = read_u32(r)?;
    let dtype = GGUFDtype::from_u32(dtype_val)
        .ok_or_else(|| Error::ExecutionError(format!("Unknown tensor dtype: {}", dtype_val)))?;
    let offset = read_u64(r)?;

    Ok(TensorInfo {
        name,
        shape,
        dtype,
        offset,
    })
}

fn compute_tensor_byte_size(dtype: &GGUFDtype, shape: &[usize]) -> Result<usize, Error> {
    // `product()` and `*` wrap in release builds; a hostile shape could
    // otherwise shrink the size and pass the bounds check with a short slice.
    let mut num_elements: usize = 1;
    for &dim in shape {
        num_elements = num_elements
            .checked_mul(dim)
            .ok_or_else(|| Error::ExecutionError("GGUF tensor shape overflows usize".into()))?;
    }
    if *dtype == GGUFDtype::F32 {
        return num_elements
            .checked_mul(4)
            .ok_or_else(|| Error::ExecutionError("GGUF tensor byte size overflows usize".into()));
    }
    if *dtype == GGUFDtype::F16 {
        return num_elements
            .checked_mul(2)
            .ok_or_else(|| Error::ExecutionError("GGUF tensor byte size overflows usize".into()));
    }
    let block_sz = dtype.block_size();
    let block_byte = dtype.block_byte_size();
    let num_blocks = num_elements.div_ceil(block_sz);
    num_blocks
        .checked_mul(block_byte)
        .ok_or_else(|| Error::ExecutionError("GGUF tensor byte size overflows usize".into()))
}
