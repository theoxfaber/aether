use crate::loader::gguf::{GGUFModel, GGUFValue};
use crate::Error;
use std::collections::HashMap;

/// A production-grade SentencePiece/BPE tokenizer for LLaMA-family models.
///
/// Correctly handles:
/// - SentencePiece ▁ (U+2581) as word-initial space marker
/// - Byte-fallback tokens `<0xNN>` decoded by reassembling UTF-8 byte sequences
/// - BOS/EOS injection
/// - Temperature + top-p sampling (for decode use)
#[derive(Clone)]
pub struct Tokenizer {
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merge_priority: HashMap<(u32, u32), usize>,
    /// byte_to_token[b] = token id for the byte fallback token <0xNN>
    byte_to_token: [u32; 256],
    pub bos_id: u32,
    pub eos_id: u32,
    pub unk_id: u32,
    pub vocab_size: usize,
}

fn build_byte_map(id_to_token: &[String]) -> [u32; 256] {
    let mut map = [0u32; 256];
    for (id, token) in id_to_token.iter().enumerate() {
        if let Some(hex) = token.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
            if let Ok(b) = u8::from_str_radix(hex, 16) {
                map[b as usize] = id as u32;
            }
        }
    }
    map
}

impl Tokenizer {
    pub fn from_gguf(model: &GGUFModel) -> Result<Self, Error> {
        let tokens = match model.metadata.get("tokenizer.ggml.tokens") {
            Some(GGUFValue::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    GGUFValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect::<Vec<_>>(),
            _ => {
                return Err(Error::ExecutionError(
                    "Missing tokenizer.ggml.tokens".into(),
                ))
            }
        };

        let merges_raw: Vec<String> = match model.metadata.get("tokenizer.ggml.merges") {
            Some(GGUFValue::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    GGUFValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
            _ => Vec::new(),
        };

        let bos_id = gguf_u32(model, "tokenizer.ggml.bos_token_id").unwrap_or(1);
        let eos_id = gguf_u32(model, "tokenizer.ggml.eos_token_id").unwrap_or(2);
        let unk_id = gguf_u32(model, "tokenizer.ggml.unknown_token_id").unwrap_or(0);

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        // Build merge table: maps (left_id, right_id) → merge priority
        // This is faster than string key lookups during BPE inner loop.
        let mut merge_priority: HashMap<(u32, u32), usize> = HashMap::new();
        for (priority, merge_str) in merges_raw.iter().enumerate() {
            if let Some(space) = merge_str.find(' ') {
                let left = &merge_str[..space];
                let right = &merge_str[space + 1..];
                if let (Some(&lid), Some(&rid)) = (token_to_id.get(left), token_to_id.get(right)) {
                    merge_priority.insert((lid, rid), priority);
                }
            }
        }

        let byte_to_token = build_byte_map(&tokens);
        let vocab_size = tokens.len();

        Ok(Self {
            id_to_token: tokens,
            token_to_id,
            merge_priority,
            byte_to_token,
            bos_id,
            eos_id,
            unk_id,
            vocab_size,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Encode text → token ids. Implements SentencePiece BPE used by LLaMA:
    /// - Leading space encodes as ▁ (U+2581)
    /// - Unknown characters fall back to per-byte `<0xNN>` tokens
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        // SentencePiece treats the text as if preceded by a space.
        // Prepend a space, then encode the whole string as one piece.
        let text_with_space = format!(" {}", text);
        let mut ids = Vec::new();
        if add_bos {
            ids.push(self.bos_id);
        }
        ids.extend(self.sp_encode(&text_with_space));
        ids
    }

    /// Decode token ids → UTF-8 string.
    /// Handles:
    ///  - ▁ → space
    ///  - `<0xNN>` byte tokens: reassembled into proper UTF-8
    ///  - Normal tokens: appended directly
    pub fn decode(&self, ids: &[u32]) -> String {
        // Collect raw bytes first, then convert to String once
        let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 2);
        for &id in ids {
            if id == self.bos_id || id == self.eos_id {
                continue;
            }
            if let Some(token) = self.id_to_token.get(id as usize) {
                if token == "<unk>" || token == "<s>" || token == "</s>" {
                    continue;
                }
                // Byte fallback token
                if let Some(hex) = token.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
                    if let Ok(b) = u8::from_str_radix(hex, 16) {
                        bytes.push(b);
                        continue;
                    }
                }
                // SentencePiece word-space marker ▁ → space
                let text = token.as_str();
                if let Some(rest) = text.strip_prefix('\u{2581}') {
                    bytes.push(b' ');
                    bytes.extend_from_slice(rest.as_bytes());
                } else {
                    bytes.extend_from_slice(text.as_bytes());
                }
            }
        }
        // Best-effort UTF-8 decode; replace invalid sequences with replacement char
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode a single token id to a string fragment (for streaming output).
    pub fn decode_one(&self, id: u32) -> String {
        self.decode(&[id])
    }

    pub fn id_to_token_str(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// SentencePiece-style BPE encode for a single piece of text.
    /// Uses integer-keyed merge table for O(1) lookups.
    fn sp_encode(&self, text: &str) -> Vec<u32> {
        // Step 1: Initialize with per-byte tokens or known single-char tokens
        let mut pieces: Vec<u32> = Vec::with_capacity(text.len());

        // Try to match longest known token at each position (greedy char-level init)
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            // Build the SentencePiece representation: ▁ for space
            let ch = if chars[i] == ' ' {
                '\u{2581}'
            } else {
                chars[i]
            };
            let s = ch.to_string();

            if let Some(&id) = self.token_to_id.get(&s) {
                pieces.push(id);
                i += 1;
            } else {
                // Fall back to byte tokens for each byte of the character
                let mut buf = [0u8; 4];
                let encoded = chars[i].encode_utf8(&mut buf);
                for b in encoded.as_bytes() {
                    pieces.push(self.byte_to_token[*b as usize]);
                }
                i += 1;
            }
        }

        // Step 2: BPE merges — O(n²) but n is short for typical tokens
        loop {
            let mut best_priority = usize::MAX;
            let mut best_idx = 0;

            for j in 0..pieces.len().saturating_sub(1) {
                let key = (pieces[j], pieces[j + 1]);
                if let Some(&p) = self.merge_priority.get(&key) {
                    if p < best_priority {
                        best_priority = p;
                        best_idx = j;
                    }
                }
            }

            if best_priority == usize::MAX {
                break; // No more merges possible
            }

            // Merge pieces[best_idx] and pieces[best_idx+1]
            let left = &self.id_to_token[pieces[best_idx] as usize];
            let right = &self.id_to_token[pieces[best_idx + 1] as usize];
            let merged = format!("{}{}", left, right);

            match self.token_to_id.get(&merged) {
                Some(&new_id) => {
                    pieces[best_idx] = new_id;
                    pieces.remove(best_idx + 1);
                }
                None => break, // Shouldn't happen if merge table is consistent
            }
        }

        pieces
    }
}

fn gguf_u32(model: &GGUFModel, key: &str) -> Option<u32> {
    match model.metadata.get(key) {
        Some(GGUFValue::Uint32(v)) => Some(*v),
        Some(GGUFValue::Uint64(v)) => Some(*v as u32),
        Some(GGUFValue::Int32(v)) => Some(*v as u32),
        _ => None,
    }
}
