/// Static pre-allocated KV cache for autoregressive LLM inference.
///
/// Design contract:
/// - Allocated once at runner construction time: [layers × kv_heads × max_seq × head_dim]
/// - `append()` is a slice copy — O(1) per token, no heap allocation
/// - Reads return raw `&[f32]` slices into the pre-allocated buffer
///
/// Uses kv_heads (not n_heads) since GQA means K/V have fewer heads than Q.
pub struct StaticKVCache {
    /// Raw storage: [num_layers, 2 (k/v), kv_heads, max_seq, head_dim]
    storage: Vec<f32>,

    pub num_layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub seq_len: usize,
}

impl StaticKVCache {
    pub fn new(num_layers: usize, kv_heads: usize, head_dim: usize, max_seq: usize) -> Self {
        let total = num_layers * 2 * kv_heads * max_seq * head_dim;
        Self {
            storage: vec![0.0f32; total],
            num_layers,
            kv_heads,
            head_dim,
            max_seq,
            seq_len: 0,
        }
    }

    pub fn size_bytes(&self) -> usize {
        self.storage.len() * 4
    }

    /// Append one token's K and V vectors for a given layer.
    /// `k_vec`: [kv_heads × head_dim]
    /// `v_vec`: [kv_heads × head_dim]
    pub fn append(&mut self, layer: usize, k_vec: &[f32], v_vec: &[f32]) -> bool {
        if self.seq_len >= self.max_seq {
            return false;
        }
        debug_assert_eq!(k_vec.len(), self.kv_heads * self.head_dim);
        debug_assert_eq!(v_vec.len(), self.kv_heads * self.head_dim);

        let slot = self.seq_len;
        let kv_stride = self.kv_heads * self.max_seq * self.head_dim;
        let layer_base = layer * 2 * kv_stride;

        for h in 0..self.kv_heads {
            let k_src = &k_vec[h * self.head_dim..(h + 1) * self.head_dim];
            let k_dst_start = layer_base + h * self.max_seq * self.head_dim + slot * self.head_dim;
            self.storage[k_dst_start..k_dst_start + self.head_dim].copy_from_slice(k_src);
        }

        let v_base = layer_base + kv_stride;
        for h in 0..self.kv_heads {
            let v_src = &v_vec[h * self.head_dim..(h + 1) * self.head_dim];
            let v_dst_start = v_base + h * self.max_seq * self.head_dim + slot * self.head_dim;
            self.storage[v_dst_start..v_dst_start + self.head_dim].copy_from_slice(v_src);
        }

        true
    }

    pub fn advance(&mut self) {
        if self.seq_len < self.max_seq {
            self.seq_len += 1;
        }
    }

    /// Return base offset and stride for direct-indexing into storage.
    pub fn layer_stride(&self) -> usize {
        2 * self.kv_heads * self.max_seq * self.head_dim
    }

    pub fn head_stride(&self) -> usize {
        self.max_seq * self.head_dim
    }

    /// Index into K storage for a given layer and KV head at position `seq_pos`.
    pub fn k_index(&self, layer: usize, kv_head: usize, seq_pos: usize) -> usize {
        layer * self.layer_stride() + kv_head * self.head_stride() + seq_pos * self.head_dim
    }

    /// Index into V storage for a given layer and KV head at position `seq_pos`.
    pub fn v_index(&self, layer: usize, kv_head: usize, seq_pos: usize) -> usize {
        layer * self.layer_stride()
            + self.kv_heads * self.head_stride()
            + kv_head * self.head_stride()
            + seq_pos * self.head_dim
    }

    /// Raw storage reference for direct indexing.
    pub fn storage(&self) -> &[f32] {
        &self.storage
    }

    /// Mutable raw storage for batch write operations.
    pub fn storage_mut(&mut self) -> &mut [f32] {
        &mut self.storage
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}
