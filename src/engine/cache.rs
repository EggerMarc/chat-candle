//! KV cache.
//!
//! candle tensors are immutable, so where chat-mlx wrote new tokens into a
//! pre-allocated buffer with `index_mut`, we grow by concatenating along the
//! sequence axis — the candle-idiomatic form, and what candle's own decoders
//! do. The external surface (`offset`, `update_and_fetch`, `truncate`) matches
//! the mlx version so `model.rs`/`generate.rs` are unchanged.
//!
//! TODO: port chat-mlx's rotating attention-sink window (bounded memory). For
//! now `max_size`/`keep` are accepted and retained but the cache grows
//! unbounded; a long-generation memory cap is a follow-up.

use candle_core::{Result, Tensor};

pub struct KvCache {
    keys: Option<Tensor>,
    values: Option<Tensor>,
    offset: usize,
    #[allow(dead_code)]
    max_size: Option<usize>,
    #[allow(dead_code)]
    keep: usize,
}

impl KvCache {
    pub fn new(max_size: Option<i32>, keep: i32) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            max_size: max_size.filter(|m| *m > 0).map(|m| m as usize),
            keep: keep.max(0) as usize,
        }
    }

    /// Number of tokens already cached (the RoPE / mask position offset).
    /// Read *before* `update_and_fetch` folds in the current step.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Drop everything past `len` tokens (used by speculative decoding).
    #[allow(dead_code)]
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        let len = len.min(self.offset);
        if let (Some(k), Some(v)) = (&self.keys, &self.values) {
            self.keys = Some(k.narrow(2, 0, len)?);
            self.values = Some(v.narrow(2, 0, len)?);
        }
        self.offset = len;
        Ok(())
    }

    /// Append `k`/`v` (shape `(batch, kv_heads, seq, head_dim)`) and return the
    /// full cached keys/values to attend over.
    pub fn update_and_fetch(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let keys = match &self.keys {
            Some(prev) => Tensor::cat(&[prev, k], 2)?,
            None => k.clone(),
        };
        let values = match &self.values {
            Some(prev) => Tensor::cat(&[prev, v], 2)?,
            None => v.clone(),
        };
        self.offset = keys.dim(2)?;
        self.keys = Some(keys.clone());
        self.values = Some(values.clone());
        Ok((keys, values))
    }
}
