//! Per-request Key/Value cache stored in Metal unified memory.
//!
//! Buffers are pre-allocated to `max_len` rows. Appends are bounds-checked:
//! previously `current_len` was advanced unconditionally, so a sequence longer
//! than `max_len` would write past the end of the cache buffers.

use metal::{Buffer, Device, MTLResourceOptions};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KvCacheError {
    #[error("KV cache full: have {current}/{max} rows, cannot append {requested}")]
    Full {
        current: usize,
        max: usize,
        requested: usize,
    },
}

pub struct KVCache {
    pub k_buffer: Buffer,
    pub v_buffer: Buffer,
    pub current_len: usize,
    pub max_len: usize,
    pub head_dim: usize,
}

impl KVCache {
    pub fn new(device: &Device, max_len: usize, head_dim: usize) -> Self {
        let buffer_size = max_len
            .checked_mul(head_dim)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .expect("KV Cache buffer size overflowed");

        let k_buffer = device.new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);
        let v_buffer = device.new_buffer(buffer_size as u64, MTLResourceOptions::StorageModeShared);

        Self {
            k_buffer,
            v_buffer,
            current_len: 0,
            max_len,
            head_dim,
        }
    }

    /// Whether `m` more rows fit without overflowing the pre-allocated buffers.
    pub fn can_fit(&self, m: usize) -> bool {
        self.current_len.saturating_add(m) <= self.max_len
    }

    /// Reserves room for `m` new rows and returns the offset they should be
    /// written at, advancing `current_len`. Errors instead of overflowing.
    pub fn advance(&mut self, m: usize) -> Result<usize, KvCacheError> {
        if !self.can_fit(m) {
            return Err(KvCacheError::Full {
                current: self.current_len,
                max: self.max_len,
                requested: m,
            });
        }
        let offset = self.current_len;
        self.current_len += m;
        Ok(offset)
    }
}

pub struct KVStorage {
    device: Device,
    caches: HashMap<u64, KVCache>,
    max_seq_len: usize,
    head_dim: usize,
}

impl KVStorage {
    pub fn new(device: Device, max_seq_len: usize, head_dim: usize) -> Self {
        Self {
            device,
            caches: HashMap::new(),
            max_seq_len,
            head_dim,
        }
    }

    pub fn get_or_create(&mut self, request_id: u64) -> &mut KVCache {
        let device = &self.device;
        let (max_seq_len, head_dim) = (self.max_seq_len, self.head_dim);
        self.caches
            .entry(request_id)
            .or_insert_with(|| KVCache::new(device, max_seq_len, head_dim))
    }

    pub fn remove(&mut self, request_id: u64) {
        self.caches.remove(&request_id);
    }

    pub fn len(&self) -> usize {
        self.caches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.caches.is_empty()
    }
}
