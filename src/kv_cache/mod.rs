use metal::{Buffer, Device, MTLResourceOptions};
use std::collections::HashMap;

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
        self.caches.entry(request_id).or_insert_with(|| {
            KVCache::new(&self.device, self.max_seq_len, self.head_dim)
        })
    }

    pub fn remove(&mut self, request_id: u64) {
        self.caches.remove(&request_id);
    }
}
