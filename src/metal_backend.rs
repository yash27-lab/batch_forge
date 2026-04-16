use metal::{Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize};
use std::error::Error;
use tracing::{info};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BackendError {
    #[error("Buffer size computation overflowed")]
    BufferOverflow,
    #[error("Metal buffer allocation failed")]
    AllocationFailed,
    #[error("Compute pipeline dispatch failed")]
    DispatchFailed,
    #[error("Initialization error: {0}")]
    Init(String),
}

pub struct MetalBackend {
    pub device: Device,
    command_queue: CommandQueue,
    library: Library,
    matmul_pipeline: ComputePipelineState,
    quant_matmul_pipeline: ComputePipelineState,
    kv_attention_pipeline: ComputePipelineState,
    update_kv_cache_pipeline: ComputePipelineState,
}

impl MetalBackend {
    pub fn new(shader_source: &str) -> Result<Self, BackendError> {
        let device = Device::system_default().ok_or_else(|| BackendError::Init("No Metal device found. Are you on a Mac?".to_string()))?;
        info!("Initialized Metal device: {}", device.name());

        let command_queue = device.new_command_queue();
        
        let options = CompileOptions::new();
        let library = device.new_library_with_source(shader_source, &options)
            .map_err(|e| BackendError::Init(format!("Failed to compile shader: {}", e)))?;

        let matmul_func = library.get_function("matmul", None)
            .map_err(|e| BackendError::Init(format!("Failed to find function 'matmul': {}", e)))?;
        let matmul_pipeline = device.new_compute_pipeline_state_with_function(&matmul_func)
            .map_err(|e| BackendError::Init(format!("Failed to create compute pipeline: {}", e)))?;

        let quant_matmul_func = library.get_function("quant_matmul", None)
            .map_err(|e| BackendError::Init(format!("Failed to find function 'quant_matmul': {}", e)))?;
        let quant_matmul_pipeline = device.new_compute_pipeline_state_with_function(&quant_matmul_func)
            .map_err(|e| BackendError::Init(format!("Failed to create compute pipeline: {}", e)))?;

        let kv_attention_func = library.get_function("kv_attention", None)
            .map_err(|e| BackendError::Init(format!("Failed to find function 'kv_attention': {}", e)))?;
        let kv_attention_pipeline = device.new_compute_pipeline_state_with_function(&kv_attention_func)
            .map_err(|e| BackendError::Init(format!("Failed to create compute pipeline: {}", e)))?;

        let update_kv_cache_func = library.get_function("update_kv_cache", None)
            .map_err(|e| BackendError::Init(format!("Failed to find function 'update_kv_cache': {}", e)))?;
        let update_kv_cache_pipeline = device.new_compute_pipeline_state_with_function(&update_kv_cache_func)
            .map_err(|e| BackendError::Init(format!("Failed to create compute pipeline: {}", e)))?;

        Ok(Self {
            device,
            command_queue,
            library,
            matmul_pipeline,
            quant_matmul_pipeline,
            kv_attention_pipeline,
            update_kv_cache_pipeline,
        })
    }

    pub fn create_buffer<T>(&self, data: &[T]) -> Result<Buffer, BackendError> {
        let length = data.len().checked_mul(std::mem::size_of::<T>()).ok_or(BackendError::BufferOverflow)?;
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            length as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }

    pub fn create_buffer_uninitialized<T>(&self, len: usize) -> Result<Buffer, BackendError> {
        let length = len.checked_mul(std::mem::size_of::<T>()).ok_or(BackendError::BufferOverflow)?;
        let buffer = self.device.new_buffer(
            length as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }

    pub fn matmul(&self, a: &Buffer, b: &Buffer, c: &Buffer, m: u32, n: u32, k: u32) -> Result<(), BackendError> {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.matmul_pipeline);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(c), 0);
        
        let m_buf = self.create_buffer(&[m])?;
        let n_buf = self.create_buffer(&[n])?;
        let k_buf = self.create_buffer(&[k])?;
        
        encoder.set_buffer(3, Some(&m_buf), 0);
        encoder.set_buffer(4, Some(&n_buf), 0);
        encoder.set_buffer(5, Some(&k_buf), 0);

        let w = self.matmul_pipeline.thread_execution_width();
        let h = self.matmul_pipeline.max_total_threads_per_threadgroup() / w;
        
        let threads_per_threadgroup = MTLSize::new(w, h, 1);
        let threadgroups_per_grid = MTLSize::new(
            (n as u64 + w - 1) / w,
            (m as u64 + h - 1) / h,
            1,
        );

        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }

    pub fn kv_attention(&self, q: &Buffer, k_cache: &Buffer, v_cache: &Buffer, o: &Buffer, m: u32, cur_seq_len: u32, d: u32) -> Result<(), BackendError> {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.kv_attention_pipeline);
        encoder.set_buffer(0, Some(q), 0);
        encoder.set_buffer(1, Some(k_cache), 0);
        encoder.set_buffer(2, Some(v_cache), 0);
        encoder.set_buffer(3, Some(o), 0);
        
        let m_buf = self.create_buffer(&[m])?;
        let cur_seq_len_buf = self.create_buffer(&[cur_seq_len])?;
        let d_buf = self.create_buffer(&[d])?;
        
        encoder.set_buffer(4, Some(&m_buf), 0);
        encoder.set_buffer(5, Some(&cur_seq_len_buf), 0);
        encoder.set_buffer(6, Some(&d_buf), 0);

        let max_threads = self.kv_attention_pipeline.max_total_threads_per_threadgroup();
        let threads_per_threadgroup = MTLSize::new(std::cmp::min(m as u64, max_threads), 1, 1);
        let threadgroups_per_grid = MTLSize::new(
            (m as u64 + threads_per_threadgroup.width - 1) / threads_per_threadgroup.width,
            1,
            1,
        );

        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }

    pub fn update_kv_cache(&self, new_k: &Buffer, new_v: &Buffer, k_cache: &Buffer, v_cache: &Buffer, m: u32, offset: u32, d: u32) -> Result<(), BackendError> {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.update_kv_cache_pipeline);
        encoder.set_buffer(0, Some(new_k), 0);
        encoder.set_buffer(1, Some(new_v), 0);
        encoder.set_buffer(2, Some(k_cache), 0);
        encoder.set_buffer(3, Some(v_cache), 0);
        
        let m_buf = self.create_buffer(&[m])?;
        let offset_buf = self.create_buffer(&[offset])?;
        let d_buf = self.create_buffer(&[d])?;
        
        encoder.set_buffer(4, Some(&m_buf), 0);
        encoder.set_buffer(5, Some(&offset_buf), 0);
        encoder.set_buffer(6, Some(&d_buf), 0);

        let w = self.update_kv_cache_pipeline.thread_execution_width();
        let h = self.update_kv_cache_pipeline.max_total_threads_per_threadgroup() / w;
        
        let threads_per_threadgroup = MTLSize::new(w, h, 1);
        let threadgroups_per_grid = MTLSize::new(
            (m as u64 + w - 1) / w,
            (d as u64 + h - 1) / h,
            1,
        );

        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }
}

