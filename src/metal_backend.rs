//! Metal compute backend.
//!
//! Each method mirrors a function in [`crate::ops`] and is validated against it
//! by `tests/parity.rs`. Buffers use Metal's unified (shared) memory, so on
//! Apple Silicon there is no discrete host↔device copy — `create_buffer` and
//! `read_buffer` read/write the same physical pages the GPU sees.

use bytemuck::Pod;
use metal::{
    Buffer, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};
use thiserror::Error;
use tracing::info;

use crate::model::Backend;
use crate::tensor::Tensor;

#[derive(Error, Debug)]
pub enum BackendError {
    #[error("Buffer size computation overflowed")]
    BufferOverflow,
    #[error("Initialization error: {0}")]
    Init(String),
}

pub struct MetalBackend {
    pub device: Device,
    command_queue: CommandQueue,
    #[allow(dead_code)] // kept alive so pipeline states remain valid
    library: Library,
    matmul_pipeline: ComputePipelineState,
    linear_pipeline: ComputePipelineState,
    quant_matmul_pipeline: ComputePipelineState,
    gelu_pipeline: ComputePipelineState,
    kv_attention_pipeline: ComputePipelineState,
    update_kv_cache_pipeline: ComputePipelineState,
    layernorm_pipeline: ComputePipelineState,
    rmsnorm_pipeline: ComputePipelineState,
    rope_pipeline: ComputePipelineState,
    matmul_tiled_pipeline: ComputePipelineState,
    mha_pipeline: ComputePipelineState,
}

impl MetalBackend {
    pub fn new(shader_source: &str) -> Result<Self, BackendError> {
        let device = Device::system_default()
            .ok_or_else(|| BackendError::Init("No Metal device found. Are you on a Mac?".into()))?;
        info!("Initialized Metal device: {}", device.name());

        let command_queue = device.new_command_queue();
        let library = device
            .new_library_with_source(shader_source, &metal::CompileOptions::new())
            .map_err(|e| BackendError::Init(format!("Failed to compile shaders: {e}")))?;

        let pso = |name: &str| -> Result<ComputePipelineState, BackendError> {
            let func = library
                .get_function(name, None)
                .map_err(|e| BackendError::Init(format!("missing kernel '{name}': {e}")))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| BackendError::Init(format!("pipeline '{name}': {e}")))
        };

        Ok(Self {
            matmul_pipeline: pso("matmul")?,
            linear_pipeline: pso("linear")?,
            quant_matmul_pipeline: pso("quant_matmul")?,
            gelu_pipeline: pso("gelu_forward")?,
            kv_attention_pipeline: pso("kv_attention")?,
            update_kv_cache_pipeline: pso("update_kv_cache")?,
            layernorm_pipeline: pso("layernorm")?,
            rmsnorm_pipeline: pso("rmsnorm")?,
            rope_pipeline: pso("rope")?,
            matmul_tiled_pipeline: pso("matmul_tiled")?,
            mha_pipeline: pso("mha")?,
            command_queue,
            library,
            device,
        })
    }

    // --- buffer helpers ----------------------------------------------------

    pub fn create_buffer<T>(&self, data: &[T]) -> Result<Buffer, BackendError> {
        let length = std::mem::size_of_val(data);
        Ok(self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            length as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }

    pub fn create_buffer_uninitialized<T>(&self, len: usize) -> Result<Buffer, BackendError> {
        let length = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(BackendError::BufferOverflow)?;
        Ok(self
            .device
            .new_buffer(length.max(1) as u64, MTLResourceOptions::StorageModeShared))
    }

    /// Copies `len` elements of type `T` out of a shared buffer.
    pub fn read_buffer<T: Pod>(&self, buffer: &Buffer, len: usize) -> Vec<T> {
        let mut out = vec![T::zeroed(); len];
        let ptr = buffer.contents() as *const T;
        // SAFETY: shared-storage buffer, `len` elements were allocated/written.
        unsafe { std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len) };
        out
    }

    // --- dispatch helpers --------------------------------------------------

    fn run_1d(
        &self,
        pso: &ComputePipelineState,
        set: impl FnOnce(&ComputeCommandEncoderRef),
        n: u64,
    ) {
        let cb = self.command_queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        set(enc);
        let max = pso.max_total_threads_per_threadgroup();
        let tgw = n.clamp(1, max);
        enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(tgw), 1, 1), MTLSize::new(tgw, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    fn run_2d(
        &self,
        pso: &ComputePipelineState,
        set: impl FnOnce(&ComputeCommandEncoderRef),
        gx: u64,
        gy: u64,
    ) {
        let cb = self.command_queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        set(enc);
        let w = pso.thread_execution_width();
        let h = (pso.max_total_threads_per_threadgroup() / w).max(1);
        enc.dispatch_thread_groups(
            MTLSize::new(gx.div_ceil(w), gy.div_ceil(h), 1),
            MTLSize::new(w, h, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // --- high-level ops (Vec in / Vec out, used by parity tests) -----------

    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let ba = self.create_buffer(a).unwrap();
        let bb = self.create_buffer(b).unwrap();
        let bc = self.create_buffer_uninitialized::<f32>(m * n).unwrap();
        let (bm, bn, bk) = self.dims3(m, n, k);
        self.run_2d(
            &self.matmul_pipeline,
            |e| {
                e.set_buffer(0, Some(&ba), 0);
                e.set_buffer(1, Some(&bb), 0);
                e.set_buffer(2, Some(&bc), 0);
                e.set_buffer(3, Some(&bm), 0);
                e.set_buffer(4, Some(&bn), 0);
                e.set_buffer(5, Some(&bk), 0);
            },
            n as u64,
            m as u64,
        );
        self.read_buffer(&bc, m * n)
    }

    /// Shared-memory tiled matmul: `C[m,n] = A[m,k] * B[k,n]`. Same result as
    /// [`MetalBackend::matmul`] but uses full 16x16 threadgroups with on-chip tiles.
    pub fn matmul_tiled(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        const TILE: u64 = 16;
        let ba = self.create_buffer(a).unwrap();
        let bb = self.create_buffer(b).unwrap();
        let bc = self.create_buffer_uninitialized::<f32>(m * n).unwrap();
        let (bm, bn, bk) = self.dims3(m, n, k);
        let cb = self.command_queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.matmul_tiled_pipeline);
        enc.set_buffer(0, Some(&ba), 0);
        enc.set_buffer(1, Some(&bb), 0);
        enc.set_buffer(2, Some(&bc), 0);
        enc.set_buffer(3, Some(&bm), 0);
        enc.set_buffer(4, Some(&bn), 0);
        enc.set_buffer(5, Some(&bk), 0);
        // Full TILExTILE threadgroups so boundary threads still load zeros and
        // participate in the barriers.
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(TILE), (m as u64).div_ceil(TILE), 1),
            MTLSize::new(TILE, TILE, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        self.read_buffer(&bc, m * n)
    }

    /// Multi-head causal self-attention. Q/K/V are `[seq, heads*head_dim]`.
    pub fn mha(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let bq = self.create_buffer(q).unwrap();
        let bk = self.create_buffer(k).unwrap();
        let bv = self.create_buffer(v).unwrap();
        let bo = self
            .create_buffer_uninitialized::<f32>(seq * heads * head_dim)
            .unwrap();
        let (bs, bh, bd) = self.dims3(seq, heads, head_dim);
        let cb = self.command_queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.mha_pipeline);
        enc.set_buffer(0, Some(&bq), 0);
        enc.set_buffer(1, Some(&bk), 0);
        enc.set_buffer(2, Some(&bv), 0);
        enc.set_buffer(3, Some(&bo), 0);
        enc.set_buffer(4, Some(&bs), 0);
        enc.set_buffer(5, Some(&bh), 0);
        enc.set_buffer(6, Some(&bd), 0);
        // One thread per (query, head); non-uniform dispatch handles the edges.
        let th = heads.min(1024) as u64;
        let tw = (1024 / th).min(seq as u64).max(1);
        enc.dispatch_threads(
            MTLSize::new(seq as u64, heads as u64, 1),
            MTLSize::new(tw, th, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        self.read_buffer(&bo, seq * heads * head_dim)
    }

    pub fn linear(
        &self,
        x: &[f32],
        w: &[f32],
        bias: &[f32],
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) -> Vec<f32> {
        let bx = self.create_buffer(x).unwrap();
        let bw = self.create_buffer(w).unwrap();
        let bb = self.create_buffer(bias).unwrap();
        let by = self
            .create_buffer_uninitialized::<f32>(rows * out_f)
            .unwrap();
        let (br, bi, bo) = self.dims3(rows, in_f, out_f);
        self.run_2d(
            &self.linear_pipeline,
            |e| {
                e.set_buffer(0, Some(&bx), 0);
                e.set_buffer(1, Some(&bw), 0);
                e.set_buffer(2, Some(&bb), 0);
                e.set_buffer(3, Some(&by), 0);
                e.set_buffer(4, Some(&br), 0);
                e.set_buffer(5, Some(&bi), 0);
                e.set_buffer(6, Some(&bo), 0);
            },
            out_f as u64,
            rows as u64,
        );
        self.read_buffer(&by, rows * out_f)
    }

    pub fn quant_matmul(
        &self,
        a_int8: &[i8],
        scales: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let ba = self.create_buffer(a_int8).unwrap();
        let bb = self.create_buffer(b).unwrap();
        let bc = self.create_buffer_uninitialized::<f32>(m * n).unwrap();
        let bs = self.create_buffer(scales).unwrap();
        let (bm, bn, bk) = self.dims3(m, n, k);
        self.run_2d(
            &self.quant_matmul_pipeline,
            |e| {
                e.set_buffer(0, Some(&ba), 0);
                e.set_buffer(1, Some(&bb), 0);
                e.set_buffer(2, Some(&bc), 0);
                e.set_buffer(3, Some(&bs), 0);
                e.set_buffer(4, Some(&bm), 0);
                e.set_buffer(5, Some(&bn), 0);
                e.set_buffer(6, Some(&bk), 0);
            },
            n as u64,
            m as u64,
        );
        self.read_buffer(&bc, m * n)
    }

    pub fn gelu(&self, x: &[f32]) -> Vec<f32> {
        let bx = self.create_buffer(x).unwrap();
        let by = self.create_buffer_uninitialized::<f32>(x.len()).unwrap();
        let bn = self.create_buffer(&[x.len() as u32]).unwrap();
        self.run_1d(
            &self.gelu_pipeline,
            |e| {
                e.set_buffer(0, Some(&bx), 0);
                e.set_buffer(1, Some(&by), 0);
                e.set_buffer(2, Some(&bn), 0);
            },
            x.len() as u64,
        );
        self.read_buffer(&by, x.len())
    }

    pub fn layernorm(
        &self,
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Vec<f32> {
        let bx = self.create_buffer(x).unwrap();
        let bg = self.create_buffer(gamma).unwrap();
        let bb = self.create_buffer(beta).unwrap();
        let by = self.create_buffer_uninitialized::<f32>(rows * d).unwrap();
        let br = self.create_buffer(&[rows as u32]).unwrap();
        let bd = self.create_buffer(&[d as u32]).unwrap();
        let be = self.create_buffer(&[eps]).unwrap();
        self.run_1d(
            &self.layernorm_pipeline,
            |e| {
                e.set_buffer(0, Some(&bx), 0);
                e.set_buffer(1, Some(&bg), 0);
                e.set_buffer(2, Some(&bb), 0);
                e.set_buffer(3, Some(&by), 0);
                e.set_buffer(4, Some(&br), 0);
                e.set_buffer(5, Some(&bd), 0);
                e.set_buffer(6, Some(&be), 0);
            },
            rows as u64,
        );
        self.read_buffer(&by, rows * d)
    }

    pub fn rmsnorm(&self, x: &[f32], gamma: &[f32], rows: usize, d: usize, eps: f32) -> Vec<f32> {
        let bx = self.create_buffer(x).unwrap();
        let bg = self.create_buffer(gamma).unwrap();
        let by = self.create_buffer_uninitialized::<f32>(rows * d).unwrap();
        let br = self.create_buffer(&[rows as u32]).unwrap();
        let bd = self.create_buffer(&[d as u32]).unwrap();
        let be = self.create_buffer(&[eps]).unwrap();
        self.run_1d(
            &self.rmsnorm_pipeline,
            |e| {
                e.set_buffer(0, Some(&bx), 0);
                e.set_buffer(1, Some(&bg), 0);
                e.set_buffer(2, Some(&by), 0);
                e.set_buffer(3, Some(&br), 0);
                e.set_buffer(4, Some(&bd), 0);
                e.set_buffer(5, Some(&be), 0);
            },
            rows as u64,
        );
        self.read_buffer(&by, rows * d)
    }

    pub fn rope(
        &self,
        x: &[f32],
        positions: &[u32],
        rows: usize,
        d: usize,
        theta: f32,
    ) -> Vec<f32> {
        let bx = self.create_buffer(x).unwrap();
        let bp = self.create_buffer(positions).unwrap();
        let br = self.create_buffer(&[rows as u32]).unwrap();
        let bd = self.create_buffer(&[d as u32]).unwrap();
        let bt = self.create_buffer(&[theta]).unwrap();
        self.run_1d(
            &self.rope_pipeline,
            |e| {
                e.set_buffer(0, Some(&bx), 0);
                e.set_buffer(1, Some(&bp), 0);
                e.set_buffer(2, Some(&br), 0);
                e.set_buffer(3, Some(&bd), 0);
                e.set_buffer(4, Some(&bt), 0);
            },
            rows as u64,
        );
        self.read_buffer(&bx, rows * d)
    }

    /// Single-head attention over Vec inputs (used by parity tests).
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        m: usize,
        seq: usize,
        d: usize,
        causal: bool,
        q_offset: usize,
    ) -> Vec<f32> {
        let bq = self.create_buffer(q).unwrap();
        let bk = self.create_buffer(k).unwrap();
        let bv = self.create_buffer(v).unwrap();
        let bo = self.create_buffer_uninitialized::<f32>(m * d).unwrap();
        self.kv_attention(&bq, &bk, &bv, &bo, m, seq, d, causal, q_offset);
        self.read_buffer(&bo, m * d)
    }

    // --- buffer-level ops (used by the cached generation loop) -------------

    /// Attention reading K/V straight from persistent cache buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_attention(
        &self,
        q: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        o: &Buffer,
        m: usize,
        cur_seq_len: usize,
        d: usize,
        causal: bool,
        q_offset: usize,
    ) {
        let bm = self.create_buffer(&[m as u32]).unwrap();
        let bs = self.create_buffer(&[cur_seq_len as u32]).unwrap();
        let bd = self.create_buffer(&[d as u32]).unwrap();
        let bc = self.create_buffer(&[causal as u32]).unwrap();
        let bo = self.create_buffer(&[q_offset as u32]).unwrap();
        self.run_1d(
            &self.kv_attention_pipeline,
            |e| {
                e.set_buffer(0, Some(q), 0);
                e.set_buffer(1, Some(k_cache), 0);
                e.set_buffer(2, Some(v_cache), 0);
                e.set_buffer(3, Some(o), 0);
                e.set_buffer(4, Some(&bm), 0);
                e.set_buffer(5, Some(&bs), 0);
                e.set_buffer(6, Some(&bd), 0);
                e.set_buffer(7, Some(&bc), 0);
                e.set_buffer(8, Some(&bo), 0);
            },
            m as u64,
        );
    }

    /// Writes `m` new K/V rows into the cache buffers at `offset`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_kv_cache(
        &self,
        new_k: &Buffer,
        new_v: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        m: usize,
        offset: usize,
        d: usize,
    ) {
        let bm = self.create_buffer(&[m as u32]).unwrap();
        let boff = self.create_buffer(&[offset as u32]).unwrap();
        let bd = self.create_buffer(&[d as u32]).unwrap();
        self.run_2d(
            &self.update_kv_cache_pipeline,
            |e| {
                e.set_buffer(0, Some(new_k), 0);
                e.set_buffer(1, Some(new_v), 0);
                e.set_buffer(2, Some(k_cache), 0);
                e.set_buffer(3, Some(v_cache), 0);
                e.set_buffer(4, Some(&bm), 0);
                e.set_buffer(5, Some(&boff), 0);
                e.set_buffer(6, Some(&bd), 0);
            },
            m as u64,
            d as u64,
        );
    }

    fn dims3(&self, a: usize, b: usize, c: usize) -> (Buffer, Buffer, Buffer) {
        (
            self.create_buffer(&[a as u32]).unwrap(),
            self.create_buffer(&[b as u32]).unwrap(),
            self.create_buffer(&[c as u32]).unwrap(),
        )
    }
}

impl Backend for MetalBackend {
    fn linear(&self, x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
        let (n, in_f) = x.dims2().expect("linear input must be 2-D");
        let (out_f, in_w) = w.dims2().expect("weight must be 2-D");
        assert_eq!(in_f, in_w, "linear in-features mismatch");
        let data = MetalBackend::linear(self, &x.data, &w.data, &b.data, n, in_f, out_f);
        Tensor::new(data, vec![n, out_f]).expect("linear output shape is consistent")
    }

    fn gelu(&self, x: &Tensor) -> Tensor {
        let data = MetalBackend::gelu(self, &x.data);
        Tensor::new(data, x.shape.clone()).expect("gelu preserves shape")
    }

    fn name(&self) -> &'static str {
        "metal"
    }
}
