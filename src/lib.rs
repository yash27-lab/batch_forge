//! batch_forge — a small, correctness-first inference runtime for Apple Silicon.
//!
//! The crate is split into a portable CPU reference (`ops`) that defines the
//! ground-truth numerics for every operator, and a Metal backend
//! (`metal_backend`) whose kernels are validated against that reference by the
//! parity tests in `tests/`. This mirrors how production engines (ggml, candle)
//! keep a CPU reference next to each accelerated kernel. On non-macOS targets,
//! the portable CPU modules remain available while the Metal-specific modules
//! are omitted.

/// Asynchronous request/response inference engine built on Tokio channels.
pub mod engine;
/// From-scratch GPT-2 inference shared by the CPU and Metal backends.
pub mod gpt2;
/// Zero-copy SafeTensors loading backed by memory mapping.
pub mod loader;
/// Model definitions and backend abstractions used to run them.
pub mod model;
/// Pure-Rust reference implementations used to verify accelerated kernels.
pub mod ops;
/// Lightweight tensor types and dtype conversions shared by the backends.
pub mod tensor;
/// From-scratch byte-level BPE tokenization for GPT-2-compatible vocabularies.
pub mod tokenizer;

#[cfg(target_os = "macos")]
pub mod kv_cache;

#[cfg(target_os = "macos")]
pub mod metal_backend;

/// Default Metal shader source, embedded at compile time.
#[cfg(target_os = "macos")]
pub const SHADER_SOURCE: &str = include_str!("shaders/compute.metal");
