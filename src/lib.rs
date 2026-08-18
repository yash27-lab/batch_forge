//! batch_forge — a small, correctness-first inference runtime for Apple Silicon.
//!
//! The crate is split into a portable CPU reference (`ops`) that defines the
//! ground-truth numerics for every operator, and a Metal backend
//! (`metal_backend`) whose kernels are validated against that reference by the
//! parity tests in `tests/`. This mirrors how production engines (ggml, candle)
//! keep a CPU reference next to each accelerated kernel. On non-macOS targets,
//! the portable CPU modules remain available while the Metal-specific modules
//! are omitted.

pub mod engine;
pub mod gpt2;
pub mod loader;
pub mod model;
pub mod ops;
pub mod tensor;
pub mod tokenizer;

#[cfg(target_os = "macos")]
pub mod kv_cache;

#[cfg(target_os = "macos")]
pub mod metal_backend;

/// Default Metal shader source, embedded at compile time.
#[cfg(target_os = "macos")]
pub const SHADER_SOURCE: &str = include_str!("shaders/compute.metal");
