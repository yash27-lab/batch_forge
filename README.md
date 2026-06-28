# batch_forge

A small, **correctness-first** inference runtime for Apple Silicon, written in Rust.

batch_forge loads models exported from JAX/Equinox (via [safetensors](https://github.com/huggingface/safetensors)) and runs them with custom **Metal** compute kernels. Every GPU kernel has a pure-Rust CPU reference, and the two are checked against each other by automated parity tests — so the numbers it produces are verifiable, not asserted.

> **Scope, honestly.** This is a focused engine, not a drop-in replacement for [MLX](https://github.com/ml-explore/mlx), [llama.cpp](https://github.com/ggerganov/llama.cpp), or [candle](https://github.com/huggingface/candle). What it does today — a verified op library, a Metal backend with CPU parity, a zero-copy loader, an async request engine, and an end-to-end MLP that matches its JAX/NumPy reference to ~1e-6 — it does end-to-end and tests rigorously. Transformer LM generation, quantized model pipelines, and SSM/diffusion support are on the roadmap, marked clearly below.

[![CI](https://github.com/yash27-lab/batch_forge/actions/workflows/ci.yml/badge.svg)](https://github.com/yash27-lab/batch_forge/actions/workflows/ci.yml)

## What works today

| Component | Status | Verified by |
|-----------|--------|-------------|
| CPU reference ops (matmul, linear, attention, layernorm, rmsnorm, rope, gelu, int8 dequant) | ✅ | `cargo test --lib` |
| Metal kernels for all of the above + KV-cache update | ✅ | `cargo test --test parity` (CPU↔Metal parity on-device) |
| Zero-copy `mmap` safetensors loader | ✅ | unit + e2e |
| Single-head attention with KV-cache + causal masking | ✅ | parity + cached-path integration test |
| INT8 weight-only matmul kernel | ✅ | parity vs dequantize+matmul reference |
| End-to-end MLP forward (CPU **and** Metal), verified vs JAX/NumPy | ✅ | `--verify` (matches reference to ~1e-6) |
| Async request engine (`tokio` mpsc + oneshot, backend-agnostic) | ✅ | runnable via `--requests N` |
| Reproducible microbenchmarks | ✅ | `cargo run --bin bench` |

## Roadmap (not yet implemented)

These were over-claimed in earlier versions of this README and are now tracked honestly:

- ⏳ **Transformer LM generation** — tokenizer, multi-head attention, full model wiring (the building blocks exist; the end-to-end LM does not).
- ⏳ **Quantized model pipeline** — the INT8 kernel is done and tested; loading/serving a fully quantized checkpoint is not. INT4 is not implemented.
- ⏳ **Continuous batching** — the async engine does request/response now; fusing queued requests into one dispatch is future work.
- ⏳ **State-Space Models (Mamba), Diffusion (UNet/DiT), Vulkan/WebGPU backends** — design stage only.

## Architecture

```
          safetensors (mmap, zero-copy)
                     │
            ┌────────▼─────────┐
            │   loader::SafeModel
            └────────┬─────────┘
                     │  Tensor (owned f32)
        ┌────────────▼─────────────┐
        │      model::Mlp           │  generic over Backend
        └───────┬───────────┬──────┘
                │           │
     ┌──────────▼──┐   ┌────▼───────────────┐
     │ CpuBackend  │   │ MetalBackend (MSL) │
     │ (reference) │   │  custom kernels    │
     └──────┬──────┘   └─────────┬──────────┘
            └──── parity tests ──┘   (CPU is ground truth for GPU)

   engine::RequestManager  ──  tokio mpsc/oneshot, Arc<dyn Backend>
```

The design choice that everything else hangs off: **a CPU reference defines correct numerics, and the Metal kernels are validated against it.** This is how ggml/candle stay trustworthy, and it's what lets a reviewer believe the GPU path without owning the hardware.

- `src/ops.rs` — portable, dependency-free reference implementations (the spec).
- `src/metal_backend.rs` + `src/shaders/compute.metal` — the accelerated kernels.
- `src/model.rs` — the `Backend` trait and the `Mlp` model, generic over backend.
- `src/loader.rs` — sound, owning `mmap` loader (no `transmute`/leak).
- `src/engine.rs` — async request/response inference engine.
- `tests/parity.rs` — randomized CPU↔Metal equivalence tests.

## Quickstart

```bash
# 1. Build (Apple Silicon for the Metal path; CPU path builds anywhere)
cargo build --release

# 2. Run the test suite (unit tests everywhere; parity tests on macOS)
cargo test                       # CPU unit tests
cargo test --test parity -- --nocapture   # CPU↔Metal parity (Apple Silicon)

# 3. Generate a demo model + reference  (NumPy only — no JAX needed)
python python/make_demo_model.py

# 4. Run the engine: forward on CPU + Metal, cross-check, verify vs reference
cargo run --release --bin batch_forge -- --verify reference.safetensors

# 5. Benchmark on your machine
cargo run --release --bin bench
```

Example output from step 4 (Apple M2):

```
loaded MLP: 3 layers, in=256, out=256
[cpu]   output: shape [1, 256], ‖·‖₂=6.2420, head=[+0.3035, -0.3338, …]
[metal] output: shape [1, 256], ‖·‖₂=6.2420, head=[+0.3035, -0.3338, …]
[check] CPU vs Metal max|Δ| = 7.749e-7
[verify] PASS — max|Δ| vs reference = 1.311e-6 (tol 1e-3)
```

### Exporting your own Equinox model

```bash
pip install -r python/requirements.txt          # jax, equinox, safetensors, numpy
python python/export_eqx.py --out model.safetensors --ref reference.safetensors
cargo run --release --bin batch_forge -- --verify reference.safetensors
```

The exporter names weights `layers.{i}.weight` / `layers.{i}.bias` and writes a sample `input`/`output` pair the Rust engine checks itself against.

## Performance & correctness

Both are measured, reproducible, and documented — no hard-coded results:

- **[docs/benchmarks.md](docs/benchmarks.md)** — real CPU-vs-Metal numbers from `cargo run --bin bench`, with methodology and known limitations (the kernels are intentionally naive — there is large, honest headroom).
- **[docs/correctness.md](docs/correctness.md)** — the parity-testing methodology and the actual observed CPU↔Metal deviations per operator.

## Contributing

The highest-value next steps are tiled/`simdgroup_matrix` matmul, a real tokenizer + multi-head attention to reach transformer generation, and a quantized checkpoint loader. Any new kernel **must** ship with a CPU reference in `ops.rs` and a parity test in `tests/parity.rs`.

## License

MIT OR Apache-2.0
