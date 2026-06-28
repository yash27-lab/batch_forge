# batch_forge Benchmarks

All numbers here are produced by `cargo run --release --bin bench` — there are **no
hard-coded results**. Re-run it on your machine to get your own. The table below
was captured on the reference device described next.

## Environment

| | |
|---|---|
| **Device** | Apple M2 (8-core GPU), unified memory |
| **OS** | macOS (Metal 4) |
| **Toolchain** | Rust 1.96, `--release` (LTO thin, codegen-units = 1) |
| **Compute** | Custom MSL kernels (no MPS), FP32 |

## Methodology

- Each op is warmed up, then timed over many iterations; the mean is reported.
- Metal timings are **end-to-end**: buffer allocation + kernel dispatch +
  `wait_until_completed` + readback. On Apple Silicon's unified memory there is
  no discrete host↔device transfer, but per-call allocation overhead **is**
  included — which is why the GPU loses at tiny sizes and wins as work grows.
- CPU is the same naive reference used for correctness (single-threaded, no SIMD
  intrinsics, simple loop-order blocking only).

## Results (Apple M2)

### Square matmul, FP32

| N | CPU (ms) | CPU GFLOP/s | Metal (ms) | Metal GFLOP/s | Speedup |
|------:|--------:|--------:|---------:|---------:|------:|
| 128 | 0.171 | 24.6 | 0.324 | 12.9 | 0.5× |
| 256 | 1.553 | 21.6 | 0.894 | 37.5 | 1.7× |
| 512 | 11.172 | 24.0 | 2.483 | 108.1 | 4.5× |

At N=128 the GPU is *slower* — dispatch + allocation overhead dominates a tiny
problem. The crossover is around N=256, and the gap widens with size.

### GELU (elementwise, 2²⁰ elements)

| CPU (ms) | Metal (ms) | Speedup |
|------:|------:|------:|
| 6.136 | 1.178 | 5.2× |

### MLP forward (256 → 1024 → 1024 → 256)

| Batch | CPU (ms) | Metal (ms) | Speedup |
|------:|------:|------:|------:|
| 1 | 1.122 | 1.826 | 0.6× |
| 8 | 8.966 | 2.270 | 3.9× |
| 32 | 35.985 | 5.556 | 6.5× |

Batching amortizes per-dispatch overhead: at batch 1 the GPU trails, by batch 32
it is 6.5× ahead.

## Honest limitations (a.k.a. headroom)

These results are from **deliberately simple kernels**. They are correct first;
fast second. Known gaps, in rough priority order:

1. **Naive matmul.** One thread per output element, no threadgroup tiling, no
   `simdgroup_matrix`. ~108 GFLOP/s is a small fraction of the M2 GPU's FP32
   peak (~3.6 TFLOP/s). A tiled kernel should close most of that gap.
2. **Per-call buffer allocation.** The ergonomic `Vec`-in/`Vec`-out methods
   allocate buffers every call. The buffer-level API (used by the KV-cache path)
   avoids this; the high-level API should pool buffers.
3. **Attention is single-thread-per-query and recomputes scores** (O(M·S·D), down
   from the previous O(M·S·D²) bug, but not yet flash-attention style with
   threadgroup reductions).
4. **No FP16/BF16 compute path yet** — everything runs in FP32.

## Reproducing

```bash
cargo run --release --bin bench
```

For end-to-end model latency with a real checkpoint:

```bash
python python/make_demo_model.py
cargo run --release --bin batch_forge -- --requests 256   # prints req/s
```
