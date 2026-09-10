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

The numbers below are a reference point, not a cross-device ranking; Apple GPU performance varies with chip generation, OS version, and thermal conditions.

## Results (Apple M2)

### GPT-2 (124M) text generation

End-to-end autoregressive generation on the Metal backend, no KV cache yet
(each step recomputes the full context):

| Backend | Throughput |
|---------|-----------:|
| Metal (M2) | ~8 tok/s |

The dominant cost today is (a) recomputing the whole sequence each step and
(b) re-uploading weights to the GPU per call. Both are on the roadmap; a KV cache
alone removes the `O(n²)` blowup.

### Square matmul, FP32 — naive vs tiled Metal

The tiled kernel stages 16×16 tiles into threadgroup memory; the speedup over the
naive one-thread-per-output kernel grows with size:

| N | CPU GF/s | naive GF/s | tiled GF/s | tiled / naive |
|------:|--------:|--------:|--------:|------:|
| 128 | 33.4 | 15.0 | 16.1 | 1.1× |
| 256 | 27.6 | 38.5 | 52.3 | 1.4× |
| 512 | 26.2 | 113.6 | 202.4 | **1.8×** |
| 1024 | 28.2 | 206.9 | 295.7 | 1.4× |

At N=128 the GPU is *slower* than CPU — dispatch + allocation overhead dominates a
tiny problem. Tiled matmul reaches ~296 GF/s at N=1024 (still well below the M2's
FP32 peak; a `simdgroup_matrix` kernel is the next step).

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
