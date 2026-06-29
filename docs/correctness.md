# batch_forge Correctness

An inference engine is only useful if you can trust its output. batch_forge backs
that trust with a concrete, runnable method rather than a promise:

> **The pure-Rust CPU implementation in `src/ops.rs` is the ground truth. Every
> Metal kernel is tested against it on randomized inputs.** When the GPU and the
> reference agree to within tolerance, the kernel is correct by construction of
> the test.

This is the same discipline ggml and candle use, and it means a reviewer can
believe the Metal path without owning a Mac — the test either passes in CI-visible
output or it doesn't.

## How verification works

1. **CPU unit tests** (`cargo test --lib`) pin the reference ops to known values
   (e.g. GELU reference points, softmax sums to 1, RoPE preserves norm, causal
   masking hides future keys).
2. **CPU↔Metal parity tests** (`cargo test --test parity`) generate random inputs,
   run the reference and the Metal kernel, and assert the maximum absolute
   deviation is within tolerance. Run on Apple Silicon.
3. **End-to-end verification** (`--verify`) compares the full model forward pass
   against a reference `output` exported alongside the weights by the Python
   tooling (JAX in `export_eqx.py`, NumPy in `make_demo_model.py`).

```bash
cargo test --lib
cargo test --test parity -- --nocapture
python python/make_demo_model.py
cargo run --release --bin batch_forge -- --verify reference.safetensors
```

## Observed deviations (Apple M2, FP32)

Measured by `cargo test --test parity -- --nocapture`. These are the *actual*
maximum absolute differences between the Metal kernel and the CPU reference, not
the (looser) thresholds the tests assert against.

| Operator | Test tolerance | Observed max\|Δ\| | Notes |
|----------|---------------:|-------------------:|-------|
| MatMul (32×64×48) | 1e-3 | 1.4e-6 | FP32 accumulation order differs |
| Linear (16×64×40) | 1e-3 | 1.4e-6 | `y = x·Wᵀ + b` |
| GELU (4096) | 1e-5 | 7.5e-8 | tanh approximation |
| LayerNorm (16×64) | 1e-4 | 2.4e-7 | population variance |
| RMSNorm (16×64) | 1e-4 | 1.2e-7 | |
| RoPE (8×64) | 1e-3 | 6.0e-7 | rotate-half, `sin/cos/pow` |
| Attention, full (m=4,s=12,d=32) | 1e-3 | 8.9e-8 | |
| Attention, causal (q_offset=3) | 1e-3 | 1.2e-7 | masking + softmax |
| INT8 quant matmul (24×48×32) | 1e-3 | 7.6e-6 | vs dequantize+matmul |
| Tiled matmul (40×72×56) | 1e-3 | 1.9e-6 | shared-memory GEMM |
| Multi-head attention (s=7,h=3,d=8) | 1e-3 | 1.2e-7 | causal, GPT-2 layout |
| GELU at scale (5×3072, ±12) | 1e-2 | 4.8e-7 | regression test for the tanh-overflow fix |
| Cached attention (e2e) | 1e-3 | 6.0e-8 | `update_kv_cache` → `kv_attention` |
| KV-cache write | exact | 0 | bitwise copy check |
| **MLP forward vs reference** | 1e-3 | **1.3e-6** | full model, CPU & Metal |
| **GPT-2 logits, CPU vs Metal** | 1e-2 | **9.2e-5** | full 12-layer forward |
| **GPT-2 top-5 vs HuggingFace** | exact | match | `tests/gpt2_e2e.rs` |

### The GELU / tanh-overflow bug

A worked example of why the CPU reference matters. GPT-2 activations drove the
GELU tanh argument past ~70. Metal's fast-math `tanh` evaluates `exp(2·arg)`,
which overflows f32 to `inf` and yields `NaN`; Rust's CPU `tanh` saturates. Every
op passed parity at small magnitudes, so the bug only surfaced in the composed
forward as all-`NaN` logits. A magnitude-scaled GELU parity test localized it
immediately, and the fix (clamping the tanh argument, exact since tanh saturates
by |arg|=15) is now a permanent regression test (the "GELU at scale" row above).

Deviations are dominated by FP32 summation-order differences between the
sequential CPU loop and the parallel GPU kernel — i.e. the kernels are doing the
same math, just associating the additions differently. Tolerances are set with
generous margin above the observed values.

## Tolerance rationale

| Precision | Typical bound | Why |
|-----------|---------------|-----|
| FP32 | 1e-3 abs (observed ~1e-6) | accumulation-order differences only |
| INT8 (W8A16) | 1e-3 abs (observed ~7e-6) | per-row scale dequant is exact up to FP32 rounding |
| FP16 / BF16 | n/a | no half-precision compute path yet (roadmap) |

## What is *not* yet covered

Being explicit, since the previous version of this doc claimed tests that did not
exist:

- No FP16/BF16 numerics (no half-precision kernels yet).
- No SSM `scan` op (not implemented).
- No full transformer-LM end-to-end parity (no LM yet) — only the MLP is verified
  end-to-end, plus every individual operator above.
