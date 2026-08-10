# batch_forge

A from-scratch **GPT-2 inference engine** for Apple Silicon, written in Rust with hand-written **Metal** compute kernels.

No PyTorch, no Python runtime, no `tokenizers` library. batch_forge loads real HuggingFace `gpt2` weights, tokenizes with its own byte-level BPE, runs the transformer on custom Metal kernels, and generates text — and **every GPU kernel is checked against a pure-Rust CPU reference**, so its output is verifiable, not asserted.

[![CI](https://github.com/yash27-lab/batch_forge/actions/workflows/ci.yml/badge.svg)](https://github.com/yash27-lab/batch_forge/actions/workflows/ci.yml)

```
$ cargo run --release --bin generate -- --prompt "The meaning of life is" --greedy

The meaning of life is not the same as the meaning of death.

[25 tokens in 3.06s = 8.2 tok/s on metal]
```

```
$ cargo run --release --bin generate -- \
      --prompt "In a shocking turn of events, scientists discovered" --temperature 0.7

In a shocking turn of events, scientists discovered that when they were forced to
eat a "clean meal" each morning, they found that they were nearly three times more
likely to lose weight.
```

That text is produced by a 124M-parameter GPT-2 running on the Metal backend. The next-token predictions are **bit-for-bit rank-identical to HuggingFace `transformers`**, verified by [`tests/gpt2_e2e.rs`](tests/gpt2_e2e.rs) and the NumPy reference in [`python/gpt2_reference.py`](python/gpt2_reference.py) — CPU and Metal logits agree to **9e-5**.

## Why this is more than a toy

The hard part of an inference engine isn't the architecture — it's being *correct*. batch_forge keeps a pure-Rust CPU implementation of every operator as the ground truth, and validates each Metal kernel against it on randomized inputs ([`tests/parity.rs`](tests/parity.rs)). That discipline caught a real bug during development:

> GPT-2's GELU drove the tanh argument past ~70. Metal's fast-math `tanh` evaluates `exp(2·arg)`, which **overflows f32 to inf → NaN**, while Rust's CPU `tanh` saturates correctly. Every individual kernel passed parity at small magnitudes; only the composed forward produced `NaN`. The CPU reference + a magnitude-scaled parity test pinned it to GELU in minutes. Fix: clamp the tanh argument (it's saturated to ±1 by |arg|=15 anyway). That regression test now lives in the suite.

That is the whole point of the design: a reviewer can trust the GPU path without owning a Mac, because the tests prove CPU≡Metal.

## What works today

| Component | Status | Verified by |
|-----------|--------|-------------|
| **GPT-2 (124M) text generation, CPU + Metal** | Done | `tests/gpt2_e2e.rs` (rank-identical to HF) |
| From-scratch byte-level **BPE tokenizer** | Done | `encode("Hello world") == [15496, 995]`, round-trips |
| **Metal kernels**: tiled matmul, multi-head attention, layernorm, rmsnorm, rope, gelu, int8 dequant | Done | `cargo test --test parity` (CPU↔Metal on-device) |
| Pure-Rust CPU reference for every op | Done | `cargo test --lib` |
| **Tiled matmul** (threadgroup shared memory) | Done | 1.8× over naive @ 512, 296 GF/s @ 1024 |
| Zero-copy `mmap` safetensors loader (unaligned-safe) | Done | unit + e2e |
| Sampling: greedy, temperature, top-k | Done | demo |
| Async request engine (`tokio` mpsc/oneshot) | Done | `--requests N` on the MLP path |

## Roadmap (not yet built — stated honestly)

- **KV cache for generation.** Today each step recomputes the full sequence (`O(n²)` over the context). The cache kernels exist (`update_kv_cache`, `kv_attention`); wiring them into the GPT-2 loop is next and is the biggest generation speedup available.
- **Resident weights.** The ergonomic op API re-uploads weights to the GPU each call; pooling/persisting them is a large, easy win.
- **FP16/BF16 compute**, **larger GPT-2 / Llama**, **INT4**, **flash-attention-style fused kernel**, **Vulkan/WebGPU**.

This is not competing with [MLX](https://github.com/ml-explore/mlx) / [llama.cpp](https://github.com/ggerganov/llama.cpp) / [candle](https://github.com/huggingface/candle). It's a correctness-first engine that runs a real LLM end-to-end and proves it.

## Architecture

```
   prompt ──► BPE tokenizer (from scratch) ──► token ids
                                                  │
   gpt2 safetensors ──► mmap loader ──► Gpt2 ◄────┘
                                         │  forward<B: LlmOps>
                          ┌──────────────┴───────────────┐
                          ▼                               ▼
                    CpuBackend (ops.rs)          MetalBackend (compute.metal)
                    the ground truth      ◄─ parity ─►  tiled matmul · MHA ·
                                                         layernorm · gelu · …
```

- `src/gpt2.rs` — model, the `LlmOps` backend trait, forward pass, sampling.
- `src/tokenizer.rs` — byte-level BPE (`bytes_to_unicode`, merges, pre-tokenizer).
- `src/ops.rs` — pure-Rust reference numerics (the spec).
- `src/metal_backend.rs` + `src/shaders/compute.metal` — the Metal kernels.
- `tests/parity.rs`, `tests/gpt2_e2e.rs` — CPU↔Metal + end-to-end verification.

## Quickstart

### Requirements

- Rust 1.75 or later
- Apple Silicon and macOS for the Metal backend. The CPU backend builds on other platforms; use `--backend cpu` there.

```bash
# 1. Build (Apple Silicon for Metal; the CPU path builds anywhere)
cargo build --release

# 2. Get GPT-2 weights + tokenizer (~550 MB, gitignored)
python python/fetch_gpt2.py        # downloads into models/gpt2/

# 3. Generate
cargo run --release --bin generate -- --prompt "Once upon a time" --max-new 60

# 4. Verify against the NumPy/HuggingFace reference
python python/gpt2_reference.py            # prints HF predictions
cargo test --test gpt2_e2e -- --nocapture  # asserts Rust matches

# 5. Tests + benchmarks
cargo test --test parity -- --nocapture    # CPU↔Metal parity (Apple Silicon)
cargo run --release --bin bench            # matmul/gelu/MLP numbers on your machine
```

`generate` flags: `--prompt/-p`, `--max-new/-n`, `--temperature/-t`, `--top-k/-k`, `--seed/-s`, `--greedy`, `--backend cpu|metal`.

## Performance & correctness

Both measured and reproducible — see [docs/benchmarks.md](docs/benchmarks.md) (real M2 numbers, tiled vs naive matmul) and [docs/correctness.md](docs/correctness.md) (per-op CPU↔Metal deviations + the GPT-2 end-to-end check). For common setup and platform questions, see the [troubleshooting guide](docs/troubleshooting.md); browse the [documentation index](docs/README.md) for the complete guide list. Contributors can start with [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT OR Apache-2.0
