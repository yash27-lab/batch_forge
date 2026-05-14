# Troubleshooting

## Metal backend is unavailable

The Metal backend requires macOS on Apple Silicon. On other platforms, run the portable CPU reference with a demo checkpoint:

```bash
python python/make_demo_model.py
cargo run --release -- --backend cpu
```

## GPT-2 reference assets are missing

Model weights and tokenizer files are intentionally not committed. Download them before running the GPT-2 end-to-end reference check:

```bash
python python/fetch_gpt2.py
```

The files are placed under `models/gpt2/` and require roughly 550 MB of disk space.

## Parity tests need a Mac with Metal

`cargo test --test parity -- --nocapture` validates Metal kernels against the pure-Rust CPU reference, so it must run on an Apple Silicon Mac. The regular library test suite remains useful for CPU-only environments:

```bash
cargo test --lib
```

## `cargo fmt` is unavailable

Install Rust's formatting component for the active toolchain, then rerun the formatting check:

```bash
rustup component add rustfmt
cargo fmt --check
```

## `cargo clippy` is unavailable

Install the Clippy component for the active toolchain, then rerun the lint check:

```bash
rustup component add clippy
cargo clippy --all-targets
```

## Generation arguments and checkpoints

Run cargo run --bin generate -- --help for generation options. Unknown flags,
missing values, invalid integers, and non-finite or negative temperatures return
an error before loading assets. A zero top-k disables filtering; --greedy takes
precedence over temperature regardless of option order.

A checkpoint must match the default GPT-2 small configuration. Incorrect shapes,
inconsistent tokenizer IDs, and mismatched vocabulary sizes are reported at load
time. Keep model.safetensors, vocab.json, and merges.txt from the same model
under the directory selected by --model-dir.

Generation streams complete UTF-8 characters even when a character spans tokens.
Diagnostics go to stderr. A consumer that closes a pipe early stops generation
without a panic. Invalid bytes are replaced, and an unfinished final character
is replaced only at the end of the stream.

## MLP verification exit status

The default MLP binary returns failure for missing checkpoints, invalid input
buffers, failed inference requests, and failed numerical checks. A reference
file selected with --verify must contain both input and output; a
synthetic fallback is not used to claim verification.

## Detailed usage references

See the [generation guide](generation.md) for backend, sampling, output, and
context-window behavior. The [tensor guide](tensors.md) explains shape and
buffer errors; the [engine guide](engine.md) explains queue bounds and shutdown
behavior. Use the [test command reference](test-matrix.md) to choose a local
check for a reproduction.
