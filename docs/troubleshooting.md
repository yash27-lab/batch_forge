# Troubleshooting

## Metal backend is unavailable

The Metal backend requires macOS on Apple Silicon. On other platforms, run the portable CPU reference with a demo checkpoint:

```bash
python python/make_demo_model.py
cargo run --release -- --backend cpu
```

## GPT-2 assets are missing

Model weights and tokenizer files are intentionally not committed. Download them before running generation or the end-to-end reference check:

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
