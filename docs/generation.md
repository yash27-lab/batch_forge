# GPT-2 generation guide

The `generate` binary runs GPT-2 small (124M). The default `batch_forge` binary
runs exported MLP checkpoints. Invoke the generation binary explicitly:

```bash
python3 python/fetch_gpt2.py
cargo run --release --bin generate -- --backend cpu --prompt "Hello" --greedy
```

Run commands from the repository root. Downloads are setup work; generation
runs in Rust and does not require a Python runtime.
