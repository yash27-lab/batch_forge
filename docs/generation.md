# GPT-2 generation guide

The `generate` binary runs GPT-2 small (124M). The default `batch_forge` binary
runs exported MLP checkpoints. Invoke the generation binary explicitly:

```bash
python3 python/fetch_gpt2.py
cargo run --release --bin generate -- --backend cpu --prompt "Hello" --greedy
```

Run commands from the repository root. Downloads are setup work; generation
runs in Rust and does not require a Python runtime.

## Choose an asset directory

Keep `model.safetensors`, `vocab.json`, and `merges.txt` together. The download
script's `--out` directory and the generator's `--model-dir` must refer to the
same location. Quote paths that contain spaces:

```bash
python3 python/fetch_gpt2.py --out "/path/to/gpt2 assets"
cargo run --release --bin generate -- --model-dir "/path/to/gpt2 assets" --backend cpu
```

Existing nonempty assets are skipped by the download script. Its `--force`
option replaces them after a completed transfer; `--timeout SECONDS` sets the
socket timeout. Failed transfers leave an existing final asset intact.
