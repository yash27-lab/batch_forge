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

## Inspect options without downloading a model

```bash
cargo run --bin generate -- --help
cargo run --bin generate -- --version
```

Help and version return before loading model assets. Unknown arguments,
missing values, invalid integers, and invalid temperatures return argument-error
exit status 2. Asset and execution errors return failure. The default MLP
binary has its own `--help`; its flags differ from the generator's flags.

## Select the compute backend

`--backend cpu` selects the portable Rust reference. On macOS, the generator
defaults to `metal`; elsewhere it defaults to `cpu`. An explicit Metal request
fails on a non-macOS platform or when Metal initialization fails. It does not
silently select a different backend.

```bash
cargo run --release --bin generate -- --backend cpu --prompt "Once upon a time"
cargo run --release --bin generate -- --backend metal --prompt "Once upon a time"
```

See [platform troubleshooting](troubleshooting.md) for Metal requirements.
