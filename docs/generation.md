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

## Redirect generated text

Generation writes text to stdout and setup, backend, and timing information
to stderr. Redirect them independently:

```bash
cargo run --release --bin generate -- --backend cpu --prompt "Hello" > answer.txt 2> run.log
```

The output stream ends with a newline. Characters that span multiple tokenizer
tokens are emitted after their UTF-8 bytes are complete. Closing an output pipe
early stops the token loop and treats a broken pipe as normal termination;
other output errors report failure.

## Empty prompts and end-of-text

An empty CLI prompt is seeded internally with GPT-2's end-of-text token. This
provides the first input token required by the model without printing that seed.
The loop stops when it samples end-of-text and does not print the marker.

```bash
cargo run --release --bin generate -- --backend cpu --prompt "" --max-new 20
```

`--max-new` must be a positive integer. Timing statistics count sampled tokens,
including a sampled stop token, so the token count is not a character count.
