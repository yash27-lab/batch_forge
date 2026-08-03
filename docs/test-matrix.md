# Test command reference

Use the smallest relevant check while iterating:

| Goal | Command | Platform |
| --- | --- | --- |
| Validate CPU reference operations | `cargo test --lib` | Any supported Rust platform |
| Check Metal-to-CPU kernel parity | `cargo test --test parity -- --nocapture` | Apple Silicon macOS |
| Check end-to-end GPT-2 predictions | `cargo test --test gpt2_e2e -- --nocapture` | Requires downloaded GPT-2 assets |

Before the end-to-end check, download the model weights and tokenizer:

```bash
python python/fetch_gpt2.py
```

For platform and asset setup notes, see the [troubleshooting guide](troubleshooting.md).
