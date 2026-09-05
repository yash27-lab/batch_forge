# Contributing

Small, focused changes are easiest to review. Before opening a pull request, run the checks that match the code you changed:

```bash
cargo fmt --check
cargo check
cargo test --lib
```

If the formatting check reports changes, run `cargo fmt`, review the diff, then run the check again.

The formatting and library-test preflight is CPU-only and needs no model assets. Metal parity checks require Apple Silicon macOS, and the end-to-end GPT-2 check requires the downloaded model assets. See the [test command reference](docs/test-matrix.md) for the complete matrix and [troubleshooting guide](docs/troubleshooting.md) for setup notes.

## Reporting issues

Use the issue templates when reporting a bug or proposing an enhancement. For runtime bugs, include the backend, hardware, and smallest reproduction command.
