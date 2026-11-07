## Summary

Describe the change and why it is needed.

## Risks

Note any compatibility, performance, or behavior risks.

## Documentation

- [ ] Public API docs and user-facing guides updated when applicable

## Tests

- [ ] Relevant tests added or updated when behavior changes

## Validation

- [ ] `cargo fmt --check`
- [ ] `cargo check`
- [ ] `cargo clippy --all-targets`
- [ ] `cargo test --lib`
- [ ] Metal parity check run when applicable (`cargo test --test parity -- --nocapture`)
- [ ] End-to-end GPT-2 check run when applicable (`cargo test --test gpt2_e2e -- --nocapture`)
