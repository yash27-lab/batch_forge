# Async inference engine guide

The engine serves MLP requests through a bounded Tokio channel. Create a
submitter while a Tokio runtime is active, then await each request result:

```rust
use std::sync::Arc;
use batch_forge::engine::spawn;
use batch_forge::model::{CpuBackend, Mlp, ModelError};
use batch_forge::tensor::Tensor;

async fn infer_once(model: Arc<Mlp>, input: Tensor) -> Result<Tensor, ModelError> {
    let client = spawn(Arc::new(CpuBackend), model, 8);
    client.infer(1, input).await
}
```

The input is owned by the submitted request. The request ID is used for
tracing; it is not a deduplication key or a guarantee of result persistence.

## Queue capacity and execution

`queue_depth` sets the bounded channel capacity; zero is normalized to one
slot. When the queue is full, sending waits for room. This bounds queued
requests, but not the number of caller futures retaining inputs while waiting
to submit.

The manager runs one forward pass at a time. The forward call is synchronous
inside its Tokio task, so CPU work occupies a runtime worker during inference.
Async channels do not fuse requests into a batch or make the numerical work
itself asynchronous.
