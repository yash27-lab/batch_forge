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
