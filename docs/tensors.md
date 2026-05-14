# Tensor API guide

`Tensor` is an owned row-major F32 buffer. Its shape records dimensions; its
data vector records elements. `Tensor::new` checks that these agree:

```rust
use batch_forge::tensor::{Tensor, TensorError};

fn matrix() -> Result<Tensor, TensorError> {
    Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
}
```

`dims2()` checks rank and returns `(rows, cols)`. A 2×2 tensor contains four
F32 elements; byte-backed views count bytes separately.
