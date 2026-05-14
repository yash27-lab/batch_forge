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

## Scalars and empty dimensions

An empty shape `vec![]` describes a scalar containing one element:

```rust
let scalar = Tensor::new(vec![3.0], vec![])?;
```

A shape containing a zero dimension describes zero elements. For example,
`vec![2, 0, 3]` requires an empty data vector. A zero dimension makes the count
zero regardless of its position in the shape. These are shape semantics;
model layers and reduction operators can impose additional positive-width
requirements.

## Fallible zero allocation

Use `try_zeros` when dimensions come from a caller or checkpoint:

```rust
fn empty_batch() -> Result<Tensor, TensorError> {
    Tensor::try_zeros(vec![0, 768])
}
```

An overflowing element count returns `TensorError::BufferOverflow`. A failed
buffer reservation returns `TensorError::AllocationFailed`. The convenience
`zeros` constructor panics on those failures. The fallible API handles errors
reported by buffer reservation; it is not a process-wide memory limit.
