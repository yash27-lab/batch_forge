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

## Validate after changing public fields

`shape` and `data` are public. If a caller edits either after construction,
call `tensor.validate()` before using that structure:

```rust
let mut tensor = Tensor::new(vec![1.0, 2.0], vec![2])?;
tensor.shape = vec![1, 2];
tensor.validate()?;
```

Validation checks structural consistency and shape overflow. It does not
require every F32 value to be finite. MLP inference revalidates its public
layer tensors and the supplied input before forwarding.

## Interpret parity differences

`max_abs_diff` returns the largest absolute difference for valid, equal-shaped
buffers. It returns infinity for shape mismatch, invalid buffers, or a
non-finite element difference; even two equally truncated tensors fail.

```rust
let difference = actual.max_abs_diff(&reference);
let passes = difference.is_finite() && difference <= 1e-3;
```

Choose a tolerance appropriate to the operation and test. A matching numeric
buffer is evidence about that comparison, not proof that a whole model or
asset set has been verified. See [correctness](correctness.md) for the test
boundaries used by this repository.
