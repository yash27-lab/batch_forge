use bytemuck::Pod;
use safetensors::tensor::Dtype;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("Unsupported dtype mapping: {0:?}")]
    UnsupportedDtype(Dtype),
    #[error("Shape mismatch: expected {expected} bytes, found {found}")]
    ShapeMismatch { expected: usize, found: usize },
    #[error("Buffer overflow detected when computing tensor size")]
    BufferOverflow,
    #[error("Failed to allocate tensor buffer for {elements} elements")]
    AllocationFailed { elements: usize },
    #[error("Expected a {expected}-D tensor, found shape {shape:?}")]
    RankMismatch { expected: usize, shape: Vec<usize> },
    #[error("Tensor is dtype {0:?}, expected F32")]
    NotF32(DataType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F32,
    F16,
    BF16,
    I8,
    U8,
    I32,
    I64,
}

impl DataType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DataType::F32 | DataType::I32 => 4,
            DataType::F16 | DataType::BF16 => 2,
            DataType::I64 => 8,
            DataType::I8 | DataType::U8 => 1,
        }
    }
}

impl TryFrom<Dtype> for DataType {
    type Error = TensorError;

    fn try_from(dt: Dtype) -> Result<Self, Self::Error> {
        match dt {
            Dtype::F32 => Ok(DataType::F32),
            Dtype::F16 => Ok(DataType::F16),
            Dtype::BF16 => Ok(DataType::BF16),
            Dtype::I8 => Ok(DataType::I8),
            Dtype::U8 => Ok(DataType::U8),
            Dtype::I32 => Ok(DataType::I32),
            Dtype::I64 => Ok(DataType::I64),
            _ => Err(TensorError::UnsupportedDtype(dt)),
        }
    }
}

/// A view into a memory-mapped tensor buffer: the bytes are borrowed zero-copy
/// from the mapping, while the (tiny) shape vector is owned so the view can
/// outlive the transient parser handle it came from.
#[derive(Debug, Clone)]
pub struct TensorView<'data> {
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub data: &'data [u8],
}

impl<'data> TensorView<'data> {
    pub fn new(shape: &[usize], dtype: DataType, data: &'data [u8]) -> Result<Self, TensorError> {
        let expected_bytes = num_bytes(shape, dtype)?;
        if data.len() != expected_bytes {
            return Err(TensorError::ShapeMismatch {
                expected: expected_bytes,
                found: data.len(),
            });
        }
        Ok(Self {
            shape: shape.to_vec(),
            dtype,
            data,
        })
    }

    /// Number of elements described by the shape.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Safely casts the underlying byte buffer to a typed slice if the dtype matches.
    pub fn as_slice<T: Pod>(&self) -> Option<&[T]> {
        bytemuck::try_cast_slice(self.data).ok()
    }

    /// Materializes an owned f32 [`Tensor`], copying out of the mapped buffer.
    ///
    /// Reads little-endian f32 directly from the bytes, so it works even when the
    /// tensor's offset in the mmap is not 4-byte aligned (which a zero-copy
    /// `bytemuck` cast would reject). Only F32 source data is supported.
    pub fn to_tensor_f32(&self) -> Result<Tensor, TensorError> {
        if self.dtype != DataType::F32 {
            return Err(TensorError::NotF32(self.dtype));
        }
        if self.data.len() % 4 != 0 {
            return Err(TensorError::ShapeMismatch {
                expected: self.numel() * 4,
                found: self.data.len(),
            });
        }
        let data = self
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
        })
    }
}

fn checked_numel(shape: &[usize]) -> Result<usize, TensorError> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |elements, &dim| {
        elements.checked_mul(dim).ok_or(TensorError::BufferOverflow)
    })
}

fn num_bytes(shape: &[usize], dtype: DataType) -> Result<usize, TensorError> {
    checked_numel(shape)?
        .checked_mul(dtype.size_in_bytes())
        .ok_or(TensorError::BufferOverflow)
}

/// A simple owned, row-major f32 tensor used by the CPU reference ops and as the
/// common currency between backends.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Creates a tensor from data, validating that the element count matches the shape.
    /// Returns [`TensorError::BufferOverflow`] if the shape's element count overflows.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected = checked_numel(&shape)?;
        if data.len() != expected {
            return Err(TensorError::ShapeMismatch {
                expected,
                found: data.len(),
            });
        }
        Ok(Self { data, shape })
    }

    /// A zero-filled tensor of the given shape.
    ///
    /// # Panics
    ///
    /// Panics if the shape overflows or the backing buffer cannot be allocated.
    /// Use [`Tensor::try_zeros`] when those failures need to be handled.
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::try_zeros(shape).expect("failed to allocate zero-filled tensor")
    }

    /// Creates a zero-filled tensor, returning shape-overflow and allocation errors.
    pub fn try_zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        let elements = checked_numel(&shape)?;
        let mut data = Vec::new();
        data.try_reserve_exact(elements)
            .map_err(|_| TensorError::AllocationFailed { elements })?;
        data.resize(elements, 0.0);
        Ok(Self { data, shape })
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Interprets the tensor as 2-D, returning `(rows, cols)`.
    pub fn dims2(&self) -> Result<(usize, usize), TensorError> {
        match self.shape.as_slice() {
            [r, c] => Ok((*r, *c)),
            _ => Err(TensorError::RankMismatch {
                expected: 2,
                shape: self.shape.clone(),
            }),
        }
    }

    /// Maximum absolute element-wise difference against another tensor of equal shape.
    /// Useful for parity / tolerance assertions. Returns `f32::INFINITY` on shape or
    /// data-length mismatch, or when an element-wise difference is not finite.
    pub fn max_abs_diff(&self, other: &Tensor) -> f32 {
        if self.shape != other.shape || self.data.len() != other.data.len() {
            return f32::INFINITY;
        }
        let mut max_diff = 0.0_f32;
        for (a, b) in self.data.iter().zip(&other.data) {
            let diff = (a - b).abs();
            if !diff.is_finite() {
                return f32::INFINITY;
            }
            max_diff = max_diff.max(diff);
        }
        max_diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_tensor_view() {
        let data = vec![0u8; 8];
        let view = TensorView::new(&[2, 1], DataType::F32, &data);
        assert!(view.is_ok());
    }

    #[test]
    fn test_shape_mismatch() {
        let data = vec![0u8; 7]; // F32 requires multiple of 4
        let view = TensorView::new(&[2, 1], DataType::F32, &data);
        assert!(matches!(view, Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_buffer_overflow() {
        let data = vec![0u8; 8];
        let view = TensorView::new(&[usize::MAX, 2], DataType::F32, &data);
        assert!(matches!(view, Err(TensorError::BufferOverflow)));
    }

    #[test]
    fn test_view_to_tensor_roundtrip() {
        let floats = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(&floats);
        let view = TensorView::new(&[2, 2], DataType::F32, bytes).unwrap();
        let t = view.to_tensor_f32().unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_max_abs_diff() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![1.0, 2.5, 3.0], vec![3]).unwrap();
        assert!((a.max_abs_diff(&b) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_owned_tensor_rejects_shape_overflow() {
        assert!(matches!(
            Tensor::new(Vec::new(), vec![usize::MAX, 2]),
            Err(TensorError::BufferOverflow)
        ));
    }

    #[test]
    fn test_max_abs_diff_rejects_non_finite_values() {
        let finite = Tensor::new(vec![0.0], vec![1]).unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let invalid = Tensor::new(vec![value], vec![1]).unwrap();
            assert_eq!(invalid.max_abs_diff(&finite), f32::INFINITY);
            assert_eq!(finite.max_abs_diff(&invalid), f32::INFINITY);
            assert_eq!(invalid.max_abs_diff(&invalid), f32::INFINITY);
        }
    }

    #[test]
    fn test_max_abs_diff_rejects_mismatched_data_lengths() {
        let valid = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let truncated = Tensor {
            data: vec![1.0],
            shape: vec![2],
        };
        assert_eq!(valid.max_abs_diff(&truncated), f32::INFINITY);
        assert_eq!(truncated.max_abs_diff(&valid), f32::INFINITY);
    }

    #[test]
    fn test_fallible_zeros_rejects_shape_overflow() {
        assert!(matches!(
            Tensor::try_zeros(vec![usize::MAX, 2]),
            Err(TensorError::BufferOverflow)
        ));
    }

    #[test]
    fn test_fallible_zeros_preserves_empty_shape() {
        let tensor = Tensor::try_zeros(vec![2, 0, 3]).unwrap();
        assert_eq!(tensor.shape, vec![2, 0, 3]);
        assert!(tensor.data.is_empty());
    }

    #[test]
    fn test_fallible_zeros_initializes_values() {
        let tensor = Tensor::try_zeros(vec![2, 2]).unwrap();
        assert_eq!(tensor.data, vec![0.0; 4]);
    }

    #[test]
    fn zero_dimensions_short_circuit_overflow_in_any_order() {
        for shape in [vec![usize::MAX, 2, 0], vec![0, usize::MAX, 2]] {
            assert!(Tensor::try_zeros(shape).unwrap().data.is_empty());
        }
    }
}
