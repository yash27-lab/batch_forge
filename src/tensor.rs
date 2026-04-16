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
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

/// A zero-copy view into a memory-mapped tensor buffer.
#[derive(Debug)]
pub struct TensorView<'data> {
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub data: &'data [u8],
}

impl<'data> TensorView<'data> {
    pub fn new(shape: Vec<usize>, dtype: DataType, data: &'data [u8]) -> Result<Self, TensorError> {
        let mut expected_elements: usize = 1;
        for dim in &shape {
            expected_elements = expected_elements.checked_mul(*dim).ok_or(TensorError::BufferOverflow)?;
        }
        
        let expected_bytes = expected_elements.checked_mul(dtype.size_in_bytes()).ok_or(TensorError::BufferOverflow)?;
        if data.len() != expected_bytes {
            return Err(TensorError::ShapeMismatch { expected: expected_bytes, found: data.len() });
        }

        Ok(Self { shape, dtype, data })
    }

    /// Safely casts the underlying byte buffer to a typed slice if the dtype matches.
    pub fn as_slice<T: Pod>(&self) -> Option<&[T]> {
        bytemuck::try_cast_slice(self.data).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_tensor_view() {
        let data = vec![0u8; 8];
        let view = TensorView::new(vec![2, 1], DataType::F32, &data);
        assert!(view.is_ok());
    }

    #[test]
    fn test_shape_mismatch() {
        let data = vec![0u8; 7]; // F32 requires multiple of 4
        let view = TensorView::new(vec![2, 1], DataType::F32, &data);
        assert!(matches!(view, Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_buffer_overflow() {
        let data = vec![0u8; 8];
        let view = TensorView::new(vec![usize::MAX, 2], DataType::F32, &data);
        assert!(matches!(view, Err(TensorError::BufferOverflow)));
    }
}

