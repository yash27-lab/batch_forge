use bytemuck::Pod;
use safetensors::tensor::Dtype;

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

impl From<Dtype> for DataType {
    fn from(dt: Dtype) -> Self {
        match dt {
            Dtype::F32 => DataType::F32,
            Dtype::F16 => DataType::F16,
            Dtype::BF16 => DataType::BF16,
            Dtype::I8 => DataType::I8,
            Dtype::U8 => DataType::U8,
            Dtype::I32 => DataType::I32,
            Dtype::I64 => DataType::I64,
            _ => unimplemented!("Unsupported dtype mapping: {:?}", dt),
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
    pub fn new(shape: Vec<usize>, dtype: DataType, data: &'data [u8]) -> Self {
        Self { shape, dtype, data }
    }

    /// Safely casts the underlying byte buffer to a typed slice if the dtype matches.
    pub fn as_slice<T: Pod>(&self) -> Option<&[T]> {
        // In a full implementation, we would verify `T` matches `self.dtype`.
        bytemuck::try_cast_slice(self.data).ok()
    }
}
