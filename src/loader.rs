use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use thiserror::Error;

use crate::tensor::{DataType, TensorView};

#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
}

/// Loads a Safetensors file via memory mapping, returning zero-copy tensor views.
pub fn load_safetensors<'a>(path: &Path) -> Result<HashMap<String, TensorView<'static>>, LoaderError> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    // We use Box::leak to keep the mmap object alive for the lifetime of the program,
    // which allows us to have a &'static [u8] view of the memory-mapped file.
    let mmap_ref: &'static [u8] = unsafe { std::mem::transmute(&mmap[..]) };
    std::mem::forget(mmap); // Prevent mmap from being dropped

    let st = SafeTensors::deserialize(mmap_ref)?;
    let mut tensors = HashMap::new();

    for name in st.names() {
        let view = st.tensor(name)?;
        let dtype = DataType::from(view.dtype());
        let shape = view.shape().to_vec();
        let data = view.data();

        tensors.insert(name.to_string(), TensorView::new(shape, dtype, data));
    }

    Ok(tensors)
}
