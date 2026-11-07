//! Safetensors loading over `mmap`.
//!
//! [`SafeModel`] owns the memory map for the lifetime of the handle, so tensor
//! views borrow from a stable backing store. This replaces the previous
//! `transmute` + `mem::forget` approach, which laundered a borrowed slice into
//! `&'static` and leaked the mapping on every load.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use safetensors::SafeTensors;
use thiserror::Error;

use crate::tensor::{DataType, Tensor, TensorError, TensorView};

#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("Tensor error: {0}")]
    Tensor(#[from] TensorError),
}

/// A memory-mapped safetensors checkpoint. Holding this handle keeps the mapping
/// alive; tensor views and owned tensors are produced on demand from it.
pub struct SafeModel {
    mmap: Mmap,
}

impl SafeModel {
    /// Opens and memory-maps a safetensors file. The header is not parsed until
    /// [`SafeModel::with_tensors`] or [`SafeModel::load_f32`] is called.
    pub fn open(path: &Path) -> Result<Self, LoaderError> {
        let file = File::open(path)?;
        // SAFETY: the file is not mutated for the lifetime of the mapping; the
        // mapping is owned by `self` and dropped (unmapped) with it.
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap })
    }

    /// Parses the checkpoint and invokes `f` with zero-copy views that borrow
    /// from the mapping. The scoped-closure shape keeps the views' lifetime tied
    /// to the borrow of `self`, which is what makes this sound *and* copy-free
    /// (e.g. uploading weights straight to a GPU buffer without a heap copy).
    pub fn with_tensors<R>(
        &self,
        f: impl FnOnce(&HashMap<String, TensorView<'_>>) -> R,
    ) -> Result<R, LoaderError> {
        let st = SafeTensors::deserialize(&self.mmap)?;
        let mut views = HashMap::new();
        for name in st.names() {
            let raw = st.tensor(name)?;
            let dtype = DataType::try_from(raw.dtype())?;
            let view = TensorView::new(raw.shape(), dtype, raw.data())?;
            views.insert(name.to_string(), view);
        }
        Ok(f(&views))
    }

    /// Loads every F32 tensor into owned [`Tensor`]s. Non-F32 tensors cause a
    /// [`TensorError::NotF32`]; the demo model is exported entirely in F32.
    pub fn load_f32(&self) -> Result<HashMap<String, Tensor>, LoaderError> {
        self.with_tensors(|views| {
            let mut out = HashMap::with_capacity(views.len());
            for (name, view) in views {
                out.insert(name.clone(), view.to_tensor_f32()?);
            }
            Ok(out)
        })?
    }
}

/// Convenience: open `path` and load all F32 tensors as owned [`Tensor`]s.
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, Tensor>, LoaderError> {
    SafeModel::open(path)?.load_f32()
}
