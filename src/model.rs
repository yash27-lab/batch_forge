//! Model definitions and the backend abstraction used to run them.
//!
//! [`Mlp`] mirrors `python/export_eqx.py::SimpleMLP`. It is generic over a
//! [`Backend`], so the exact same forward pass runs on the portable
//! [`CpuBackend`] reference and on the Metal backend, which is how the
//! end-to-end test asserts CPU/GPU agreement.

use std::collections::HashMap;
use thiserror::Error;

use crate::ops;
use crate::tensor::Tensor;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("missing tensor '{0}' in checkpoint")]
    MissingTensor(String),
    #[error("tensor '{name}' has unexpected shape {shape:?}")]
    BadShape { name: String, shape: Vec<usize> },
    #[error("no layers found in checkpoint")]
    NoLayers,
    #[error("dimension mismatch: layer expects input width {expected}, got {got}")]
    WidthMismatch { expected: usize, got: usize },
}

/// The minimal compute surface a feed-forward model needs from a backend.
///
/// Keeping this trait tiny is deliberate: a new backend (CPU, Metal, and later
/// Vulkan/WebGPU) only has to implement these two ops to run the whole model.
pub trait Backend {
    /// `y = x · Wᵀ + b`, with `x`[n×in], `w`[out×in], `b`[out] → `[n×out]`.
    fn linear(&self, x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor;
    /// Element-wise GELU (tanh approximation).
    fn gelu(&self, x: &Tensor) -> Tensor;
    /// Human-readable backend name, used in logs and the demo output.
    fn name(&self) -> &'static str;
}

/// Portable, always-available reference backend. Defines correct behavior.
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn linear(&self, x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
        let (n, in_f) = x.dims2().expect("linear input must be 2-D");
        let (out_f, in_w) = w.dims2().expect("weight must be 2-D");
        assert_eq!(in_f, in_w, "linear in-features mismatch");
        let data = ops::linear(&x.data, &w.data, &b.data, n, in_f, out_f);
        Tensor::new(data, vec![n, out_f]).expect("linear output shape is consistent")
    }

    fn gelu(&self, x: &Tensor) -> Tensor {
        let mut out = x.clone();
        ops::gelu_inplace(&mut out.data);
        out
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}

/// A stack of `Linear` layers with GELU between them (no activation after the
/// final layer) — the structure exported by `SimpleMLP`.
pub struct Mlp {
    /// `(weight[out×in], bias[out])` per layer, in execution order.
    pub layers: Vec<(Tensor, Tensor)>,
}

impl Mlp {
    /// Builds an MLP from a name→tensor map using the `layers.{i}.weight` /
    /// `layers.{i}.bias` naming emitted by the exporter.
    pub fn from_tensors(map: &HashMap<String, Tensor>) -> Result<Self, ModelError> {
        let mut layers = Vec::new();
        let mut i = 0;
        loop {
            let w_name = format!("layers.{i}.weight");
            let b_name = format!("layers.{i}.bias");
            let Some(w) = map.get(&w_name) else { break };
            let b = map
                .get(&b_name)
                .ok_or_else(|| ModelError::MissingTensor(b_name.clone()))?;
            if w.shape.len() != 2 {
                return Err(ModelError::BadShape {
                    name: w_name,
                    shape: w.shape.clone(),
                });
            }
            if b.shape.len() != 1 || b.shape[0] != w.shape[0] {
                return Err(ModelError::BadShape {
                    name: b_name,
                    shape: b.shape.clone(),
                });
            }
            layers.push((w.clone(), b.clone()));
            i += 1;
        }
        if layers.is_empty() {
            return Err(ModelError::NoLayers);
        }
        Ok(Self { layers })
    }

    /// Input feature width expected by the first layer.
    pub fn in_features(&self) -> usize {
        self.layers[0].0.shape[1]
    }

    /// Output feature width produced by the last layer.
    pub fn out_features(&self) -> usize {
        self.layers.last().unwrap().0.shape[0]
    }

    /// Runs the forward pass on the given backend. `x` is `[n × in_features]`.
    ///
    /// `?Sized` so it accepts both concrete backends and `&dyn Backend` (used by
    /// the async engine, which holds an `Arc<dyn Backend>`).
    pub fn forward<B: Backend + ?Sized>(
        &self,
        backend: &B,
        x: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let (_, in_f) = x.dims2().map_err(|_| ModelError::WidthMismatch {
            expected: self.in_features(),
            got: 0,
        })?;
        if in_f != self.in_features() {
            return Err(ModelError::WidthMismatch {
                expected: self.in_features(),
                got: in_f,
            });
        }
        let mut h = x.clone();
        let last = self.layers.len() - 1;
        for (idx, (w, b)) in self.layers.iter().enumerate() {
            h = backend.linear(&h, w, b);
            if idx != last {
                h = backend.gelu(&h);
            }
        }
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model() -> Mlp {
        // 2 -> 2 -> 2, identity-ish weights.
        let mut map = HashMap::new();
        map.insert(
            "layers.0.weight".into(),
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap(),
        );
        map.insert(
            "layers.0.bias".into(),
            Tensor::new(vec![0.0, 0.0], vec![2]).unwrap(),
        );
        map.insert(
            "layers.1.weight".into(),
            Tensor::new(vec![2.0, 0.0, 0.0, 2.0], vec![2, 2]).unwrap(),
        );
        map.insert(
            "layers.1.bias".into(),
            Tensor::new(vec![1.0, 1.0], vec![2]).unwrap(),
        );
        Mlp::from_tensors(&map).unwrap()
    }

    #[test]
    fn builds_layers_in_order() {
        let m = tiny_model();
        assert_eq!(m.layers.len(), 2);
        assert_eq!(m.in_features(), 2);
        assert_eq!(m.out_features(), 2);
    }

    #[test]
    fn cpu_forward_is_correct() {
        let m = tiny_model();
        let x = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        // layer0 (identity) -> [1,-1]; gelu -> [gelu(1), gelu(-1)];
        // layer1 (2x + 1) -> [2*gelu(1)+1, 2*gelu(-1)+1]
        let y = m.forward(&CpuBackend, &x).unwrap();
        let expected0 = 2.0 * ops::gelu(1.0) + 1.0;
        let expected1 = 2.0 * ops::gelu(-1.0) + 1.0;
        assert!((y.data[0] - expected0).abs() < 1e-5);
        assert!((y.data[1] - expected1).abs() < 1e-5);
    }

    #[test]
    fn rejects_wrong_input_width() {
        let m = tiny_model();
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        assert!(matches!(
            m.forward(&CpuBackend, &x),
            Err(ModelError::WidthMismatch { .. })
        ));
    }
}
