//! Asynchronous inference engine.
//!
//! A `tokio` task owns the model and backend and serves inference requests off
//! an `mpsc` queue, replying on a per-request `oneshot` channel. This is the
//! non-blocking request/response architecture; true continuous batching (fusing
//! queued requests into one dispatch) is future work — see the README roadmap.
//!
//! The engine is backend-agnostic (`Arc<dyn Backend>`), so it runs on the CPU
//! reference or the Metal backend without changes.

use std::sync::Arc;

use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info};

use crate::model::{Backend, Mlp, ModelError};
use crate::tensor::Tensor;

/// A unit of work: an input activation plus a channel to return the result on.
pub struct InferenceRequest {
    pub request_id: u64,
    pub input: Tensor,
    pub response_tx: oneshot::Sender<Result<Tensor, ModelError>>,
}

/// Owns the model + backend and serves requests from the queue.
pub struct RequestManager {
    backend: Arc<dyn Backend + Send + Sync>,
    model: Arc<Mlp>,
    request_rx: mpsc::Receiver<InferenceRequest>,
}

impl RequestManager {
    pub fn new(
        backend: Arc<dyn Backend + Send + Sync>,
        model: Arc<Mlp>,
        request_rx: mpsc::Receiver<InferenceRequest>,
    ) -> Self {
        Self {
            backend,
            model,
            request_rx,
        }
    }

    /// Drains the queue until all senders are dropped, running one forward pass
    /// per request on the configured backend.
    pub async fn run(mut self) {
        info!(
            "RequestManager listening (backend: {})",
            self.backend.name()
        );
        let mut served = 0u64;
        while let Some(req) = self.request_rx.recv().await {
            let out = self.model.forward(&*self.backend, &req.input);
            debug!(request_id = req.request_id, "served");
            // The receiver may have gone away; that's fine.
            let _ = req.response_tx.send(out);
            served += 1;
        }
        info!("RequestManager shutting down after {served} requests");
    }
}

/// Convenience: spin up a manager on the current runtime and return a handle for
/// submitting requests. The manager stops when the returned `Submitter` (and all
/// its clones) are dropped.
pub fn spawn(
    backend: Arc<dyn Backend + Send + Sync>,
    model: Arc<Mlp>,
    queue_depth: usize,
) -> Submitter {
    let (tx, rx) = mpsc::channel(queue_depth);
    let manager = RequestManager::new(backend, model, rx);
    tokio::spawn(manager.run());
    Submitter { tx }
}

/// Cloneable client handle for submitting inference requests to the engine.
#[derive(Clone)]
pub struct Submitter {
    tx: mpsc::Sender<InferenceRequest>,
}

impl Submitter {
    /// Submits one request and awaits its result.
    pub async fn infer(&self, request_id: u64, input: Tensor) -> Result<Tensor, ModelError> {
        let (response_tx, response_rx) = oneshot::channel();
        self.tx
            .send(InferenceRequest {
                request_id,
                input,
                response_tx,
            })
            .await
            .expect("engine receiver dropped");
        response_rx.await.expect("engine dropped response channel")
    }
}
