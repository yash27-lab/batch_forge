use tracing::{info, error};
use tokio::sync::{mpsc, oneshot};
use std::sync::Arc;
use std::path::PathBuf;

mod tensor;
mod loader;

#[cfg(target_os = "macos")]
mod metal_backend;

#[cfg(target_os = "macos")]
mod kv_cache;

/// Represents a request for model inference.
#[derive(Debug)]
struct InferenceRequest {
    request_id: u64,
    input_data: Vec<f32>,
    response_tx: oneshot::Sender<Vec<f32>>,
}

/// Manages the asynchronous request queue and dynamic batching.
struct RequestManager {
    backend: Arc<metal_backend::MetalBackend>,
    kv_storage: kv_cache::KVStorage,
    request_rx: mpsc::Receiver<InferenceRequest>,
}

impl RequestManager {
    pub fn new(backend: Arc<metal_backend::MetalBackend>, rx: mpsc::Receiver<InferenceRequest>) -> Self {
        let kv_storage = kv_cache::KVStorage::new(backend.device.clone(), 1024, 64);
        Self { backend, kv_storage, request_rx: rx }
    }

    pub async fn run(mut self) {
        info!("RequestManager: Listening for incoming requests...");
        
        while let Some(req) = self.request_rx.recv().await {
            let backend = Arc::clone(&self.backend);
            let kv_cache = self.kv_storage.get_or_create(req.request_id);
            
            // For testing: dummy KV update and attention
            let m = 1; // 1 token at a time for generation
            let d = 64; 
            
            let new_k_data = vec![0.1f32; (m * d) as usize];
            let new_v_data = vec![0.2f32; (m * d) as usize];
            let q_data = req.input_data; // Assume input is Q for this test
            
            let buf_new_k = backend.create_buffer(&new_k_data);
            let buf_new_v = backend.create_buffer(&new_v_data);
            let buf_q = backend.create_buffer(&q_data);
            let buf_o = backend.create_buffer_uninitialized::<f32>((m * d) as usize);

            // 1. Update KV Cache
            backend.update_kv_cache(
                &buf_new_k, 
                &buf_new_v, 
                &kv_cache.k_buffer, 
                &kv_cache.v_buffer, 
                m as u32, 
                kv_cache.current_len as u32, 
                d as u32
            );
            kv_cache.current_len += m;

            // 2. Perform KV Attention
            backend.kv_attention(
                &buf_q, 
                &kv_cache.k_buffer, 
                &kv_cache.v_buffer, 
                &buf_o, 
                m as u32, 
                kv_cache.current_len as u32, 
                d as u32
            );

            let ptr = buf_o.contents() as *const f32;
            let mut o_data = vec![0.0f32; (m * d) as usize];
            unsafe { std::ptr::copy_nonoverlapping(ptr, o_data.as_mut_ptr(), o_data.len()); }
            
            let _ = req.response_tx.send(o_data);
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    info!("Starting batch_forge Async Engine");

    // Phase 1: Load SafeTensors (if available)
    let model_path = PathBuf::from("model.safetensors");
    if model_path.exists() {
        match loader::load_safetensors(&model_path) {
            Ok(tensors) => {
                info!("Successfully loaded {} tensors from Safetensors.", tensors.len());
            }
            Err(e) => {
                error!("Failed to load model: {}", e);
            }
        }
    } else {
        info!("No model.safetensors found. Run python export_eqx.py to generate it.");
    }

    // Initialize backend
    let shader_source = include_str!("shaders/compute.metal");
    let backend = Arc::new(metal_backend::MetalBackend::new(shader_source).expect("Failed to init Metal"));

    // Set up request queue
    let (tx, rx) = mpsc::channel(100);
    let manager = RequestManager::new(Arc::clone(&backend), rx);
    
    // Spawn the manager in its own task
    tokio::spawn(async move {
        manager.run().await;
    });

    // Simulate an autoregressive generation loop for a single request
    let tx_clone = tx.clone();
    let handle = tokio::spawn(async move {
        let request_id = 42;
        info!("Starting Autoregressive Generation for Request {}", request_id);

        for step in 0..5 {
            let (resp_tx, resp_rx) = oneshot::channel();
            // In a real model, Q would be derived from the previous step's output
            let input_q = vec![0.5f32; 64]; 
            
            tx_clone.send(InferenceRequest {
                request_id,
                input_data: input_q,
                response_tx: resp_tx,
            }).await.unwrap();

            let result = resp_rx.await.unwrap();
            info!("Step {}: Generation output (first 5 elements): {:?}", step, &result[0..5]);
        }
    });

    handle.await.unwrap();
}
