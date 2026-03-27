# batch_forge

batch_forge is a high-performance inference engine written in Rust, designed for large-scale Transformer, Diffusion, and State-Space Models (SSMs) authored in JAX and Equinox. It provides a bare-metal, zero-Python runtime for executing complex models on edge devices and consumer hardware using Metal and Vulkan compute kernels.

## Key Features

- **Asynchronous Request Management**: Built on `tokio`, featuring a non-blocking `RequestManager` for high-concurrency token generation and batching.
- **KV-Cache System**: Session-based Key-Value cache management in Metal's unified memory for efficient autoregressive generation without redundant re-computation.
- **Hybrid Metal Backend**: Combines Apple's highly optimized **Metal Performance Shaders (MPS)** for standard precision operations with custom **MSL (Metal Shading Language)** kernels for specialized tasks.
- **JAX/Equinox Integration**: Direct zero-copy loading of Equinox PyTrees via Safetensors, bypassing heavyweight XLA or TFLite runtimes.
- **Quantized Inference**: On-the-fly INT8 and INT4 dequantization kernels to minimize memory bandwidth and footprint on edge devices.

## Architecture

The engine is built on four core pillars:
1. **Async Runner**: A `tokio`-based server architecture that manages concurrent inference requests via mpsc channels and oneshot responses.
2. **Stateful Session Storage**: Persistent KV-cache buffers pre-allocated per request ID to support large-scale language generation.
3. **Zero-Copy Loader**: Uses memory-mapped I/O (`mmap`) to load weights instantly from Safetensors files without additional memory overhead.
4. **Optimized Dispatch**: Dynamically branches between Apple MPS for standard matmuls and custom hand-written kernels for fused attention and quantized states.

## Getting Started

### 1. Exporting Models from JAX/Equinox

Install the required Python utilities:
```bash
pip install jax equinox safetensors numpy
```

Export your Equinox model (PyTree) to the Safetensors format:
```bash
python python/export_eqx.py --out model.safetensors
```

### 2. Building the Rust Engine

Ensure you have the Rust toolchain installed. Build the project in release mode:
```bash
cargo build --release
```

### 3. Running Inference & Benchmarks

Run the engine to start the async inference loop and execute the built-in performance benchmarks:
```bash
./target/release/batch_forge
```

## Performance Comparison

batch_forge includes built-in benchmarking to compare custom compute kernels against native Apple Silicon hardware acceleration (MPS):
- **Custom Kernel**: Hand-written MSL shaders for specialized operations (INT8, Fused Attention).
- **Apple MPS**: Assembly-tuned matrix multiplication for standard FP32/FP16 precision.

## Supported Architectures

- **Transformers**: Autoregressive LLMs with KV-Cache support.
- **Vision Language Models (VLMs)**: High-throughput vision feature extraction.
- **State-Space Models (SSMs)**: Selective scan operations (e.g., Mamba).
- **Diffusion**: Fast UNet/DiT inference for image generation.

## Contributing

Contributions focusing on new compute kernels, quantization techniques (FP8/NF4), or additional hardware backends (Vulkan/WebGPU) are welcome. Please ensure all new kernels include numerical parity tests against JAX references.

## License

MIT OR Apache-2.0
