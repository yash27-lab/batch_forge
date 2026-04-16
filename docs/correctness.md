# batch_forge Correctness Guarantees

Inference engines require absolute trust. `batch_forge` provides strict correctness guarantees and tests against numerical parity with reference implementations (JAX/PyTorch).

## Tolerance Bounds

We define the following standard error bounds for our custom Metal shaders and MPS dispatches:

| Precision / Mode | Absolute Tolerance (`atol`) | Relative Tolerance (`rtol`) | Notes |
|------------------|-----------------------------|-----------------------------|-------|
| **FP32** | 1e-5 | 1e-4 | Standard IEEE 754 precision, verified against CPU ground truth. |
| **FP16** | 1e-3 | 1e-3 | Evaluated dynamically based on dynamic range of activation. |
| **INT8 (W8A16)** | 5e-2 | 1e-2 | Dequantization introduces quantization noise; validated on weight distribution. |
| **INT4 (W4A16)** | 1e-1 | 5e-2 | Aggressive quantization with group-wise scaling. |

## Per-Op Parity Status

Every operator in `batch_forge` is backed by automated tests comparing output against CPU reference logic.

| Operator | Precision Support | Parity Test Status | Hardware |
|----------|-------------------|--------------------|----------|
| **MatMul** | FP32, FP16, INT8 | ✅ Passing | Apple MPS / MSL |
| **Attention** | FP32, FP16 | ✅ Passing (Flash/Standard) | Custom MSL |
| **Dequantize** | INT8, INT4 | ✅ Passing | Custom MSL |
| **Scan (SSM)** | FP32 | 🚧 In Progress | Custom MSL |
| **LayerNorm** | FP32, FP16 | ✅ Passing | Custom MSL |
| **RoPE** | FP32, FP16 | ✅ Passing | Custom MSL |

## Automated Verification

All tensor operations and kernels undergo correctness verification via:
1. **CPU vs Metal Tests**: Assertions ensure that custom shaders output the exact numerical results as their pure-Rust CPU equivalents (within tolerance bounds).
2. **SafeTensors Validation**: The loader validates expected shapes and data types strictly to prevent misinterpretation of binary data.
3. **KV-Cache Regression Tests**: Ensures cache size constraints and cyclic buffer behaviors do not corrupt generation sequences.

## How to Run Tests

Run the full correctness suite:
```bash
cargo test --release --lib
```
