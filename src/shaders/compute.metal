#include <metal_stdlib>
using namespace metal;

// GELU (tanh approximation), matching ops::gelu / jax.nn.gelu(approximate=True).
inline float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// ---------------------------------------------------------------------------
// Dense matmul: C[M,N] = A[M,K] * B[K,N]  (row-major, naive one-thread-per-output)
// ---------------------------------------------------------------------------
kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (uint i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Linear (equinox.nn.Linear): Y[N,Out] = X[N,In] * W[Out,In]^T + B[Out]
// Avoids an explicit weight transpose by indexing W row-major as [Out,In].
// ---------------------------------------------------------------------------
kernel void linear(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* Bias [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& Rows [[buffer(4)]],
    constant uint& In [[buffer(5)]],
    constant uint& Out [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint o = gid.x;
    if (row < Rows && o < Out) {
        float acc = Bias[o];
        for (uint i = 0; i < In; ++i) {
            acc += X[row * In + i] * W[o * In + i];
        }
        Y[row * Out + o] = acc;
    }
}

// ---------------------------------------------------------------------------
// Weight-only INT8 matmul: C = (A_int8 * per-row scale) * B
// ---------------------------------------------------------------------------
kernel void quant_matmul(
    device const char* A_int8 [[buffer(0)]],
    device const float* B_f32 [[buffer(1)]],
    device float* C_f32 [[buffer(2)]],
    device const float* A_scales [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        float scale = A_scales[row];
        for (uint i = 0; i < K; ++i) {
            float a_val = (float)A_int8[row * K + i] * scale;
            sum += a_val * B_f32[i * N + col];
        }
        C_f32[row * N + col] = sum;
    }
}

// Elementwise GELU.
kernel void gelu_forward(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < N) {
        Y[gid] = gelu_approx(X[gid]);
    }
}

// ---------------------------------------------------------------------------
// Single-head attention with KV-cache.
//
// One thread per query row. The previous version recomputed the full Q·K dot
// product inside the output-dimension loop, making it O(M·S·D^2). This computes
// each score once per key, so the cost is O(M·S·D). Causal masking restricts
// query row i to keys j <= i + QOffset (QOffset = absolute position of the first
// new query, i.e. the cache length before this step).
// ---------------------------------------------------------------------------
kernel void kv_attention(
    device const float* Q [[buffer(0)]],       // M x D
    device const float* K_cache [[buffer(1)]], // CurSeqLen x D
    device const float* V_cache [[buffer(2)]], // CurSeqLen x D
    device float* O [[buffer(3)]],             // M x D
    constant uint& M [[buffer(4)]],
    constant uint& CurSeqLen [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& Causal [[buffer(7)]],
    constant uint& QOffset [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint q_idx = gid;
    if (q_idx >= M) return;

    uint limit = CurSeqLen;
    if (Causal != 0u) {
        uint c = q_idx + QOffset + 1u;
        limit = (c < CurSeqLen) ? c : CurSeqLen;
    }

    // Output starts at zero; if nothing is visible it stays zero.
    for (uint d = 0; d < D; ++d) O[q_idx * D + d] = 0.0f;
    if (limit == 0u) return;

    float inv_sqrt_d = rsqrt((float)D);

    // Pass 1: running max for numerical stability.
    float max_score = -INFINITY;
    for (uint k = 0; k < limit; ++k) {
        float s = 0.0f;
        for (uint d = 0; d < D; ++d) s += Q[q_idx * D + d] * K_cache[k * D + d];
        s *= inv_sqrt_d;
        max_score = max(max_score, s);
    }

    // Pass 2: denominator.
    float sum_exp = 0.0f;
    for (uint k = 0; k < limit; ++k) {
        float s = 0.0f;
        for (uint d = 0; d < D; ++d) s += Q[q_idx * D + d] * K_cache[k * D + d];
        s *= inv_sqrt_d;
        sum_exp += exp(s - max_score);
    }

    // Pass 3: weighted sum of V (score computed once per key, not per dim).
    for (uint k = 0; k < limit; ++k) {
        float s = 0.0f;
        for (uint d = 0; d < D; ++d) s += Q[q_idx * D + d] * K_cache[k * D + d];
        s *= inv_sqrt_d;
        float w = exp(s - max_score) / sum_exp;
        for (uint d = 0; d < D; ++d) O[q_idx * D + d] += w * V_cache[k * D + d];
    }
}

// Append M new tokens to the KV cache at row `Offset`.
kernel void update_kv_cache(
    device const float* NewK [[buffer(0)]],   // M x D
    device const float* NewV [[buffer(1)]],   // M x D
    device float* K_cache [[buffer(2)]],      // MaxSeq x D
    device float* V_cache [[buffer(3)]],      // MaxSeq x D
    constant uint& M [[buffer(4)]],
    constant uint& Offset [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint tok_idx = gid.x;
    uint d_idx = gid.y;
    if (tok_idx < M && d_idx < D) {
        uint cache_pos = (tok_idx + Offset) * D + d_idx;
        uint input_pos = tok_idx * D + d_idx;
        K_cache[cache_pos] = NewK[input_pos];
        V_cache[cache_pos] = NewV[input_pos];
    }
}

// Row-wise LayerNorm over the last dimension.
kernel void layernorm(
    device const float* X [[buffer(0)]],
    device const float* Gamma [[buffer(1)]],
    device const float* Beta [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& Rows [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant float& Eps [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint r = gid;
    if (r >= Rows) return;
    float mean = 0.0f;
    for (uint c = 0; c < D; ++c) mean += X[r * D + c];
    mean /= (float)D;
    float var = 0.0f;
    for (uint c = 0; c < D; ++c) {
        float diff = X[r * D + c] - mean;
        var += diff * diff;
    }
    var /= (float)D;
    float inv_std = rsqrt(var + Eps);
    for (uint c = 0; c < D; ++c) {
        Y[r * D + c] = (X[r * D + c] - mean) * inv_std * Gamma[c] + Beta[c];
    }
}

// Row-wise RMSNorm over the last dimension.
kernel void rmsnorm(
    device const float* X [[buffer(0)]],
    device const float* Gamma [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& Rows [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant float& Eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint r = gid;
    if (r >= Rows) return;
    float ms = 0.0f;
    for (uint c = 0; c < D; ++c) ms += X[r * D + c] * X[r * D + c];
    ms /= (float)D;
    float inv = rsqrt(ms + Eps);
    for (uint c = 0; c < D; ++c) {
        Y[r * D + c] = X[r * D + c] * inv * Gamma[c];
    }
}

// Rotary position embedding (rotate-half / GPT-NeoX convention), applied in place.
kernel void rope(
    device float* X [[buffer(0)]],
    device const uint* Positions [[buffer(1)]],
    constant uint& Rows [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    constant float& Theta [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint r = gid;
    if (r >= Rows) return;
    uint half_d = D / 2;
    float pos = (float)Positions[r];
    for (uint i = 0; i < half_d; ++i) {
        float inv_freq = pow(Theta, -2.0f * (float)i / (float)D);
        float angle = pos * inv_freq;
        float s = sin(angle);
        float c = cos(angle);
        float x1 = X[r * D + i];
        float x2 = X[r * D + i + half_d];
        X[r * D + i] = x1 * c - x2 * s;
        X[r * D + i + half_d] = x2 * c + x1 * s;
    }
}
