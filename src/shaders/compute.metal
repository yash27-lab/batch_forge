#include <metal_stdlib>
using namespace metal;

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
        float sum = 0.0;
        for (uint i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

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
        float sum = 0.0;
        float scale = A_scales[row];
        for (uint i = 0; i < K; ++i) {
            float a_val = (float)A_int8[row * K + i] * scale;
            sum += a_val * B_f32[i * N + col];
        }
        C_f32[row * N + col] = sum;
    }
}

// Fused Attention with KV-Cache support
kernel void kv_attention(
    device const float* Q [[buffer(0)]],      // M x D (Queries for current step)
    device const float* K_cache [[buffer(1)]], // MaxSeq x D (Keys Cache)
    device const float* V_cache [[buffer(2)]], // MaxSeq x D (Values Cache)
    device float* O [[buffer(3)]],             // M x D (Output)
    constant uint& M [[buffer(4)]],            // Current query length (usually 1 for generation)
    constant uint& CurSeqLen [[buffer(5)]],    // Current total sequence length including new tokens
    constant uint& D [[buffer(6)]],            // Head Dimension
    uint gid [[thread_position_in_grid]]
) {
    uint q_idx = gid;
    if (q_idx >= M) return;

    float max_score = -1e9;
    
    // Iterate over the cached keys up to the current sequence length
    for (uint k_idx = 0; k_idx < CurSeqLen; ++k_idx) {
        float score = 0.0;
        for (uint d = 0; d < D; ++d) {
            score += Q[q_idx * D + d] * K_cache[k_idx * D + d];
        }
        score /= sqrt((float)D);
        max_score = max(max_score, score);
    }
    
    float sum_exp = 0.0;
    for (uint k_idx = 0; k_idx < CurSeqLen; ++k_idx) {
        float score = 0.0;
        for (uint d = 0; d < D; ++d) {
            score += Q[q_idx * D + d] * K_cache[k_idx * D + d];
        }
        score /= sqrt((float)D);
        sum_exp += exp(score - max_score);
    }
    
    for (uint d = 0; d < D; ++d) {
        float out_val = 0.0;
        for (uint k_idx = 0; k_idx < CurSeqLen; ++k_idx) {
            float score = 0.0;
            for (uint d_inner = 0; d_inner < D; ++d_inner) {
                score += Q[q_idx * D + d_inner] * K_cache[k_idx * D + d_inner];
            }
            score /= sqrt((float)D);
            float weight = exp(score - max_score) / sum_exp;
            out_val += weight * V_cache[k_idx * D + d];
        }
        O[q_idx * D + d] = out_val;
    }
}

// Helper kernel to update KV-Cache with new tokens
kernel void update_kv_cache(
    device const float* NewK [[buffer(0)]],   // M x D
    device const float* NewV [[buffer(1)]],   // M x D
    device float* K_cache [[buffer(2)]],      // MaxSeq x D
    device float* V_cache [[buffer(3)]],      // MaxSeq x D
    constant uint& M [[buffer(4)]],           // New tokens length
    constant uint& Offset [[buffer(5)]],      // Starting position in cache
    constant uint& D [[buffer(6)]],           // Head Dimension
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
