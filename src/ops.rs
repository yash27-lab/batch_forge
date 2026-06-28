//! Portable, dependency-free CPU reference implementations of every operator.
//!
//! These functions define the *ground-truth* numerics for the engine. The Metal
//! kernels in `metal_backend` are validated against them by the parity tests in
//! `tests/parity.rs`, and the documented tolerance bounds in `docs/correctness.md`
//! refer to the maximum observed deviation between these and the GPU path.
//!
//! Conventions:
//! * All matrices are row-major.
//! * `gelu` uses the tanh approximation, matching `jax.nn.gelu(approximate=True)`.
//! * `rope` uses the rotate-half (GPT-NeoX / HF) convention.

/// Standard matrix multiply: `A`[m×k] · `B`[k×n] → `C`[m×n] (row-major).
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A has wrong length");
    assert_eq!(b.len(), k * n, "B has wrong length");
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for i in 0..k {
            let a_ik = a[row * k + i];
            // Hoisting a_ik and walking B contiguously keeps the reference
            // cache-friendly without changing the (well-defined) summation order.
            let b_row = &b[i * n..i * n + n];
            let c_row = &mut c[row * n..row * n + n];
            for col in 0..n {
                c_row[col] += a_ik * b_row[col];
            }
        }
    }
    c
}

/// Affine layer matching `equinox.nn.Linear`: `y = x · Wᵀ + b`.
///
/// `x`[n×in], `w`[out×in], `b`[out] → `[n×out]`.
pub fn linear(x: &[f32], w: &[f32], b: &[f32], n: usize, in_f: usize, out_f: usize) -> Vec<f32> {
    assert_eq!(x.len(), n * in_f);
    assert_eq!(w.len(), out_f * in_f);
    assert_eq!(b.len(), out_f);
    let mut y = vec![0.0f32; n * out_f];
    for row in 0..n {
        for o in 0..out_f {
            let mut acc = b[o];
            let x_row = &x[row * in_f..row * in_f + in_f];
            let w_row = &w[o * in_f..o * in_f + in_f];
            for i in 0..in_f {
                acc += x_row[i] * w_row[i];
            }
            y[row * out_f + o] = acc;
        }
    }
    y
}

const SQRT_2_OVER_PI: f32 = 0.7978846;

/// GELU activation (tanh approximation), matching `jax.nn.gelu(approximate=True)`.
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

pub fn gelu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = gelu(*v);
    }
}

/// SiLU / swish activation: `x · sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Numerically-stable softmax over a single vector (in place).
pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// Row-wise LayerNorm over the last dimension of size `d`.
/// `y = (x - mean) / sqrt(var + eps) * gamma + beta`, population variance.
pub fn layernorm(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    rows: usize,
    d: usize,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(x.len(), rows * d);
    assert_eq!(gamma.len(), d);
    assert_eq!(beta.len(), d);
    let mut out = vec![0.0f32; rows * d];
    for r in 0..rows {
        let row = &x[r * d..r * d + d];
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for c in 0..d {
            out[r * d + c] = (row[c] - mean) * inv_std * gamma[c] + beta[c];
        }
    }
    out
}

/// Row-wise RMSNorm over the last dimension of size `d`: `y = x / sqrt(mean(x²) + eps) * gamma`.
pub fn rmsnorm(x: &[f32], gamma: &[f32], rows: usize, d: usize, eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), rows * d);
    assert_eq!(gamma.len(), d);
    let mut out = vec![0.0f32; rows * d];
    for r in 0..rows {
        let row = &x[r * d..r * d + d];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for c in 0..d {
            out[r * d + c] = row[c] * inv * gamma[c];
        }
    }
    out
}

/// Applies rotary position embeddings (rotate-half / GPT-NeoX convention) to
/// `x`[rows×d] in place. `positions[r]` gives the absolute position of row `r`.
/// `d` must be even.
pub fn rope_inplace(x: &mut [f32], positions: &[usize], rows: usize, d: usize, theta: f32) {
    assert_eq!(x.len(), rows * d);
    assert_eq!(positions.len(), rows);
    assert_eq!(d % 2, 0, "rope head dim must be even");
    let half = d / 2;
    for r in 0..rows {
        let pos = positions[r] as f32;
        let row = &mut x[r * d..r * d + d];
        for i in 0..half {
            let inv_freq = theta.powf(-2.0 * i as f32 / d as f32);
            let angle = pos * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x1 = row[i];
            let x2 = row[i + half];
            row[i] = x1 * cos - x2 * sin;
            row[i + half] = x2 * cos + x1 * sin;
        }
    }
}

/// Single-head scaled dot-product attention.
///
/// `q`[m×d] attends over `k`/`v`[seq×d] → `out`[m×d]. When `causal` is set, query
/// row `i` may only attend to keys `j ≤ i + q_offset` (matching cached generation,
/// where the `m` new queries start at absolute position `q_offset`).
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    m: usize,
    seq: usize,
    d: usize,
    causal: bool,
    q_offset: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), m * d);
    assert_eq!(k.len(), seq * d);
    assert_eq!(v.len(), seq * d);
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0f32; m * d];
    let mut scores = vec![0.0f32; seq];
    for qi in 0..m {
        let limit = if causal {
            (qi + q_offset + 1).min(seq)
        } else {
            seq
        };
        for (kj, score) in scores.iter_mut().enumerate().take(limit) {
            let mut dot = 0.0f32;
            for di in 0..d {
                dot += q[qi * d + di] * k[kj * d + di];
            }
            *score = dot * scale;
        }
        softmax_inplace(&mut scores[..limit]);
        for di in 0..d {
            let mut acc = 0.0f32;
            for (kj, &w) in scores[..limit].iter().enumerate() {
                acc += w * v[kj * d + di];
            }
            out[qi * d + di] = acc;
        }
    }
    out
}

/// Dequantizes a per-row INT8 weight matrix: `out[r,c] = q[r,c] · scale[r]`.
pub fn dequantize_int8(q: &[i8], scales: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(q.len(), rows * cols);
    assert_eq!(scales.len(), rows);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let s = scales[r];
        for c in 0..cols {
            out[r * cols + c] = q[r * cols + c] as f32 * s;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= tol, "expected {y}, got {x}");
        }
    }

    #[test]
    fn matmul_identity() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let id = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(matmul(&a, &id, 2, 2, 2), a.to_vec());
    }

    #[test]
    fn matmul_known() {
        // [1 2 3] · [[1],[0],[-1]] = [1*1 + 2*0 + 3*-1] = [-2]
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 0.0, -1.0];
        assert_eq!(matmul(&a, &b, 1, 3, 1), vec![-2.0]);
    }

    #[test]
    fn linear_matches_manual() {
        // x = [1,2], W = [[1,0],[0,1],[1,1]] (out=3,in=2), b=[1,2,3]
        let x = [1.0, 2.0];
        let w = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let b = [1.0, 2.0, 3.0];
        let y = linear(&x, &w, &b, 1, 2, 3);
        assert_eq!(y, vec![1.0 + 1.0, 2.0 + 2.0, 3.0 + 3.0]);
    }

    #[test]
    fn gelu_reference_points() {
        assert!((gelu(0.0)).abs() < 1e-7);
        // gelu(1) ≈ 0.8412 with tanh approximation
        assert!((gelu(1.0) - 0.841_192).abs() < 1e-4);
        assert!((gelu(-1.0) - -0.158_808).abs() < 1e-4);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0, 2.0, 3.0, 4.0];
        softmax_inplace(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        // monotonic: larger logit -> larger prob
        assert!(x[3] > x[0]);
    }

    #[test]
    fn layernorm_zero_mean_unit_var() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let y = layernorm(&x, &gamma, &beta, 1, 4, 1e-5);
        let mean: f32 = y.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4);
        let var: f32 = y.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-2);
    }

    #[test]
    fn rmsnorm_scales_correctly() {
        let x = [3.0, 4.0]; // rms = sqrt((9+16)/2) = sqrt(12.5)
        let gamma = [1.0, 1.0];
        let y = rmsnorm(&x, &gamma, 1, 2, 0.0);
        let rms = (12.5f32).sqrt();
        approx(&y, &[3.0 / rms, 4.0 / rms], 1e-6);
    }

    #[test]
    fn rope_preserves_norm() {
        // Rotation is norm-preserving per (i, i+half) pair.
        let mut x = [0.3, 0.7, -0.2, 0.5];
        let before: f32 = x.iter().map(|v| v * v).sum();
        rope_inplace(&mut x, &[5], 1, 4, 10000.0);
        let after: f32 = x.iter().map(|v| v * v).sum();
        assert!((before - after).abs() < 1e-5);
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let mut x = [0.3, 0.7, -0.2, 0.5];
        let orig = x;
        rope_inplace(&mut x, &[0], 1, 4, 10000.0);
        approx(&x, &orig, 1e-6);
    }

    #[test]
    fn attention_uniform_when_keys_equal() {
        // All keys identical -> uniform weights -> output = mean of V rows.
        let d = 2;
        let seq = 3;
        let q = [1.0, 1.0];
        let k = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let v = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let out = attention(&q, &k, &v, 1, seq, d, false, 0);
        approx(&out, &[2.0, 0.0], 1e-5); // mean of [1,2,3] = 2
    }

    #[test]
    fn attention_causal_first_query_sees_only_first_key() {
        let d = 2;
        let seq = 2;
        let q = [1.0, 0.0]; // query at offset 0 -> only key 0 visible
        let k = [10.0, 0.0, 0.0, 10.0];
        let v = [5.0, 0.0, 0.0, 9.0];
        let out = attention(&q, &k, &v, 1, seq, d, true, 0);
        approx(&out, &[5.0, 0.0], 1e-5);
    }

    #[test]
    fn dequant_int8_roundtrip() {
        let q = [10i8, -10, 100];
        let scales = [0.1f32];
        let out = dequantize_int8(&q, &scales, 1, 3);
        approx(&out, &[1.0, -1.0, 10.0], 1e-6);
    }
}
