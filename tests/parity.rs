//! CPU-reference vs Metal parity tests.
//!
//! Every Metal kernel is checked against the pure-Rust reference in
//! `batch_forge::ops` on randomized inputs. These run on Apple Silicon only
//! (the kernels need a Metal device) and back the tolerance bounds documented
//! in `docs/correctness.md`. Run with `cargo test --test parity -- --nocapture`
//! to print the observed maximum absolute deviation for each op.
#![cfg(target_os = "macos")]

use batch_forge::metal_backend::MetalBackend;
use batch_forge::{ops, SHADER_SOURCE};

/// Tiny deterministic xorshift RNG so tests are reproducible without a dep.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform f32 in [-1, 1).
    fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn i8(&mut self) -> i8 {
        ((self.next_u64() % 255) as i32 - 127) as i8
    }
}

fn vecf(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.f32()).collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn backend() -> MetalBackend {
    MetalBackend::new(SHADER_SOURCE).expect("Metal init failed")
}

/// Asserts parity and prints the observed deviation (visible with --nocapture).
fn check(name: &str, cpu: &[f32], gpu: &[f32], tol: f32) {
    let d = max_diff(cpu, gpu);
    eprintln!("[parity] {name:<22} max|Δ| = {d:.3e}  (tol {tol:.0e})");
    assert!(d <= tol, "{name}: max diff {d:.3e} exceeds tol {tol:.0e}");
}

#[test]
fn matmul_parity() {
    let (m, k, n) = (32, 64, 48);
    let mut rng = Rng::new(1);
    let a = vecf(&mut rng, m * k);
    let b = vecf(&mut rng, k * n);
    let cpu = ops::matmul(&a, &b, m, k, n);
    let gpu = backend().matmul(&a, &b, m, k, n);
    check("matmul", &cpu, &gpu, 1e-3);
}

#[test]
fn linear_parity() {
    let (rows, in_f, out_f) = (16, 64, 40);
    let mut rng = Rng::new(2);
    let x = vecf(&mut rng, rows * in_f);
    let w = vecf(&mut rng, out_f * in_f);
    let b = vecf(&mut rng, out_f);
    let cpu = ops::linear(&x, &w, &b, rows, in_f, out_f);
    let gpu = backend().linear(&x, &w, &b, rows, in_f, out_f);
    check("linear", &cpu, &gpu, 1e-3);
}

#[test]
fn gelu_parity() {
    let mut rng = Rng::new(3);
    let x = vecf(&mut rng, 4096);
    let cpu: Vec<f32> = x.iter().map(|&v| ops::gelu(v)).collect();
    let gpu = backend().gelu(&x);
    check("gelu", &cpu, &gpu, 1e-5);
}

#[test]
fn layernorm_parity() {
    let (rows, d) = (16, 64);
    let mut rng = Rng::new(4);
    let x = vecf(&mut rng, rows * d);
    let gamma = vecf(&mut rng, d);
    let beta = vecf(&mut rng, d);
    let cpu = ops::layernorm(&x, &gamma, &beta, rows, d, 1e-5);
    let gpu = backend().layernorm(&x, &gamma, &beta, rows, d, 1e-5);
    check("layernorm", &cpu, &gpu, 1e-4);
}

#[test]
fn rmsnorm_parity() {
    let (rows, d) = (16, 64);
    let mut rng = Rng::new(5);
    let x = vecf(&mut rng, rows * d);
    let gamma = vecf(&mut rng, d);
    let cpu = ops::rmsnorm(&x, &gamma, rows, d, 1e-5);
    let gpu = backend().rmsnorm(&x, &gamma, rows, d, 1e-5);
    check("rmsnorm", &cpu, &gpu, 1e-4);
}

#[test]
fn rope_parity() {
    let (rows, d) = (8, 64);
    let mut rng = Rng::new(6);
    let mut cpu = vecf(&mut rng, rows * d);
    let gpu_in = cpu.clone();
    let positions: Vec<usize> = (0..rows).collect();
    let positions_u32: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
    ops::rope_inplace(&mut cpu, &positions, rows, d, 10000.0);
    let gpu = backend().rope(&gpu_in, &positions_u32, rows, d, 10000.0);
    check("rope", &cpu, &gpu, 1e-3);
}

#[test]
fn attention_parity_noncausal() {
    let (m, seq, d) = (4, 12, 32);
    let mut rng = Rng::new(7);
    let q = vecf(&mut rng, m * d);
    let k = vecf(&mut rng, seq * d);
    let v = vecf(&mut rng, seq * d);
    let cpu = ops::attention(&q, &k, &v, m, seq, d, false, 0);
    let gpu = backend().attention(&q, &k, &v, m, seq, d, false, 0);
    check("attention(full)", &cpu, &gpu, 1e-3);
}

#[test]
fn attention_parity_causal() {
    let (m, seq, d, q_offset) = (4, 10, 32, 3);
    let mut rng = Rng::new(8);
    let q = vecf(&mut rng, m * d);
    let k = vecf(&mut rng, seq * d);
    let v = vecf(&mut rng, seq * d);
    let cpu = ops::attention(&q, &k, &v, m, seq, d, true, q_offset);
    let gpu = backend().attention(&q, &k, &v, m, seq, d, true, q_offset);
    check("attention(causal)", &cpu, &gpu, 1e-3);
}

#[test]
fn quant_matmul_parity() {
    let (m, k, n) = (24, 48, 32);
    let mut rng = Rng::new(9);
    let a_i8: Vec<i8> = (0..m * k).map(|_| rng.i8()).collect();
    let scales: Vec<f32> = (0..m).map(|_| 0.01 + rng.f32().abs() * 0.05).collect();
    let b = vecf(&mut rng, k * n);
    // Reference: dequantize then dense matmul.
    let deq = ops::dequantize_int8(&a_i8, &scales, m, k);
    let cpu = ops::matmul(&deq, &b, m, k, n);
    let gpu = backend().quant_matmul(&a_i8, &scales, &b, m, k, n);
    check("quant_matmul", &cpu, &gpu, 1e-3);
}

#[test]
fn update_kv_cache_writes_correct_rows() {
    let be = backend();
    let (d, max_len) = (4usize, 8usize);
    let k_cache = be.create_buffer(&vec![0.0f32; max_len * d]).unwrap();
    let v_cache = be.create_buffer(&vec![0.0f32; max_len * d]).unwrap();

    let k1: Vec<f32> = (0..2 * d).map(|i| i as f32).collect();
    let v1: Vec<f32> = (0..2 * d).map(|i| (i + 100) as f32).collect();
    let bk1 = be.create_buffer(&k1).unwrap();
    let bv1 = be.create_buffer(&v1).unwrap();
    be.update_kv_cache(&bk1, &bv1, &k_cache, &v_cache, 2, 0, d);

    let k2: Vec<f32> = (0..3 * d).map(|i| (i + 1000) as f32).collect();
    let v2: Vec<f32> = (0..3 * d).map(|i| (i + 2000) as f32).collect();
    let bk2 = be.create_buffer(&k2).unwrap();
    let bv2 = be.create_buffer(&v2).unwrap();
    be.update_kv_cache(&bk2, &bv2, &k_cache, &v_cache, 3, 2, d);

    let k_read: Vec<f32> = be.read_buffer(&k_cache, 5 * d);
    let v_read: Vec<f32> = be.read_buffer(&v_cache, 5 * d);
    assert_eq!(&k_read[..2 * d], &k1[..]);
    assert_eq!(&k_read[2 * d..5 * d], &k2[..]);
    assert_eq!(&v_read[..2 * d], &v1[..]);
    assert_eq!(&v_read[2 * d..5 * d], &v2[..]);
}

/// End-to-end check of the cached generation path: build the cache with
/// `update_kv_cache`, attend with `kv_attention` reading straight from cache
/// buffers, and compare to the reference attention over the assembled K/V.
#[test]
fn cached_attention_matches_reference() {
    let be = backend();
    let (d, seq) = (16usize, 6usize);
    let mut rng = Rng::new(10);
    let k_full = vecf(&mut rng, seq * d);
    let v_full = vecf(&mut rng, seq * d);
    let q = vecf(&mut rng, d); // single query (m = 1)

    let k_cache = be.create_buffer(&vec![0.0f32; seq * d]).unwrap();
    let v_cache = be.create_buffer(&vec![0.0f32; seq * d]).unwrap();
    let bk = be.create_buffer(&k_full).unwrap();
    let bv = be.create_buffer(&v_full).unwrap();
    be.update_kv_cache(&bk, &bv, &k_cache, &v_cache, seq, 0, d);

    let bq = be.create_buffer(&q).unwrap();
    let bo = be.create_buffer_uninitialized::<f32>(d).unwrap();
    be.kv_attention(&bq, &k_cache, &v_cache, &bo, 1, seq, d, false, 0);
    let gpu: Vec<f32> = be.read_buffer(&bo, d);

    let cpu = ops::attention(&q, &k_full, &v_full, 1, seq, d, false, 0);
    check("cached_attention", &cpu, &gpu, 1e-3);
}

#[test]
fn matmul_tiled_parity() {
    let (m, k, n) = (40, 72, 56); // deliberately non-multiples of the 16 tile
    let mut rng = Rng::new(11);
    let a = vecf(&mut rng, m * k);
    let b = vecf(&mut rng, k * n);
    let cpu = ops::matmul(&a, &b, m, k, n);
    let gpu = backend().matmul_tiled(&a, &b, m, k, n);
    check("matmul_tiled", &cpu, &gpu, 1e-3);
}

#[test]
fn gelu_scaled_parity() {
    let be = backend();
    let mut rng = Rng::new(123);
    // GPT-2 MLP hidden size (5 x 3072) with realistic magnitudes.
    let x: Vec<f32> = (0..5 * 3072).map(|_| rng.f32() * 12.0).collect();
    let cpu: Vec<f32> = x.iter().map(|&v| ops::gelu(v)).collect();
    let gpu = be.gelu(&x);
    let nan = gpu.iter().filter(|v| !v.is_finite()).count();
    eprintln!("[parity] gelu_scaled nan count = {nan}");
    check("gelu_scaled", &cpu, &gpu, 1e-2);
}

#[test]
fn gpt2_sizes_parity() {
    let be = backend();
    let mut rng = Rng::new(99);
    // c_attn-shaped matmul
    let a = vecf(&mut rng, 5 * 768);
    let b = vecf(&mut rng, 768 * 2304);
    let cpu = ops::matmul(&a, &b, 5, 768, 2304);
    let gpu = be.matmul_tiled(&a, &b, 5, 768, 2304);
    check("mm 5x768x2304", &cpu, &gpu, 3e-3);
    // GPT-2 MHA shape
    let hd = 12 * 64;
    let q = vecf(&mut rng, 5 * hd);
    let k = vecf(&mut rng, 5 * hd);
    let v = vecf(&mut rng, 5 * hd);
    let cpu = ops::mha(&q, &k, &v, 5, 12, 64);
    let gpu = be.mha(&q, &k, &v, 5, 12, 64);
    check("mha 5x12x64", &cpu, &gpu, 2e-3);
    // tied LM head shape
    let x = vecf(&mut rng, 768);
    let w = vecf(&mut rng, 50257 * 768);
    let bz = vec![0.0f32; 50257];
    let cpu = ops::linear(&x, &w, &bz, 1, 768, 50257);
    let gpu = be.linear(&x, &w, &bz, 1, 768, 50257);
    check("linear lm_head", &cpu, &gpu, 4e-3);
}

#[test]
fn mha_parity() {
    let (seq, heads, head_dim) = (7, 3, 8);
    let hd = heads * head_dim;
    let mut rng = Rng::new(12);
    let q = vecf(&mut rng, seq * hd);
    let k = vecf(&mut rng, seq * hd);
    let v = vecf(&mut rng, seq * hd);
    let cpu = ops::mha(&q, &k, &v, seq, heads, head_dim);
    let gpu = backend().mha(&q, &k, &v, seq, heads, head_dim);
    check("mha", &cpu, &gpu, 1e-3);
}
