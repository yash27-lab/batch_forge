//! GPT-2 inference.
//!
//! A from-scratch GPT-2 forward pass that runs on either the CPU reference or
//! the Metal backend through the [`LlmOps`] trait, so the two are checked for
//! parity exactly like the individual kernels. Heavy ops (matmul, attention,
//! layernorm, gelu) go through the backend; the cheap element-wise glue
//! (embedding gather, bias add, residual) stays in plain Rust.
//!
//! Weight layout follows HuggingFace `gpt2`: the `Conv1D` layers store weights
//! as `[in, out]` so `y = x @ W + b` is a plain matmul (no transpose), and the
//! LM head is tied to the token embedding.

use std::collections::HashMap;

use thiserror::Error;

use crate::ops;
use crate::tensor::Tensor;

#[derive(Error, Debug)]
pub enum Gpt2Error {
    #[error("missing tensor '{0}'")]
    Missing(String),
}

/// GPT-2 hyperparameters. Defaults are the 124M ("small") configuration.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub n_ctx: usize,
    pub vocab_size: usize,
    pub eps: f32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
            n_ctx: 1024,
            vocab_size: 50257,
            eps: 1e-5,
        }
    }
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

/// The compute surface GPT-2 needs from a backend. Implemented by both the CPU
/// reference and the Metal backend; the model is generic over it.
pub trait LlmOps {
    /// `C[m,n] = A[m,k] * B[k,n]` (row-major). Used for the Conv1D projections.
    fn mm(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32>;
    /// `y = x · Wᵀ` with `w` = `[out,in]`; used for the tied LM head.
    fn lm_head(&self, x: &[f32], w: &[f32], rows: usize, in_f: usize, out_f: usize) -> Vec<f32>;
    fn layernorm(
        &self,
        x: &[f32],
        g: &[f32],
        b: &[f32],
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Vec<f32>;
    fn gelu(&self, x: &[f32]) -> Vec<f32>;
    fn mha(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> Vec<f32>;
    fn name(&self) -> &'static str;
}

impl LlmOps for crate::model::CpuBackend {
    fn mm(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        ops::matmul(a, b, m, k, n)
    }
    fn lm_head(&self, x: &[f32], w: &[f32], rows: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        let zeros = vec![0.0f32; out_f];
        ops::linear(x, w, &zeros, rows, in_f, out_f)
    }
    fn layernorm(
        &self,
        x: &[f32],
        g: &[f32],
        b: &[f32],
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Vec<f32> {
        ops::layernorm(x, g, b, rows, d, eps)
    }
    fn gelu(&self, x: &[f32]) -> Vec<f32> {
        let mut y = x.to_vec();
        ops::gelu_inplace(&mut y);
        y
    }
    fn mha(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        ops::mha(q, k, v, seq, heads, head_dim)
    }
    fn name(&self) -> &'static str {
        "cpu"
    }
}

#[cfg(target_os = "macos")]
impl LlmOps for crate::metal_backend::MetalBackend {
    fn mm(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        self.matmul_tiled(a, b, m, k, n)
    }
    fn lm_head(&self, x: &[f32], w: &[f32], rows: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        let zeros = vec![0.0f32; out_f];
        self.linear(x, w, &zeros, rows, in_f, out_f)
    }
    fn layernorm(
        &self,
        x: &[f32],
        g: &[f32],
        b: &[f32],
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Vec<f32> {
        crate::metal_backend::MetalBackend::layernorm(self, x, g, b, rows, d, eps)
    }
    fn gelu(&self, x: &[f32]) -> Vec<f32> {
        crate::metal_backend::MetalBackend::gelu(self, x)
    }
    fn mha(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        crate::metal_backend::MetalBackend::mha(self, q, k, v, seq, heads, head_dim)
    }
    fn name(&self) -> &'static str {
        "metal"
    }
}

struct Layer {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    attn_w: Vec<f32>, // c_attn.weight [n_embd, 3*n_embd]
    attn_b: Vec<f32>, // [3*n_embd]
    proj_w: Vec<f32>, // c_proj.weight [n_embd, n_embd]
    proj_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    fc_w: Vec<f32>, // mlp.c_fc.weight [n_embd, 4*n_embd]
    fc_b: Vec<f32>,
    fc_proj_w: Vec<f32>, // mlp.c_proj.weight [4*n_embd, n_embd]
    fc_proj_b: Vec<f32>,
}

/// A loaded GPT-2 model. All weights are owned `f32`.
pub struct Gpt2 {
    pub config: Config,
    wte: Vec<f32>, // [vocab, n_embd]
    wpe: Vec<f32>, // [n_ctx, n_embd]
    layers: Vec<Layer>,
    lnf_w: Vec<f32>,
    lnf_b: Vec<f32>,
}

fn take(map: &mut HashMap<String, Tensor>, name: &str) -> Result<Vec<f32>, Gpt2Error> {
    map.remove(name)
        .map(|t| t.data)
        .ok_or_else(|| Gpt2Error::Missing(name.to_string()))
}

impl Gpt2 {
    /// Builds a model from a HuggingFace `gpt2` safetensors tensor map.
    pub fn from_tensors(
        mut map: HashMap<String, Tensor>,
        config: Config,
    ) -> Result<Self, Gpt2Error> {
        let wte = take(&mut map, "wte.weight")?;
        let wpe = take(&mut map, "wpe.weight")?;
        let lnf_w = take(&mut map, "ln_f.weight")?;
        let lnf_b = take(&mut map, "ln_f.bias")?;
        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let p = format!("h.{i}.");
            layers.push(Layer {
                ln1_w: take(&mut map, &format!("{p}ln_1.weight"))?,
                ln1_b: take(&mut map, &format!("{p}ln_1.bias"))?,
                attn_w: take(&mut map, &format!("{p}attn.c_attn.weight"))?,
                attn_b: take(&mut map, &format!("{p}attn.c_attn.bias"))?,
                proj_w: take(&mut map, &format!("{p}attn.c_proj.weight"))?,
                proj_b: take(&mut map, &format!("{p}attn.c_proj.bias"))?,
                ln2_w: take(&mut map, &format!("{p}ln_2.weight"))?,
                ln2_b: take(&mut map, &format!("{p}ln_2.bias"))?,
                fc_w: take(&mut map, &format!("{p}mlp.c_fc.weight"))?,
                fc_b: take(&mut map, &format!("{p}mlp.c_fc.bias"))?,
                fc_proj_w: take(&mut map, &format!("{p}mlp.c_proj.weight"))?,
                fc_proj_b: take(&mut map, &format!("{p}mlp.c_proj.bias"))?,
            });
        }
        Ok(Self {
            config,
            wte,
            wpe,
            layers,
            lnf_w,
            lnf_b,
        })
    }

    /// Runs the forward pass over `tokens` and returns the logits for the final
    /// position only (`[vocab_size]`), which is all generation needs.
    pub fn forward<B: LlmOps + ?Sized>(&self, backend: &B, tokens: &[usize]) -> Vec<f32> {
        let cfg = self.config;
        let (seq, d) = (tokens.len(), cfg.n_embd);
        let eps = cfg.eps;

        // Token + positional embeddings.
        let mut x = vec![0.0f32; seq * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let wt = &self.wte[tok * d..tok * d + d];
            let wp = &self.wpe[i * d..i * d + d];
            for j in 0..d {
                x[i * d + j] = wt[j] + wp[j];
            }
        }

        for layer in &self.layers {
            // --- attention block ---
            let ln1 = backend.layernorm(&x, &layer.ln1_w, &layer.ln1_b, seq, d, eps);
            let mut qkv = backend.mm(&ln1, &layer.attn_w, seq, d, 3 * d);
            add_bias(&mut qkv, &layer.attn_b, seq, 3 * d);
            let (q, k, v) = split_qkv(&qkv, seq, d);
            let attn = backend.mha(&q, &k, &v, seq, cfg.n_head, cfg.head_dim());
            let mut proj = backend.mm(&attn, &layer.proj_w, seq, d, d);
            add_bias(&mut proj, &layer.proj_b, seq, d);
            residual_add(&mut x, &proj);

            // --- MLP block ---
            let ln2 = backend.layernorm(&x, &layer.ln2_w, &layer.ln2_b, seq, d, eps);
            let mut fc = backend.mm(&ln2, &layer.fc_w, seq, d, 4 * d);
            add_bias(&mut fc, &layer.fc_b, seq, 4 * d);
            let act = backend.gelu(&fc);
            let mut fc2 = backend.mm(&act, &layer.fc_proj_w, seq, 4 * d, d);
            add_bias(&mut fc2, &layer.fc_proj_b, seq, d);
            residual_add(&mut x, &fc2);
        }

        let xf = backend.layernorm(&x, &self.lnf_w, &self.lnf_b, seq, d, eps);
        let last = &xf[(seq - 1) * d..seq * d];
        // Tied LM head: logits = last · wteᵀ.
        backend.lm_head(last, &self.wte, 1, d, cfg.vocab_size)
    }

    /// Autoregressively generates up to `max_new` tokens, returning the full
    /// token sequence (prompt + generated). Stops early on the end-of-text token.
    pub fn generate<B: LlmOps + ?Sized>(
        &self,
        backend: &B,
        prompt: &[usize],
        max_new: usize,
        sampler: &Sampler,
        eot_token: usize,
        mut on_token: impl FnMut(usize),
    ) -> Vec<usize> {
        let mut toks = prompt.to_vec();
        let n_ctx = self.config.n_ctx;
        let mut rng = Rng::new(sampler.seed);
        for _ in 0..max_new {
            let start = toks.len().saturating_sub(n_ctx);
            let logits = self.forward(backend, &toks[start..]);
            let next = sampler.sample(&logits, &mut rng);
            toks.push(next);
            on_token(next);
            if next == eot_token {
                break;
            }
        }
        toks
    }
}

fn add_bias(x: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    debug_assert_eq!(bias.len(), cols);
    for r in 0..rows {
        for c in 0..cols {
            x[r * cols + c] += bias[c];
        }
    }
}

fn residual_add(x: &mut [f32], y: &[f32]) {
    for (xi, yi) in x.iter_mut().zip(y) {
        *xi += yi;
    }
}

/// Splits a `[seq, 3*d]` QKV tensor into three contiguous `[seq, d]` tensors.
fn split_qkv(qkv: &[f32], seq: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0f32; seq * d];
    let mut k = vec![0.0f32; seq * d];
    let mut v = vec![0.0f32; seq * d];
    for i in 0..seq {
        let row = &qkv[i * 3 * d..i * 3 * d + 3 * d];
        q[i * d..i * d + d].copy_from_slice(&row[0..d]);
        k[i * d..i * d + d].copy_from_slice(&row[d..2 * d]);
        v[i * d..i * d + d].copy_from_slice(&row[2 * d..3 * d]);
    }
    (q, k, v)
}

/// Token sampling strategy.
pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub seed: u64,
}

impl Sampler {
    /// Greedy (argmax) sampling.
    pub fn greedy() -> Self {
        Sampler {
            temperature: 0.0,
            top_k: 0,
            seed: 0,
        }
    }

    fn sample(&self, logits: &[f32], rng: &mut Rng) -> usize {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }
        // Optional top-k: keep only the k highest logits.
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        if self.top_k > 0 && self.top_k < logits.len() {
            idx.select_nth_unstable_by(self.top_k, |&a, &b| {
                logits[b].partial_cmp(&logits[a]).unwrap()
            });
            idx.truncate(self.top_k);
        }
        let max = idx.iter().map(|&i| logits[i]).fold(f32::MIN, f32::max);
        let mut probs: Vec<f32> = idx
            .iter()
            .map(|&i| ((logits[i] - max) / self.temperature).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }
        let r = rng.next_f32();
        let mut acc = 0.0;
        for (j, &p) in probs.iter().enumerate() {
            acc += p;
            if r <= acc {
                return idx[j];
            }
        }
        idx[idx.len() - 1]
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// Small xorshift RNG for sampling (keeps generation dependency-free).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E3779B97F4A7C15)
    }
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0.max(1);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }
}
