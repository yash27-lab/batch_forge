//! End-to-end GPT-2 verification.
//!
//! Requires the real `gpt2` weights under `models/gpt2/` (gitignored), so these
//! tests skip cleanly when the weights are absent (e.g. in CI). Locally they
//! assert the Rust forward reproduces HuggingFace GPT-2's prediction and that
//! the CPU and Metal backends agree.

use std::path::Path;

use batch_forge::gpt2::{Config, Gpt2};
use batch_forge::loader;
use batch_forge::model::CpuBackend;

const MODEL: &str = "models/gpt2/model.safetensors";
// "The meaning of life is"
const TOKENS: [usize; 5] = [464, 3616, 286, 1204, 318];
// Greedy next-token ranking from HuggingFace GPT-2 (verified by python/gpt2_reference.py).
const EXPECTED_TOP5: [usize; 5] = [407, 284, 262, 326, 257];

fn load() -> Option<Gpt2> {
    if !Path::new(MODEL).exists() {
        eprintln!("skipping: {MODEL} not present");
        return None;
    }
    let tensors = loader::load_safetensors(Path::new(MODEL)).expect("load weights");
    Some(Gpt2::from_tensors(tensors, Config::default()).expect("build model"))
}

fn top5(logits: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(5);
    idx
}

#[test]
fn cpu_forward_matches_huggingface() {
    let Some(model) = load() else { return };
    let logits = model.forward(&CpuBackend, &TOKENS);
    assert_eq!(top5(&logits), EXPECTED_TOP5, "GPT-2 prediction diverged");
}

#[cfg(target_os = "macos")]
#[test]
fn cpu_metal_logits_agree() {
    let Some(model) = load() else { return };
    let metal = batch_forge::metal_backend::MetalBackend::new(batch_forge::SHADER_SOURCE)
        .expect("metal init");
    let cpu = model.forward(&CpuBackend, &TOKENS);
    let gpu = model.forward(&metal, &TOKENS);
    let max_diff = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[gpt2] CPU vs Metal logits max|Δ| = {max_diff:.3e}");
    assert!(max_diff < 1e-2, "CPU/Metal logits diverged: {max_diff}");
    assert_eq!(top5(&gpu), EXPECTED_TOP5);
}
