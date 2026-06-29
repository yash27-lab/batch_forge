//! GPT-2 text generation on batch_forge.
//!
//!     cargo run --release --bin generate -- --prompt "The meaning of life is"
//!
//! Loads HuggingFace `gpt2` weights, tokenizes with the from-scratch BPE
//! tokenizer, runs the transformer on the Metal backend (or CPU), and streams
//! decoded text.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use batch_forge::gpt2::{Config, Gpt2, LlmOps, Sampler};
use batch_forge::loader;
use batch_forge::tokenizer::Tokenizer;

const EOT: usize = 50256; // <|endoftext|>
const MODEL_DIR: &str = "models/gpt2";

struct Args {
    prompt: String,
    max_new: usize,
    backend: String,
    temperature: f32,
    top_k: usize,
    seed: u64,
}

fn parse_args() -> Args {
    let mut a = Args {
        prompt: "The meaning of life is".to_string(),
        max_new: 40,
        backend: if cfg!(target_os = "macos") {
            "metal"
        } else {
            "cpu"
        }
        .to_string(),
        temperature: 0.8,
        top_k: 40,
        seed: 42,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--prompt" | "-p" => a.prompt = it.next().unwrap_or_default(),
            "--max-new" | "-n" => a.max_new = it.next().and_then(|s| s.parse().ok()).unwrap_or(40),
            "--backend" | "-b" => a.backend = it.next().unwrap_or_default(),
            "--temperature" | "-t" => {
                a.temperature = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.8)
            }
            "--top-k" | "-k" => a.top_k = it.next().and_then(|s| s.parse().ok()).unwrap_or(40),
            "--seed" | "-s" => a.seed = it.next().and_then(|s| s.parse().ok()).unwrap_or(42),
            "--greedy" => a.temperature = 0.0,
            _ => {}
        }
    }
    a
}

fn run<B: LlmOps>(backend: &B, model: &Gpt2, tok: &Tokenizer, args: &Args) {
    let sampler = Sampler {
        temperature: args.temperature,
        top_k: args.top_k,
        seed: args.seed,
    };
    let prompt_ids = tok.encode(&args.prompt);
    println!(
        "backend={}  prompt_tokens={}  max_new={}  temp={}  top_k={}\n",
        backend.name(),
        prompt_ids.len(),
        args.max_new,
        args.temperature,
        args.top_k
    );

    print!("{}", args.prompt);
    std::io::stdout().flush().ok();

    let mut generated: Vec<usize> = Vec::new();
    let mut printed = 0usize;
    let start = Instant::now();
    model.generate(
        backend,
        &prompt_ids,
        args.max_new,
        &sampler,
        EOT,
        |tok_id| {
            generated.push(tok_id);
            // Decode the whole generated suffix and print only the new text, so
            // multi-byte characters that span tokens render correctly.
            let text = tok.decode(&generated);
            if text.len() > printed {
                print!("{}", &text[printed..]);
                std::io::stdout().flush().ok();
                printed = text.len();
            }
        },
    );
    let elapsed = start.elapsed();

    let n = generated.len().max(1);
    println!(
        "\n\n[{} tokens in {:.2?}  =  {:.1} tok/s on {}]",
        generated.len(),
        elapsed,
        generated.len() as f64 / elapsed.as_secs_f64(),
        backend.name(),
    );
    let _ = n;
}

fn main() {
    let args = parse_args();
    let model_path = Path::new(MODEL_DIR).join("model.safetensors");
    if !model_path.exists() {
        eprintln!(
            "GPT-2 weights not found at {}.\nDownload them with:\n  \
             python python/fetch_gpt2.py    (or)\n  \
             curl -L https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors -o {}",
            model_path.display(),
            model_path.display()
        );
        std::process::exit(1);
    }

    eprintln!("loading GPT-2 weights …");
    let tensors = loader::load_safetensors(&model_path).expect("load weights");
    let model = Gpt2::from_tensors(tensors, Config::default()).expect("build model");
    let tok = Tokenizer::from_files(
        &Path::new(MODEL_DIR).join("vocab.json"),
        &Path::new(MODEL_DIR).join("merges.txt"),
    )
    .expect("load tokenizer");

    #[cfg(target_os = "macos")]
    if args.backend == "metal" {
        match batch_forge::metal_backend::MetalBackend::new(batch_forge::SHADER_SOURCE) {
            Ok(m) => {
                run(&m, &model, &tok, &args);
                return;
            }
            Err(e) => eprintln!("Metal unavailable ({e}); using CPU"),
        }
    }
    run(&batch_forge::model::CpuBackend, &model, &tok, &args);
}
