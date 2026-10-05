//! GPT-2 text generation on batch_forge.
//!
//!     cargo run --release --bin generate -- --prompt "The meaning of life is"
//!
//! Loads HuggingFace `gpt2` weights, tokenizes with the from-scratch BPE
//! tokenizer, runs the transformer on the Metal backend (or CPU), and streams
//! decoded text.

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use batch_forge::gpt2::{Config, Gpt2, LlmOps, Sampler};
use batch_forge::loader;
use batch_forge::tokenizer::Tokenizer;

const EOT: usize = 50256; // <|endoftext|>
const MODEL_DIR: &str = "models/gpt2";

#[derive(Debug)]
struct Args {
    prompt: String,
    max_new: usize,
    backend: String,
    temperature: f32,
    top_k: usize,
    seed: u64,
}

enum Command {
    Run(Args),
    Help,
    Version,
}

fn next_value(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_args(arguments: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut a = Args {
        prompt: "The meaning of life is".to_string(),
        max_new: 40,
        backend: if cfg!(target_os = "macos") { "metal" } else { "cpu" }.to_string(),
        temperature: 0.8,
        top_k: 40,
        seed: 42,
    };
    let mut greedy = false;
    let mut it = arguments.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(Command::Help),
            "--version" | "-V" => return Ok(Command::Version),
            "--prompt" | "-p" => a.prompt = next_value(&mut it, &arg)?,
            "--max-new" | "-n" => {
                a.max_new = next_value(&mut it, &arg)?.parse().map_err(|_| "--max-new must be a positive integer")?;
                if a.max_new == 0 { return Err("--max-new must be positive".into()); }
            }
            "--backend" | "-b" => {
                a.backend = next_value(&mut it, &arg)?;
                if !matches!(a.backend.as_str(), "cpu" | "metal") {
                    return Err("--backend must be cpu or metal".into());
                }
            }
            "--temperature" | "-t" => {
                a.temperature = next_value(&mut it, &arg)?.parse().map_err(|_| "--temperature must be a number")?;
                if !a.temperature.is_finite() || a.temperature < 0.0 {
                    return Err("--temperature must be finite and non-negative".into());
                }
            }
            "--top-k" | "-k" => a.top_k = next_value(&mut it, &arg)?.parse().map_err(|_| "--top-k must be a non-negative integer")?,
            "--seed" | "-s" => a.seed = next_value(&mut it, &arg)?.parse().map_err(|_| "--seed must be an unsigned integer")?,
            "--greedy" => greedy = true,
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if greedy { a.temperature = 0.0; }
    Ok(Command::Run(a))
}

fn print_help() {
    println!("GPT-2 text generation\n\nUSAGE:\n    generate [OPTIONS]\n\nOPTIONS:\n    -p, --prompt TEXT       Prompt (default: The meaning of life is)\n    -n, --max-new N         Maximum new tokens, positive integer (default: 40)\n    -b, --backend cpu|metal Compute backend (default: metal on macOS, cpu elsewhere)\n    -t, --temperature T     Finite, non-negative temperature (default: 0.8)\n    -k, --top-k K           Keep K candidates; 0 disables filtering (default: 40)\n    -s, --seed N            Sampling seed (default: 42)\n        --greedy           Force greedy sampling, regardless of option order\n    -h, --help             Show help without loading model assets\n    -V, --version          Show package version");
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

fn main() -> ExitCode {
    let args = match parse_args(std::env::args().skip(1)) {
        Ok(Command::Help) => { print_help(); return ExitCode::SUCCESS; }
        Ok(Command::Version) => { println!("{}", env!("CARGO_PKG_VERSION")); return ExitCode::SUCCESS; }
        Ok(Command::Run(args)) => args,
        Err(error) => { eprintln!("error: {error}\nUse --help for usage."); return ExitCode::from(2); }
    };
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
                return ExitCode::SUCCESS;
            }
            Err(e) => eprintln!("Metal unavailable ({e}); using CPU"),
        }
    }
    run(&batch_forge::model::CpuBackend, &model, &tok, &args);
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(flags: &[&str]) -> Result<Command, String> {
        parse_args(flags.iter().map(|s| s.to_string()))
    }

    #[test]
    fn rejects_invalid_flags_values_and_missing_values() {
        for args in [
            vec!["--bogus"], vec!["--seed"], vec!["--max-new", "no"],
            vec!["--max-new", "0"], vec!["--backend", "cuda"],
            vec!["--temperature", "NaN"], vec!["--temperature", "inf"],
            vec!["--temperature", "-1"], vec!["--top-k", "-2"],
        ] {
            assert!(parse(&args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn greedy_order_and_zero_sampling_values_are_explicit() {
        for flags in [
            vec!["--greedy", "--temperature", "1", "--top-k", "0", "--seed", "0"],
            vec!["--temperature", "1", "--greedy", "--top-k", "0", "--seed", "0"],
        ] {
            let Command::Run(args) = parse(&flags).unwrap() else { panic!("expected run"); };
            assert_eq!(args.temperature, 0.0);
            assert_eq!(args.top_k, 0);
            assert_eq!(args.seed, 0);
        }
        assert!(matches!(parse(&["--help"]), Ok(Command::Help)));
        assert!(matches!(parse(&["--version"]), Ok(Command::Version)));
    }
}
