//! batch_forge CLI.
//!
//! Loads an exported model, runs its forward pass on the CPU reference and (on
//! Apple Silicon) the Metal backend, cross-checks the two, and optionally
//! verifies against a saved JAX/NumPy reference output. `--requests N` exercises
//! the async engine with N concurrent in-flight requests.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use tracing::{error, info, warn};

use batch_forge::loader;
use batch_forge::model::{Backend, CpuBackend, Mlp};
use batch_forge::tensor::Tensor;

/// Tolerance for the reference-verification pass/fail check.
const VERIFY_TOL: f32 = 1e-3;

#[derive(Debug)]
struct Args {
    model: PathBuf,
    verify: Option<PathBuf>,
    backend: BackendChoice,
    requests: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BackendChoice {
    Cpu,
    Metal,
    Both,
}

fn parse_args() -> Result<Args, String> {
    let mut model = PathBuf::from("model.safetensors");
    let mut verify = None;
    let default_backend = if cfg!(target_os = "macos") {
        BackendChoice::Both
    } else {
        BackendChoice::Cpu
    };
    let mut backend = default_backend;
    let mut requests = 0usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "-m" => model = args.next().ok_or("--model needs a path")?.into(),
            "--verify" | "-v" => verify = Some(args.next().ok_or("--verify needs a path")?.into()),
            "--backend" | "-b" => {
                backend = match args.next().as_deref() {
                    Some("cpu") => BackendChoice::Cpu,
                    Some("metal") => BackendChoice::Metal,
                    Some("both") => BackendChoice::Both,
                    other => return Err(format!("unknown backend {other:?}")),
                }
            }
            "--requests" | "-r" => {
                requests = args
                    .next()
                    .ok_or("--requests needs a number")?
                    .parse()
                    .map_err(|_| "--requests must be an integer")?;
            }
            "--help" | "-h" => return Err("help".into()),
            // Accepted for README compatibility; this build has no tokenizer yet.
            "--prompt" | "-p" => {
                let _ = args.next();
                warn!("--prompt is accepted but ignored: no tokenizer in this build (roadmap)");
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Args {
        model,
        verify,
        backend,
        requests,
    })
}

fn print_help() {
    println!(
        "batch_forge {} — verified Metal inference for JAX/Equinox models\n\
\n\
USAGE:\n\
    batch_forge [--model PATH] [--backend cpu|metal|both] [--verify PATH] [--requests N]\n\
\n\
OPTIONS:\n\
    -m, --model PATH      Safetensors checkpoint (default: model.safetensors)\n\
    -b, --backend WHICH   Which backend(s) to run (default: both on macOS, cpu elsewhere)\n\
    -v, --verify PATH     Reference safetensors with `input`/`output`; checks numerical parity\n\
    -r, --requests N      Run the async engine with N concurrent requests\n\
    -h, --help            Show this help\n\
\n\
Generate a demo model with:  python python/make_demo_model.py",
        env!("CARGO_PKG_VERSION")
    );
}

/// Short summary of a tensor: shape, first values, and L2 norm.
fn summarize(t: &Tensor) -> String {
    let l2 = t.data.iter().map(|v| v * v).sum::<f32>().sqrt();
    let head: Vec<String> = t.data.iter().take(4).map(|v| format!("{v:+.4}")).collect();
    format!(
        "shape {:?}, ‖·‖₂={l2:.4}, head=[{}, …]",
        t.shape,
        head.join(", ")
    )
}

/// Builds the input: from the reference file if present, else a deterministic vector.
fn build_input(verify: &Option<PathBuf>, in_features: usize) -> Result<Tensor, String> {
    if let Some(path) = verify {
        let map = loader::load_safetensors(path).map_err(|e| format!("load --verify: {e}"))?;
        if let Some(input) = map.get("input") {
            return Ok(input.clone());
        }
        warn!("--verify file has no `input` tensor; using a synthetic input");
    }
    let data = (0..in_features).map(|i| (i as f32 * 0.01).sin()).collect();
    Tensor::new(data, vec![1, in_features]).map_err(|e| e.to_string())
}

#[cfg(target_os = "macos")]
fn make_metal() -> Option<Arc<batch_forge::metal_backend::MetalBackend>> {
    match batch_forge::metal_backend::MetalBackend::new(batch_forge::SHADER_SOURCE) {
        Ok(b) => Some(Arc::new(b)),
        Err(e) => {
            warn!("Metal unavailable ({e}); falling back to CPU");
            None
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            if e != "help" {
                eprintln!("error: {e}\n");
            }
            print_help();
            return if e == "help" {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            };
        }
    };

    info!("batch_forge {} starting", env!("CARGO_PKG_VERSION"));

    if !args.model.exists() {
        warn!("model not found at {}", args.model.display());
        println!(
            "\nNo checkpoint at `{}`. Generate the demo model first:\n    python python/make_demo_model.py\n\
or export your own Equinox model:\n    python python/export_eqx.py --out model.safetensors",
            args.model.display()
        );
        return ExitCode::SUCCESS;
    }

    let model = match load_model(&args.model) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            error!("{e}");
            return ExitCode::FAILURE;
        }
    };
    info!(
        "loaded MLP: {} layers, in={}, out={}",
        model.layers.len(),
        model.in_features(),
        model.out_features()
    );

    let input = match build_input(&args.verify, model.in_features()) {
        Ok(t) => t,
        Err(e) => {
            error!("{e}");
            return ExitCode::FAILURE;
        }
    };

    // --- CPU reference forward (always available) ---
    let cpu_out = model.forward(&CpuBackend, &input).expect("cpu forward");
    if args.backend != BackendChoice::Metal {
        info!("[cpu]   output: {}", summarize(&cpu_out));
    }

    // --- Metal forward + cross-backend check ---
    // `mut` is only used on macOS, where the Metal result replaces the CPU one.
    #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
    let mut production = cpu_out.clone();
    #[cfg(target_os = "macos")]
    if args.backend != BackendChoice::Cpu {
        if let Some(metal) = make_metal() {
            let metal_out = model
                .forward(metal.as_ref(), &input)
                .expect("metal forward");
            info!("[metal] output: {}", summarize(&metal_out));
            let diff = cpu_out.max_abs_diff(&metal_out);
            info!("[check] CPU vs Metal max|Δ| = {diff:.3e}");
            production = metal_out;
        }
    }

    // --- Reference verification ---
    let mut exit = ExitCode::SUCCESS;
    if let Some(path) = &args.verify {
        match verify_against_reference(path, &production) {
            Ok(diff) => {
                if diff <= VERIFY_TOL {
                    info!(
                        "[verify] PASS — max|Δ| vs reference = {diff:.3e} (tol {VERIFY_TOL:.0e})"
                    );
                } else {
                    error!(
                        "[verify] FAIL — max|Δ| vs reference = {diff:.3e} (tol {VERIFY_TOL:.0e})"
                    );
                    exit = ExitCode::FAILURE;
                }
            }
            Err(e) => {
                error!("[verify] {e}");
                exit = ExitCode::FAILURE;
            }
        }
    }

    // --- Async engine demo ---
    if args.requests > 0 {
        run_async_demo(Arc::clone(&model), &input, args.requests, args.backend).await;
    }

    exit
}

fn load_model(path: &Path) -> Result<Mlp, String> {
    let tensors = loader::load_safetensors(path).map_err(|e| format!("load model: {e}"))?;
    Mlp::from_tensors(&tensors).map_err(|e| format!("build model: {e}"))
}

fn verify_against_reference(path: &Path, produced: &Tensor) -> Result<f32, String> {
    let map = loader::load_safetensors(path).map_err(|e| format!("load reference: {e}"))?;
    let reference = map
        .get("output")
        .ok_or("reference file has no `output` tensor")?;
    if reference.shape != produced.shape {
        return Err(format!(
            "shape mismatch: produced {:?}, reference {:?}",
            produced.shape, reference.shape
        ));
    }
    Ok(produced.max_abs_diff(reference))
}

async fn run_async_demo(model: Arc<Mlp>, input: &Tensor, requests: usize, choice: BackendChoice) {
    let backend: Arc<dyn Backend + Send + Sync> = pick_async_backend(choice);
    info!(
        "[engine] dispatching {requests} concurrent requests on `{}`",
        backend.name()
    );
    let submitter = batch_forge::engine::spawn(backend, model, requests.max(1));

    let start = std::time::Instant::now();
    let mut handles = Vec::with_capacity(requests);
    for id in 0..requests as u64 {
        let s = submitter.clone();
        let inp = input.clone();
        handles.push(tokio::spawn(async move { s.infer(id, inp).await }));
    }
    let mut ok = 0usize;
    for h in handles {
        if matches!(h.await, Ok(Ok(_))) {
            ok += 1;
        }
    }
    drop(submitter);
    let elapsed = start.elapsed();
    let rps = ok as f64 / elapsed.as_secs_f64();
    info!(
        "[engine] {ok}/{requests} succeeded in {:.2?} ({rps:.0} req/s)",
        elapsed
    );
}

fn pick_async_backend(choice: BackendChoice) -> Arc<dyn Backend + Send + Sync> {
    #[cfg(target_os = "macos")]
    if choice != BackendChoice::Cpu {
        if let Some(metal) = make_metal() {
            return metal;
        }
    }
    let _ = choice;
    Arc::new(CpuBackend)
}
