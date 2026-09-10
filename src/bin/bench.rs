//! Microbenchmarks: CPU reference vs Metal for matmul, GELU, and MLP forward.
//!
//! Numbers are produced on *your* machine — there are no hard-coded results.
//! Metal timings are end-to-end (buffer allocation + dispatch + readback);
//! on Apple Silicon's unified memory there is no discrete host↔device copy, but
//! per-call allocation overhead is included and dominates at small sizes.
//!
//! Run with: `cargo run --release --bin bench`

use std::time::{Duration, Instant};

use batch_forge::model::{CpuBackend, Mlp};
use batch_forge::ops;
use batch_forge::tensor::Tensor;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
}

/// Times `f` over `iters` runs after a warmup, returning mean duration.
fn bench(warmup: usize, iters: usize, mut f: impl FnMut()) -> Duration {
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed() / iters as u32
}

fn gflops(m: usize, k: usize, n: usize, d: Duration) -> f64 {
    (2.0 * m as f64 * k as f64 * n as f64) / d.as_secs_f64() / 1e9
}

fn main() {
    println!("batch_forge bench — {}", std::env::consts::ARCH);

    #[cfg(target_os = "macos")]
    let metal = batch_forge::metal_backend::MetalBackend::new(batch_forge::SHADER_SOURCE).ok();
    #[cfg(target_os = "macos")]
    if let Some(m) = &metal {
        println!("Metal device: {}\n", m.device.name());
    }

    let mut rng = Rng::new(0xBEEF);

    // ---- matmul: CPU vs naive Metal vs tiled Metal ----
    println!("== matmul (square, f32): CPU vs naive Metal vs tiled Metal ==");
    println!(
        "{:>6} | {:>9} | {:>9} {:>8} | {:>9} {:>8} | {:>9}",
        "N", "cpu GF/s", "naive ms", "GF/s", "tiled ms", "GF/s", "tiled/naive"
    );
    for n in [128usize, 256, 512, 1024] {
        let a = rng.vec(n * n);
        let b = rng.vec(n * n);
        let cpu_iters = if n >= 512 { 2 } else { 5 };
        let cpu = bench(1, cpu_iters, || {
            std::hint::black_box(ops::matmul(&a, &b, n, n, n));
        });
        #[cfg(target_os = "macos")]
        {
            if let Some(m) = metal.as_ref() {
                let gpu_iters = if n >= 512 { 15 } else { 50 };
                let naive = bench(3, gpu_iters, || {
                    std::hint::black_box(m.matmul(&a, &b, n, n, n));
                });
                let tiled = bench(3, gpu_iters, || {
                    std::hint::black_box(m.matmul_tiled(&a, &b, n, n, n));
                });
                println!(
                    "{n:>6} | {:>9.1} | {:>9.3} {:>8.1} | {:>9.3} {:>8.1} | {:>8.1}x",
                    gflops(n, n, n, cpu),
                    naive.as_secs_f64() * 1e3,
                    gflops(n, n, n, naive),
                    tiled.as_secs_f64() * 1e3,
                    gflops(n, n, n, tiled),
                    naive.as_secs_f64() / tiled.as_secs_f64(),
                );
                continue;
            }
        }
        println!(
            "{n:>6} | {:>9.1} | {:>9} {:>8} | {:>9} {:>8} | {:>9}",
            gflops(n, n, n, cpu),
            "-",
            "-",
            "-",
            "-",
            "-"
        );
    }

    // ---- GELU ----
    println!("\n== gelu (elementwise) ==");
    let n = 1 << 20;
    let x = rng.vec(n);
    let cpu = bench(1, 20, || {
        let mut y = x.clone();
        ops::gelu_inplace(&mut y);
        std::hint::black_box(y);
    });
    println!("{:>10} elems | cpu {:>8.3} ms", n, cpu.as_secs_f64() * 1e3);
    #[cfg(target_os = "macos")]
    if let Some(m) = &metal {
        let gpu = bench(3, 50, || {
            std::hint::black_box(m.gelu(&x));
        });
        println!(
            "{:>10} elems | metal {:>6.3} ms ({:.1}x)",
            n,
            gpu.as_secs_f64() * 1e3,
            cpu.as_secs_f64() / gpu.as_secs_f64()
        );
    }

    // ---- MLP forward ----
    println!("\n== MLP forward (256→1024→1024→256) ==");
    let model = random_mlp(&mut rng, 256, 1024);
    println!("{:>6} | {:>12} | {:>12}", "batch", "cpu (ms)", "metal (ms)");
    for batch in [1usize, 8, 32] {
        let input = Tensor::new(rng.vec(batch * 256), vec![batch, 256]).unwrap();
        let cpu = bench(1, 10, || {
            std::hint::black_box(model.forward(&CpuBackend, &input).unwrap());
        });
        let cpu_ms = cpu.as_secs_f64() * 1e3;
        #[cfg(target_os = "macos")]
        {
            if let Some(m) = &metal {
                let gpu = bench(3, 50, || {
                    std::hint::black_box(model.forward(m, &input).unwrap());
                });
                println!(
                    "{batch:>6} | {cpu_ms:>12.3} | {:>12.3}",
                    gpu.as_secs_f64() * 1e3
                );
                continue;
            }
        }
        println!("{batch:>6} | {cpu_ms:>12.3} | {:>12}", "-");
    }
}

fn random_mlp(rng: &mut Rng, width: usize, hidden: usize) -> Mlp {
    let dims = [(hidden, width), (hidden, hidden), (width, hidden)];
    let layers = dims
        .iter()
        .map(|&(out_f, in_f)| {
            let scale = 1.0 / (in_f as f32).sqrt();
            let w: Vec<f32> = rng.vec(out_f * in_f).iter().map(|v| v * scale).collect();
            let b = vec![0.0f32; out_f];
            (
                Tensor::new(w, vec![out_f, in_f]).unwrap(),
                Tensor::new(b, vec![out_f]).unwrap(),
            )
        })
        .collect();
    Mlp { layers }
}
