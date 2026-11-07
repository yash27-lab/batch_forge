"""Generate a demo model + reference WITHOUT JAX (NumPy only).

This mirrors `export_eqx.py`'s SimpleMLP exactly — same layer shapes, same
`y = x @ Wᵀ + b` convention, same tanh-GELU — so the Rust engine's output matches
the `output` tensor written here. It exists so the end-to-end demo and CI can run
on machines without the (heavy) JAX/Equinox stack installed.

    python python/make_demo_model.py
    cargo run --release -- --verify reference.safetensors
"""

import argparse

import numpy as np
from safetensors.numpy import save_file

SQRT_2_OVER_PI = 0.7978845608


def gelu(x):
    # Tanh approximation; matches batch_forge::ops::gelu and jax.nn.gelu(approximate=True).
    return 0.5 * x * (1.0 + np.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x**3)))


def linear(x, w, b):
    # eqx.nn.Linear convention: w is [out, in], so y = x @ wᵀ + b.
    return x @ w.T + b


def main():
    parser = argparse.ArgumentParser(description="NumPy-only demo model generator.")
    parser.add_argument("--out", default="model.safetensors")
    parser.add_argument("--ref", default="reference.safetensors")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=1024)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dims = [
        (args.hidden, args.width),
        (args.hidden, args.hidden),
        (args.width, args.hidden),
    ]

    tensors = {}
    weights = []
    for i, (out_f, in_f) in enumerate(dims):
        # Kaiming-ish init so activations stay in a sane range.
        w = (rng.standard_normal((out_f, in_f)) / np.sqrt(in_f)).astype(np.float32)
        b = np.zeros(out_f, dtype=np.float32)
        tensors[f"layers.{i}.weight"] = w
        tensors[f"layers.{i}.bias"] = b
        weights.append((w, b))
        print(f"  layers.{i}: weight {w.shape}, bias {b.shape}")
    save_file(tensors, args.out)
    print(f"Wrote weights -> {args.out}")

    # Reference forward (matches the Rust/JAX forward exactly).
    x = rng.standard_normal((1, args.width)).astype(np.float32)
    h = x
    for i, (w, b) in enumerate(weights):
        h = linear(h, w, b)
        if i < len(weights) - 1:
            h = gelu(h)
    save_file({"input": x, "output": h.astype(np.float32)}, args.ref)
    print(f"Wrote reference (input {x.shape} -> output {h.shape}) -> {args.ref}")
    print("Verify with:  cargo run --release -- --verify", args.ref)


if __name__ == "__main__":
    main()
