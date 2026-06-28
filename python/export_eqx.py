"""Export an Equinox MLP to safetensors with named weights and a reference I/O pair.

The Rust engine loads weights by name (`layers.{i}.weight` / `layers.{i}.bias`),
so this exporter walks the model's `Linear` layers and names them explicitly
instead of emitting opaque `leaf_{i}` keys. It also writes a `reference.safetensors`
containing a sample `input` and the model's `output`, which `batch_forge --verify`
uses to confirm the Rust forward pass matches JAX numerically.
"""

import argparse

import equinox as eqx
import jax
import numpy as np
from safetensors.numpy import save_file


class SimpleMLP(eqx.Module):
    layers: list

    def __init__(self, key, width=256, hidden=1024):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(width, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
            eqx.nn.Linear(hidden, width, key=k3),
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            # approximate=True (tanh GELU) matches batch_forge::ops::gelu.
            x = jax.nn.gelu(layer(x), approximate=True)
        return self.layers[-1](x)


def export_weights(model: SimpleMLP, path: str):
    tensors = {}
    for i, layer in enumerate(model.layers):
        # eqx.nn.Linear stores weight as [out, in] and bias as [out].
        tensors[f"layers.{i}.weight"] = np.asarray(layer.weight, dtype=np.float32)
        tensors[f"layers.{i}.bias"] = np.asarray(layer.bias, dtype=np.float32)
        print(f"  layers.{i}: weight {tensors[f'layers.{i}.weight'].shape}")
    save_file(tensors, path)
    print(f"Wrote weights -> {path}")


def export_reference(model: SimpleMLP, path: str, seed: int):
    width = model.layers[0].weight.shape[1]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(width).astype(np.float32)
    y = np.asarray(jax.vmap(model)(x[None, :]), dtype=np.float32)
    save_file({"input": x[None, :], "output": y}, path)
    print(f"Wrote reference (input {x[None, :].shape} -> output {y.shape}) -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export an Equinox MLP to safetensors.")
    parser.add_argument("--out", default="model.safetensors", help="weights output path")
    parser.add_argument("--ref", default="reference.safetensors", help="reference I/O output path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = SimpleMLP(jax.random.PRNGKey(args.seed))
    export_weights(model, args.out)
    export_reference(model, args.ref, args.seed)
    print("Done. Verify with:  cargo run --release -- --verify", args.ref)
