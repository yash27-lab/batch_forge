import jax
import jax.numpy as jnp
import equinox as eqx
from safetensors.numpy import save_file
import argparse
import numpy as np

class SimpleMLP(eqx.Module):
    layers: list

    def __init__(self, key):
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(256, 1024, key=keys[0]),
            eqx.nn.Linear(1024, 1024, key=keys[1]),
            eqx.nn.Linear(1024, 256, key=keys[2])
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)

def export_model(model: eqx.Module, output_path: str):
    print(f"Exporting Equinox model to {output_path}...")
    
    # Flatten the PyTree into leaves
    leaves, treedef = jax.tree_util.tree_flatten(model)
    
    # Filter out non-array leaves (e.g., callables, strings) if necessary
    tensor_dict = {}
    for i, leaf in enumerate(leaves):
        if hasattr(leaf, 'shape') and hasattr(leaf, 'dtype'):
            # Convert JAX array to NumPy array for safetensors
            name = f"leaf_{i}"
            tensor_dict[name] = np.array(leaf)
            print(f"Exported {name}: shape {leaf.shape}, dtype {leaf.dtype}")

    save_file(tensor_dict, output_path)
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export an Equinox model to Safetensors.")
    parser.add_argument("--out", type=str, default="../model.safetensors", help="Output path")
    args = parser.parse_args()

    key = jax.random.PRNGKey(42)
    model = SimpleMLP(key)
    
    export_model(model, args.out)
