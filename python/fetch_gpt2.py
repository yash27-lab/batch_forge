"""Download GPT-2 (124M) weights + tokenizer into models/gpt2/.

Pulls the public HuggingFace `openai-community/gpt2` files (~550 MB) with no
extra dependencies beyond the standard library. These files are gitignored.
"""

import os
import urllib.request

BASE = "https://huggingface.co/openai-community/gpt2/resolve/main"
FILES = ["model.safetensors", "vocab.json", "merges.txt", "config.json"]
OUT = os.path.join("models", "gpt2")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in FILES:
        dst = os.path.join(OUT, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  have {name}")
            continue
        print(f"  downloading {name} …")
        urllib.request.urlretrieve(f"{BASE}/{name}", dst)
    print(f"Done. Weights in {OUT}/")
    print('Try:  cargo run --release --bin generate -- --prompt "Once upon a time"')


if __name__ == "__main__":
    main()
