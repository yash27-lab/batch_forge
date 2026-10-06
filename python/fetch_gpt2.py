"""Download GPT-2 (124M) weights + tokenizer into models/gpt2/.

Pulls the public HuggingFace `openai-community/gpt2` files (~550 MB) with no
extra dependencies beyond the standard library. These files are gitignored.
"""

import os
import tempfile
import urllib.request

BASE = "https://huggingface.co/openai-community/gpt2/resolve/main"
FILES = ["model.safetensors", "vocab.json", "merges.txt", "config.json"]
OUT = os.path.join("models", "gpt2")


def download_file(url, destination):
    """Install a completed transfer atomically, preserving any previous asset."""
    directory = os.path.dirname(os.path.abspath(destination))
    descriptor, temporary = tempfile.mkstemp(prefix=".gpt2-", suffix=".part", dir=directory)
    os.close(descriptor)
    try:
        urllib.request.urlretrieve(url, temporary)
        if os.path.getsize(temporary) == 0:
            raise OSError("downloaded file is empty")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in FILES:
        dst = os.path.join(OUT, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  have {name}")
            continue
        print(f"  downloading {name} …")
        download_file(f"{BASE}/{name}", dst)
    print(f"Done. Weights in {OUT}/")
    print('Try:  cargo run --release --bin generate -- --prompt "Once upon a time"')


if __name__ == "__main__":
    main()
