"""Download GPT-2 weights and tokenizer with atomic, bounded transfers."""
import argparse
import math
import os
import shutil
import sys
import tempfile
import urllib.request

BASE = "https://huggingface.co/openai-community/gpt2/resolve/main"
FILES = ["model.safetensors", "vocab.json", "merges.txt", "config.json"]
OUT = os.path.join("models", "gpt2")


def positive_timeout(value):
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be a number") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return seconds


def download_file(url, destination, timeout=60.0):
    """Install a completed transfer atomically, preserving any previous asset."""
    directory = os.path.dirname(os.path.abspath(destination))
    descriptor, temporary = tempfile.mkstemp(prefix=".gpt2-", suffix=".part", dir=directory)
    os.close(descriptor)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(temporary, "wb") as output:
            shutil.copyfileobj(response, output)
            expected = response.headers.get("Content-Length")
        size = os.path.getsize(temporary)
        if size == 0:
            raise OSError("downloaded file is empty")
        if expected is not None and size != int(expected):
            raise OSError(f"incomplete transfer: expected {expected} bytes, received {size}")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT, help="asset directory (default: models/gpt2)")
    parser.add_argument("--timeout", type=positive_timeout, default=60.0, help="socket timeout in seconds")
    parser.add_argument("--force", action="store_true", help="replace existing assets")
    args = parser.parse_args(argv)
    try:
        os.makedirs(args.out, exist_ok=True)
        for name in FILES:
            destination = os.path.join(args.out, name)
            if not args.force and os.path.isfile(destination) and os.path.getsize(destination) > 0:
                print(f"  have {name}")
                continue
            print(f"  downloading {name} …")
            download_file(f"{BASE}/{name}", destination, args.timeout)
    except (OSError, ValueError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1
    print(f"Done. Weights in {args.out}/")
    print(f'Try: cargo run --release --bin generate -- --model-dir "{args.out}" --prompt "Once upon a time"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
