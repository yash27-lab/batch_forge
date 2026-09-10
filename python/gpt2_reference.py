"""NumPy reference forward for GPT-2 — ground truth for verifying the Rust engine.

Loads the same safetensors weights and computes next-token logits for a fixed
token sequence. Prints the top predictions and dumps the full last-position
logits to reference_logits.npy so the Rust side can assert numerical parity.
"""

import sys

import numpy as np
from safetensors.numpy import load_file

W = load_file("models/gpt2/model.safetensors")
NH, HD = 12, 64


def ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def forward(tokens):
    T = len(tokens)
    x = W["wte.weight"][tokens] + W["wpe.weight"][:T]
    for i in range(12):
        p = f"h.{i}."
        a = ln(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        qkv = a @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        heads = lambda m: m.reshape(T, NH, HD).transpose(1, 0, 2)
        qh, kh, vh = heads(q), heads(k), heads(v)
        att = qh @ kh.transpose(0, 2, 1) / np.sqrt(HD)
        att = att + np.triu(np.ones((T, T)), 1) * -1e10
        o = softmax(att) @ vh
        o = o.transpose(1, 0, 2).reshape(T, NH * HD)
        x = x + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        m = ln(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        h = gelu(m @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
        x = x + h @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    x = ln(x, W["ln_f.weight"], W["ln_f.bias"])
    return x @ W["wte.weight"].T  # [T, vocab]


if __name__ == "__main__":
    # "The meaning of life is"
    tokens = [int(t) for t in (sys.argv[1:] or ["464", "3616", "286", "1204", "318"])]
    logits = forward(tokens)
    last = logits[-1]
    top = np.argsort(last)[-5:][::-1]
    print("tokens:", tokens)
    print("top-5 next ids:", top.tolist())
    print("top-5 logits:", [round(float(last[i]), 3) for i in top])
    print("logit[0] ('!'):", round(float(last[0]), 3))
    np.save("models/gpt2/reference_logits.npy", last.astype(np.float32))
    print("saved last-position logits -> models/gpt2/reference_logits.npy")
