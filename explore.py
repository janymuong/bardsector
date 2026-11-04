#!/usr/bin/env python3
"""Compare tiny char-model shapes: bits/char vs weight bytes.

Models are trained exactly (full batch over n-gram counts) with
quantization-aware training (straight-through rounding), so the reported loss
is for the integer weights the boot sector would actually use.
"""
import sys
import time

import numpy as np

ALPHA = " abcdefghijklmnopqrstuvwxyz"
V = len(ALPHA)
CORPUS = "data/shakespeare.txt"


def load():
    text = open(CORPUS).read().lower().replace("\n", " ")
    m = {c: i for i, c in enumerate(ALPHA)}
    out = []
    for c in text:
        if c not in m:
            continue  # punctuation is dropped ("o'er" -> "oer")
        if c == " " and out and out[-1] == 0:
            continue  # collapse runs of spaces
        out.append(m[c])
    return np.array(out, np.int64)


def count_tensor(seq):
    n2 = np.zeros((V, V, V))  # [older char, previous char, next char]
    np.add.at(n2, (seq[:-2], seq[1:-1], seq[2:]), 1)
    return n2


def train(n2, ctx, k, wbits=4, relu=True, steps=3000, lr=0.06, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = -(2 ** (wbits - 1)), 2 ** (wbits - 1) - 1
    q = lambda x: np.clip(np.round(x), lo, hi)
    N = n2 if ctx == 2 else n2.sum(0, keepdims=True)  # (A,B,V)
    T = N.sum()
    nctx = N.sum(-1, keepdims=True)
    scale = 2.0 ** (wbits - 2) / 2
    P = {
        "E1": rng.normal(0, scale, (V, k)),
        "W": rng.normal(0, scale, (V, k)),
        "a": np.array(0.1),
    }
    if ctx == 2:
        P["E2"] = rng.normal(0, scale, (V, k))
    m = {n: np.zeros_like(v) for n, v in P.items()}
    s = {n: np.zeros_like(v) for n, v in P.items()}
    for t in range(1, steps + 1):
        E1, W = q(P["E1"]), q(P["W"])
        pre = E1[None, :, :] if ctx == 1 else q(P["E2"])[:, None, :] + E1[None, :, :]
        h = np.maximum(pre, 0) if relu else pre
        z = h @ W.T
        lg = P["a"] * z
        lg = lg - lg.max(-1, keepdims=True)
        p = np.exp(lg)
        p /= p.sum(-1, keepdims=True)
        loss = -(N * np.log(p + 1e-12)).sum() / T / np.log(2)
        dlog = (p * nctx - N) / T
        G = {"a": np.array((dlog * z).sum())}
        dz = dlog * P["a"]
        G["W"] = np.einsum("abj,abk->jk", dz, h)
        dh = dz @ W
        dpre = dh * (pre > 0) if relu else dh
        G["E1"] = dpre.sum(0)
        if ctx == 2:
            G["E2"] = dpre.sum(1)
        for n in P:
            m[n] = 0.9 * m[n] + 0.1 * G[n]
            s[n] = 0.999 * s[n] + 0.001 * G[n] ** 2
            step = lr * (m[n] / (1 - 0.9**t)) / (np.sqrt(s[n] / (1 - 0.999**t)) + 1e-8)
            if n == "a":
                step *= 0.02
            P[n] = P[n] - step
            if n != "a":
                P[n] = np.clip(P[n], lo - 0.49, hi + 0.49)
    return loss, P


def sample(P, ctx, relu, n=300, seed=1):
    rng = np.random.default_rng(seed)
    q = lambda x: np.clip(np.round(x), -8, 7)
    c1, c2, out = 0, 0, []
    for _ in range(n):
        pre = q(P["E1"][c1]) + (q(P["E2"][c2]) if ctx == 2 else 0)
        h = np.maximum(pre, 0) if relu else pre
        lg = P["a"] * (q(P["W"]) @ h)
        p = np.exp(lg - lg.max())
        p /= p.sum()
        nx = rng.choice(V, p=p)
        out.append(ALPHA[nx])
        c2, c1 = c1, nx
    return "".join(out)


def main():
    seq = load()
    n2 = count_tensor(seq)
    n1 = n2.sum(0)
    bits = lambda N, Pr: -(N * np.log2(Pr + 1e-12)).sum() / N.sum()
    uni = n2.sum((0, 1))
    print(f"corpus: {len(seq)} chars, {V} symbols")
    print(f"baselines (bits/char): uniform {np.log2(V):.2f} | unigram {bits(uni, uni / uni.sum()):.2f} "
          f"| full bigram {bits(n1, n1 / n1.sum(1, keepdims=True)):.2f} "
          f"| full trigram {bits(n2, (n2 + 1e-3) / (n2 + 1e-3).sum(2, keepdims=True)):.2f}")
    configs = [
        # name, ctx, k, wbits, relu
        ("bigram  K=8 4bit relu", 1, 8, 4, True),
        ("bigram  K=8 4bit lin ", 1, 8, 4, False),
        ("bigram  K=4 8bit relu", 1, 4, 8, True),
        ("ctx2    K=4 4bit relu", 2, 4, 4, True),
        ("ctx2    K=5 4bit relu", 2, 5, 4, True),
        ("ctx2    K=6 4bit relu", 2, 6, 4, True),
    ]
    sel = sys.argv[1:] or None
    for name, ctx, k, wb, relu in configs:
        if sel and name.split()[0] not in sel:
            continue
        t0 = time.time()
        loss, P = train(n2, ctx, k, wb, relu)
        nbytes = (ctx + 1) * V * k * wb // 8
        print(f"\n{name}: {loss:.3f} bits/char, weights {nbytes} B ({time.time() - t0:.0f}s)")
        print(repr(sample(P, ctx, relu, 200)))


if __name__ == "__main__":
    main()
