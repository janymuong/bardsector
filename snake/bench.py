#!/usr/bin/env python3
"""Benchmark a snake build: train several seeds, score each champion on unseen games.

  bench.py ./snake [generations=300] [seed ...]

Each champion is scored with `snake eval` on 300 games from an eval seed that is
unrelated to the training seeds and to the fixed re-test games used in training,
so the numbers are honest (no selection bias).
"""
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EVAL_GAMES, EVAL_SEED = 300, 777


def one(binary, gens, seed):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([binary, str(gens), str(seed), "0"], cwd=d, capture_output=True, check=True)
        out = subprocess.run([binary, "eval", "champion.bin", str(EVAL_GAMES), str(EVAL_SEED)],
                             cwd=d, capture_output=True, text=True, check=True).stdout
    m = re.search(r"mean ([\d.]+) min (\d+) max (\d+)", out)
    return float(m.group(1)), int(m.group(2)), int(m.group(3))


def main():
    binary = str(Path(sys.argv[1]).resolve())
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    seeds = [int(s) for s in sys.argv[3:]] or [1, 2, 3, 4, 5, 6]
    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(lambda s: one(binary, gens, s), seeds))
    means = [r[0] for r in res]
    print(f"{Path(binary).name:<14} per-seed mean apples: " + " ".join(f"{m:5.1f}" for m in means)
          + f"  | average {sum(means) / len(means):5.1f}  worst-seed {min(means):5.1f}")


if __name__ == "__main__":
    main()
