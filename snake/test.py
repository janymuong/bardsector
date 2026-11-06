#!/usr/bin/env python3
"""Run ./snake headless and check it against the Python model in ref.py.

For each (generations, seed): same stat lines, same "saved" message, and a
champion.bin with exactly the bytes the model predicts.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import ref

HERE = Path(__file__).resolve().parent
SNAKE = HERE / "snake"
CASES = [(4, 1), (3, 7), (3, 123456789), (2, 0)]


def main():
    bad = 0
    for gens, seed in CASES:
        want_lines, want_champ = ref.run(gens, seed)
        want = "\n".join(want_lines) + "\nsaved champion.bin\n"
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([str(SNAKE), str(gens), str(seed), "0"], cwd=d,
                               capture_output=True, text=True, timeout=300)
            got_champ = (Path(d) / "champion.bin").read_bytes() if (Path(d) / "champion.bin").exists() else b""
        ok = r.returncode == 0 and r.stdout == want and got_champ == want_champ
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {gens} generations, seed {seed}: "
              f"{r.stdout.splitlines()[-2] if r.stdout else r.stderr.strip()}")
        if not ok:
            for a, b in zip(r.stdout.splitlines(), want.splitlines()):
                if a != b:
                    print(f"     asm: {a}\n     ref: {b}")
                    break
            if got_champ != want_champ:
                print("     champion.bin differs")
    print("ALL MATCH" if not bad else f"{bad} MISMATCHED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
