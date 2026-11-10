#!/usr/bin/env python3
"""Check ./snake against the Python model in ref.py.

  training   same stat lines, same "saved" message, same champion.bin bytes
  eval       same `snake eval` line for a trained champion
  play       renders frames; refuses bad files and bad arguments
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import ref

HERE = Path(__file__).resolve().parent
SNAKE = str(HERE / "snake")
CASES = [(4, 1), (3, 7), (3, 123456789), (2, 0)]


def run(args, cwd, timeout=300):
    return subprocess.run([SNAKE, *map(str, args)], cwd=cwd, capture_output=True, text=True, timeout=timeout)


def main():
    bad = 0

    def report(ok, text, detail=""):
        nonlocal bad
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {text}")
        if not ok and detail:
            print("     " + detail.replace("\n", "\n     "))

    for gens, seed in CASES:
        want_lines, want_champ = ref.run(gens, seed)
        want = "\n".join(want_lines) + "\nsaved champion.bin\n"
        with tempfile.TemporaryDirectory() as d:
            r = run([gens, seed, 0], d)
            got_champ = (Path(d) / "champion.bin").read_bytes() if (Path(d) / "champion.bin").exists() else b""
            ok = r.returncode == 0 and r.stdout == want and got_champ == want_champ
            detail = ""
            if not ok:
                for a, b in zip(r.stdout.splitlines(), want.splitlines()):
                    if a != b:
                        detail = f"asm: {a}\nref: {b}"
                        break
                if got_champ != want_champ:
                    detail += "\nchampion.bin differs"
            report(ok, f"train {gens} generations, seed {seed}: {r.stdout.splitlines()[-2] if r.stdout else r.stderr.strip()}", detail)

            if seed == 7:  # eval and play on a trained champion
                for args in ([], [25, 9], [10, 0]):
                    shown = " ".join(map(str, ["eval", "champion.bin", *args]))
                    r = run(["eval", "champion.bin", *args], d)
                    games = args[0] if args else 100
                    want_e = ref.eval_line(want_champ, "champion.bin", games, args[1] if len(args) > 1 else 1) + "\n"
                    report(r.stdout == want_e, shown + f": {r.stdout.strip()}", f"asm: {r.stdout!r}\nref: {want_e!r}")

                try:
                    p = subprocess.run([SNAKE, "play", "champion.bin", "5"], cwd=d, capture_output=True,
                                       text=True, timeout=2)
                    out = p.stdout
                except subprocess.TimeoutExpired as e:  # it plays forever: expected
                    out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
                frames = len(re.findall(r"champion  apples \d+  steps \d+", out))
                report(frames >= 5 and "\x1b[42m" in out and "\x1b[41m" in out,
                       f"play renders frames ({frames} in 2 s, head/body/food colours present)")

                (Path(d) / "short.bin").write_bytes(b"x" * 64)  # an old 64-byte genome
                (Path(d) / "long.bin").write_bytes(b"x" * (ref.GSZ + 1))
                for name in ("short.bin", "long.bin", "missing.bin"):
                    for cmd in ("eval", "play"):
                        r = run([cmd, name], d, timeout=5)
                        report(r.returncode == 1 and r.stderr.startswith("error:"),
                               f"{cmd} {name} is refused: {r.stderr.strip()}")

    for args in (["generations=64"], ["64", "seed=1"], ["eval", "champion.bin", "many"]):
        with tempfile.TemporaryDirectory() as d:
            r = run(args, d, timeout=5)
            report(r.returncode == 1 and r.stderr.startswith("usage:"), f"bad arguments {args} -> usage message")

    print("ALL PASSED" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
