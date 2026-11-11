#!/usr/bin/env python3
"""Boot bardsector.bin in the Unicorn CPU emulator and check it against a Python model.

  emu.py size      show how many bytes of the sector are used / free
  emu.py test      boot the image with a fake BIOS and compare, char for char,
                   with an independent integer re-implementation of the net
  emu.py demo [N] [keys]
                   print N characters of what the sector writes (default 600)

The fake BIOS is an interrupt table pointing at tiny `iret` stubs; a code hook on
each stub plays the part of the BIOS (teletype output, keyboard, tick counter).
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIN = HERE / "bardsector.bin"
NW = 243  # weight bytes at the end of the sector, before the 55AA signature
BANNER = b"TO BOOT, OR NOT TO BOOT:\r\n"


# --------------------------------------------------------------- reference --
def reference(flat, nchars, seed, keys=b""):
    """Same arithmetic as the boot sector, in plain Python.

    flat = E1 (81 B) | E2 (81 B) | W (81 B, rows stored z..space), read straight
    out of the assembled image so the weights are exactly the shipped ones.
    The noise added to every logit is r/1024 + r/2048 (1.5x a 6-bit value).
    """
    c1 = c2 = 0  # symbols; a typed key can make them 27..31, which just reads
    rng = seed & 0xFFFF  # on into the next table, exactly like the real thing
    pending = list(keys)
    out = bytearray()
    for _ in range(nchars):
        h = []
        for m in range(3):
            b1, b2 = flat[3 * c1 + m], flat[81 + 3 * c2 + m]
            h.append(max((b1 & 15) + (b2 & 15) - 16, 0))
            h.append(max((b1 >> 4) + (b2 >> 4) - 16, 0))
        best = 0
        for s in range(26, -1, -1):
            row = 162 + 3 * (26 - s)
            acc = 0
            for m in range(3):
                b = flat[row + m]
                acc += (b & 15) * h[2 * m] + (b >> 4) * h[2 * m + 1]
            rng = (rng * 25173 + 13849) & 0xFFFF
            key = ((((rng >> 10) + (rng >> 11) + acc) << 5) | s) & 0xFFFF
            if key > best:
                best = key
        sym = best & 31
        if pending:
            sym = pending.pop(0) & 31
        out.append(0x20 if sym == 0 else (0x60 + sym) & 0xFF)
        c2, c1 = c1, sym
    return bytes(out)


# ---------------------------------------------------------------- emulator --
def boot(nchars, keys=b"", ticks=0x1234, image=BIN):
    from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_HOOK_INTR, UC_MODE_16, Uc
    from unicorn.x86_const import (UC_X86_REG_AX, UC_X86_REG_CX, UC_X86_REG_DX,
                                   UC_X86_REG_EFLAGS, UC_X86_REG_EIP)

    mu = Uc(UC_ARCH_X86, UC_MODE_16)
    mu.mem_map(0, 0x10000)
    mu.mem_write(0x7C00, Path(image).read_bytes())
    out, queue, target = bytearray(), list(keys), len(BANNER) + nchars
    stats = {"hlt": 0}

    def set_zf(uc, on):
        f = uc.reg_read(UC_X86_REG_EFLAGS)
        uc.reg_write(UC_X86_REG_EFLAGS, (f | 0x40) if on else (f & ~0x40))

    def bios(uc, intno, _):
        ax = uc.reg_read(UC_X86_REG_AX)
        ah = ax >> 8
        if intno == 0x10 and ah == 0x0E:  # teletype output
            out.append(ax & 0xFF)
            if len(out) >= target:
                uc.emu_stop()
        elif intno == 0x16:  # keyboard
            if ah == 0:
                assert queue, "program blocked reading a key that is not there"
                uc.reg_write(UC_X86_REG_AX, queue.pop(0))
            elif ah == 1:
                if queue:
                    uc.reg_write(UC_X86_REG_AX, queue[0])
                set_zf(uc, not queue)
        elif intno == 0x1A and ah == 0:  # tick counter
            uc.reg_write(UC_X86_REG_CX, 0)
            uc.reg_write(UC_X86_REG_DX, ticks)

    mu.hook_add(UC_HOOK_INTR, bios)

    # Unicorn stays halted after HLT, so let the "timer tick" wake it by hopping
    # over the instruction (HLT is followed by the loop's `jmp gen`: F4 E9).
    hlt_at = 0x7C00 + Path(image).read_bytes().index(b"\xF4\xE9")

    def tick(uc, addr, size, _):
        stats["hlt"] += 1
        uc.reg_write(UC_X86_REG_EIP, addr + 1)

    mu.hook_add(UC_HOOK_CODE, tick, begin=hlt_at, end=hlt_at)
    mu.emu_start(0x7C00, 0x20000, timeout=60_000_000)
    assert len(out) >= target, "program stopped early"
    assert bytes(out[: len(BANNER)]) == BANNER, f"banner mismatch: {bytes(out[:40])!r}"
    return bytes(out[len(BANNER):]), stats


# ------------------------------------------------------------------- cmds --
def cmd_size():
    data = BIN.read_bytes()
    assert len(data) == 512 and data[510:] == b"\x55\xaa", "not a valid boot sector"
    wstart = 510 - NW
    free = 0
    while data[wstart - 1 - free] == 0:
        free += 1
    print(f"boot sector: code+banner {wstart - free} B | free {free} B | "
          f"weights {NW} B | signature 2 B  (= 512)")


def cmd_test():
    flat = BIN.read_bytes()[510 - NW : 510]
    cases = [
        ("free-running, seed 0x1234", b"", 0x1234, 500),
        ("free-running, seed 0xBEEF", b"", 0xBEEF, 500),
        ("typed 'to be or not to be'", b"to be or not to be", 0x1234, 300),
        ("shouting + punctuation", b"HELLO, World! 123", 0x0001, 200),
        ("every key 0..127 once", bytes(range(128)), 0x7777, 200),
    ]
    bad = 0
    for name, keys, seed, n in cases:
        got, stats = boot(n, keys, seed)
        want = reference(flat, n, seed, keys)
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name} ({n} chars, {stats['hlt']} hlt)")
        if not ok:
            i = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
            print(f"     first difference at char {i}: got {got[i:i+30]!r} want {want[i:i+30]!r}")
        elif name.startswith("free-running, seed 0x1234"):
            print("     " + got[:200].decode("latin-1"))
    print("ALL MATCH" if not bad else f"{bad} MISMATCHED")
    sys.exit(1 if bad else 0)


def cmd_demo(args):
    n = int(args[0]) if args else 600
    keys = args[1].encode() if len(args) > 1 else b""
    got, _ = boot(n, keys, 0x1234)
    text = got.decode("latin-1")
    for i in range(0, len(text), 80):  # wrap like the 80-column screen
        print(text[i : i + 80])


if __name__ == "__main__":
    cmds = {"size": lambda a: cmd_size(), "test": lambda a: cmd_test(), "demo": cmd_demo}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]](sys.argv[2:])
