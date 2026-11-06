#!/usr/bin/env python3
"""Integer-exact Python model of snake.asm: the game, the net and the evolution.

This file is the spec. snake.asm must print exactly the same stat lines and write
exactly the same champion.bin for the same (generations, seed).

  ref.py [generations=200] [seed=1]
"""
import sys

# ---- hyper-parameters (keep in sync with the %defines in snake.asm) ---------
POP = 64          # genomes per generation
GSZ = 64          # bytes per genome (63 used + 1 pad)
NGAMES = 3        # games each genome plays per generation (same games for all)
HUNGER = 100      # a snake that goes this many steps without eating dies
SCORE_CAP = 200   # stop a game at this many apples
ELITE = 8         # top genomes copied unchanged into the next generation
PARENTS = 16      # children are bred from the top PARENTS genomes
MUT_MASK = 15     # a byte mutates when (rand & MUT_MASK) == 0  -> 1/16

M32 = 0xFFFFFFFF
DX = (0, 1, 0, -1)   # N E S W
DY = (-1, 0, 1, 0)


class Rng:
    """xorshift32, the same three shifts as the assembly."""

    def __init__(self, seed):
        self.x = seed & M32

    def next(self):
        x = self.x
        x ^= (x << 13) & M32
        x ^= x >> 17
        x ^= (x << 5) & M32
        self.x = x
        return x


# Genome layout (signed bytes):
#   0..35   W1[hidden 0..5][input 0..5]      36..41  B1[hidden]
#   42..59  W2[output 0..2][hidden 0..5]     60..62  B2[output]     63 unused
# Inputs: danger left, danger straight, danger right, food left, food ahead, food right
def act(g, x):
    h = []
    for j in range(6):
        a = g[36 + j] + sum(g[j * 6 + i] * x[i] for i in range(6))
        h.append(a if a > 0 else 0)
    o = [g[60 + k] + sum(g[42 + k * 6 + j] * h[j] for j in range(6)) for k in range(3)]
    best, bv = 1, o[1]  # ties favour going straight
    if o[0] > bv:
        best, bv = 0, o[0]
    if o[2] > bv:
        best = 2
    return best  # 0 = turn left, 1 = straight, 2 = turn right


def play(g, seed):
    """One game on the 16x16 board. Returns (fitness, apples)."""
    r = Rng(seed)
    occ = [0] * 256
    ring = [0] * 256
    ring[0:3] = [134, 135, 136]
    occ[134] = occ[135] = occ[136] = 1
    tail, head, pos, d = 0, 3, 136, 1
    score = steps = hunger = 0

    def new_food():
        while True:
            c = (r.next() >> 8) & 255
            if not occ[c]:
                return c

    food = new_food()
    while True:
        x, y = pos & 15, pos >> 4
        inp = [0] * 6
        for k in range(3):
            nd = (d + k + 3) & 3
            nx, ny = x + DX[nd], y + DY[nd]
            inp[k] = 1 if (nx < 0 or nx > 15 or ny < 0 or ny > 15 or occ[ny * 16 + nx]) else 0
        ddx, ddy = (food & 15) - x, (food >> 4) - y
        fw = ddx * DX[d] + ddy * DY[d]
        rt = ddx * DX[(d + 1) & 3] + ddy * DY[(d + 1) & 3]
        inp[3], inp[4], inp[5] = int(rt < 0), int(fw > 0), int(rt > 0)

        d = (d + act(g, inp) + 3) & 3
        nx, ny = x + DX[d], y + DY[d]
        if nx < 0 or nx > 15 or ny < 0 or ny > 15 or occ[ny * 16 + nx]:
            break
        cell = ny * 16 + nx
        steps += 1
        hunger += 1
        ate = cell == food
        if ate:
            score += 1
            hunger = 0
        else:
            occ[ring[tail]] = 0
            tail = (tail + 1) & 255
        occ[cell] = 1
        ring[head] = cell
        head = (head + 1) & 255
        pos = cell
        if ate:
            if score >= SCORE_CAP:
                break
            food = new_food()
        if hunger >= HUNGER:
            break
    return score * 100 + steps, score


def run(gens, seed, log=None):
    """Evolve for `gens` generations. Returns (stat lines, champion bytes)."""
    rng = Rng(seed or 1)
    pop = []
    for _ in range(POP):
        pop.append([0 if j == GSZ - 1 else ((rng.next() >> 8) & 63) - 32 for j in range(GSZ)])
    lines, champ = [], None
    for gen in range(1, gens + 1):
        seeds = [rng.next() for _ in range(NGAMES)]
        fit, apl = [], []
        for g in pop:
            f = a = 0
            for s in seeds:
                ff, sc = play(g, s)
                f, a = f + ff, a + sc
            fit.append(f)
            apl.append(a)
        keys = sorted(((fit[i] << 8) | i for i in range(POP)), reverse=True)
        best, bi = keys[0] >> 8, keys[0] & 255
        line = f"gen {gen} best {best} mean {sum(fit) >> 6} apples {apl[bi]}"
        lines.append(line)
        if log:
            log(line)
        champ = list(pop[bi])
        if gen == gens:
            break
        new = [list(pop[k & 255]) for k in keys[:ELITE]]
        for _ in range(ELITE, POP):
            r = rng.next()
            child = list(pop[keys[(r >> 8) & (PARENTS - 1)] & 255])
            for j in range(GSZ - 1):
                r = rng.next()
                if r & MUT_MASK == 0:
                    child[j] = max(-128, min(127, child[j] + ((r >> 8) & 15) - 8))
            new.append(child)
        pop = new
    return lines, bytes(b & 255 for b in champ)


if __name__ == "__main__":
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run(gens, seed, log=lambda s: print(s, flush=True))
    print("saved champion.bin")
