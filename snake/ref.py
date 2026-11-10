#!/usr/bin/env python3
"""Integer-exact Python model of snake.asm: the game, the net and the evolution.

This file is the spec. snake.asm must print exactly the same stat lines, write
exactly the same champion.bin and print the same `eval` line for the same inputs.

  ref.py [generations=500] [seed=1]
"""
import sys

# ---- hyper-parameters (keep in sync with the %defines in snake.asm) ---------
NHID = 8          # hidden units
RAYS = 1          # 1 = three extra look-ahead senses
RAYBIN = 1        # 1: sense is 0/1 "obstacle within RAY_CAP cells"; 0: free cells
RAY_CAP = 2       # how far the look-ahead senses look
NGAMES = 3        # games each genome plays per generation (same games for all)
MUT_MASK = 15     # a byte mutates when (rand & MUT_MASK) == 0  -> 1/16
ELITE = 8         # top genomes copied unchanged into the next generation
PARENTS = 16      # children are bred from the top PARENTS genomes
POP = 64          # genomes per generation
HUNGER = 100      # a snake that goes this many steps without eating dies
SCORE_CAP = 200   # stop a game at this many apples
TEST_GAMES = 40   # fresh games used to re-test candidates
RETEST_TOP = 4    # how many of a generation's best are re-tested
TEST_STATE = 0x2545F491
EVAL_MAX = 4096

# ---- genome layout (signed bytes) -------------------------------------------
#   W1[NHID][NIN] | B1[NHID] | W2[3][NHID] | B2[3] | padding up to GSZ
NIN = 6 + 3 * RAYS
OFF_B1 = NHID * NIN
OFF_W2 = OFF_B1 + NHID
OFF_B2 = OFF_W2 + 3 * NHID
GUSED = OFF_B2 + 3
GSZ = next(1 << sh for sh in (6, 7, 8, 9) if GUSED <= 1 << sh)

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


def fill_seeds(state, n):
    """n game seeds from a private xorshift32 stream (asm: fill_seeds)."""
    r = Rng(state)
    return [r.next() for _ in range(n)]


# Inputs: danger left/straight/right, food left/ahead/right,
#         then (RAYS) look-ahead left/straight/right.
def act(g, x):
    h = []
    for j in range(NHID):
        a = g[OFF_B1 + j] + sum(g[j * NIN + i] * x[i] for i in range(NIN))
        h.append(a if a > 0 else 0)
    o = [g[OFF_B2 + k] + sum(g[OFF_W2 + k * NHID + j] * h[j] for j in range(NHID)) for k in range(3)]
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

    def blocked(x, y):
        return x < 0 or x > 15 or y < 0 or y > 15 or occ[y * 16 + x]

    food = new_food()
    while True:
        x, y = pos & 15, pos >> 4
        inp = [0] * NIN
        for k in range(3):
            nd = (d + k + 3) & 3
            inp[k] = 1 if blocked(x + DX[nd], y + DY[nd]) else 0
        ddx, ddy = (food & 15) - x, (food >> 4) - y
        fw = ddx * DX[d] + ddy * DY[d]
        rt = ddx * DX[(d + 1) & 3] + ddy * DY[(d + 1) & 3]
        inp[3], inp[4], inp[5] = int(rt < 0), int(fw > 0), int(rt > 0)
        if RAYS:
            for k in range(3):
                nd = (d + k + 3) & 3
                rx, ry, dist = x, y, 0
                while True:
                    rx, ry = rx + DX[nd], ry + DY[nd]
                    if blocked(rx, ry):
                        break
                    dist += 1
                    if dist >= RAY_CAP:
                        break
                inp[6 + k] = int(dist < RAY_CAP) if RAYBIN else dist

        d = (d + act(g, inp) + 3) & 3
        nx, ny = x + DX[d], y + DY[d]
        if blocked(nx, ny):
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


def evaluate(g, seeds):
    """Play `g` on every seed. Returns (total apples, fewest, most)."""
    apples = [play(g, s)[1] for s in seeds]
    return sum(apples), min(apples), max(apples)


def tenths(total, games):
    t = total * 10 // games
    return f"{t // 10}.{t % 10}"


def eval_line(genome, name, games, seed):
    """The line `snake eval <name> <games> <seed>` prints."""
    games = max(1, min(games, EVAL_MAX))
    seed = seed or 1
    g = [b - 256 if b > 127 else b for b in genome]
    total, lo, hi = evaluate(g, fill_seeds(seed, games))
    return f"eval {name}: {games} games, apples mean {tenths(total, games)} min {lo} max {hi}"


def run(gens, seed, log=None):
    """Evolve for `gens` generations. Returns (stat lines, champion bytes)."""
    rng = Rng(seed or 1)
    tseeds = fill_seeds(TEST_STATE, TEST_GAMES)
    pop = []
    for _ in range(POP):
        pop.append([0 if j >= GUSED else ((rng.next() >> 8) & 63) - 32 for j in range(GSZ)])
    lines, champ, champ_total = [], None, -1
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
        # hall of fame: re-test this generation's best on the fixed fresh games
        for r in range(RETEST_TOP):
            cand = pop[keys[r] & 255]
            total = evaluate(cand, tseeds)[0]
            if total > champ_total:
                champ_total, champ = total, list(cand)
        best, bi = keys[0] >> 8, keys[0] & 255
        line = (f"gen {gen} best {best} mean {sum(fit) >> 6} apples {apl[bi]} "
                f"champ {tenths(champ_total, TEST_GAMES)}")
        lines.append(line)
        if log:
            log(line)
        if gen == gens:
            break
        new = [list(pop[k & 255]) for k in keys[:ELITE]]
        for _ in range(ELITE, POP):
            r = rng.next()
            child = list(pop[keys[(r >> 8) & (PARENTS - 1)] & 255])
            for j in range(GUSED):
                r = rng.next()
                if r & MUT_MASK == 0:
                    child[j] = max(-128, min(127, child[j] + ((r >> 8) & 15) - 8))
            new.append(child)
        pop = new
    return lines, bytes(b & 255 for b in champ)


if __name__ == "__main__":
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run(gens, seed, log=lambda s: print(s, flush=True))
    print("saved champion.bin")
