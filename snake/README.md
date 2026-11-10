# SSSurvival of the Fittest

Neuroevolution Snake in **pure x86-64 Linux assembly**. No libc, no runtime.

A population of 64 tiny neural nets plays Snake. The best survive, mutated copies
replace the rest, and a few hundred generations later the champion eats 50+ apples a
game. There is no backprop and no gradient: only selection. The game, the nets, the
evolution and the terminal renderer all live in `snake.asm`.

```
champion  apples 16  steps 264
+--------------------------------+
|. . . . . . . . . @@oo. . . . . |
|. . . . . . . . . . oo. . . . . |
|. . . **. . . . . . . . . . . . |
...
```

(In a terminal the head is bright green, the body green and the food red.)

## Run it:

```
make                 # build ./snake
make test            # check the assembly against the Python model
make run             # evolve 500 generations, replaying the champion every 100,
                     # then play it forever (Ctrl-C to stop)
make play            # watch the saved champion.bin (a strong one is included)
make train           # headless: 1000 generations, one stats line each (~30 s)
make bench           # train 6 seeds and score each champion (see Results)
```

NOTE: needs a terminal at least 20 lines tall: each frame is 19 lines and is redrawn in place.

## Commands

```
./snake [generations=500] [seed=1] [watch=100]       evolve a population
./snake play [file=champion.bin] [seed]              watch a saved genome play, forever
./snake eval [file=champion.bin] [games=100] [seed=1]   score a saved genome
```

All arguments are plain numbers: `./snake 64 1 20`, not `generations=64`. Anything else
prints the usage message and exits with status 1.

| Argument | Meaning |
|---|---|
| `generations` | how many generations to evolve |
| `seed` | PRNG seed; the same seed always gives the same run. For `play`, omit it for a different game each time |
| `watch` | `0`: headless, print stats and exit. `N`: replay the champion (first 300 steps) every N generations, and play it forever after training |
| `file` | a genome file written by a training run (128 bytes) |
| `games` | how many unseen games `eval` plays (up to 4096) |

Each generation prints one line:

```
gen 150 best 22043 mean 7705 apples 188 champ 51.3
```

`best` and `mean` are fitness over the generation's 3 games (`apples * 100 + steps survived`),
and `apples` is the best genome's total over those 3 games. `champ` is the champion's average
apples per game on a fixed set of 40 fresh games. It can only go up.

`eval` prints, for example:

```
eval champion.bin: 300 games, apples mean 57.4 min 11 max 73
```

Genome files from older builds (64 bytes) are refused with an error, because the net's shape
changed.

## How it works

**The game.** 16x16 board with walls. The snake starts with length 3 heading east, and a
snake that goes 100 steps without eating starves. Food is placed with a per-game xorshift32
PRNG, so a game is fully determined by its seed.

**The senses.** Nine inputs, all 0 or 1:

| Inputs | Meaning |
|---|---|
| 0-2 | wall or body in the next cell to the left / straight / right |
| 3-5 | the food is to the left / ahead / right |
| 6-8 | a wall or body within 2 cells to the left / straight / right (look-ahead) |

**The brain.** 9 inputs, 8 ReLU hidden units, 3 outputs (turn left, go straight, turn right),
the largest output wins and ties go straight. All weights are signed bytes: 72 + 8 + 24 + 3 = 107
bytes, padded to a 128-byte genome.

**The evolution.** Every generation, all 64 genomes play the same 3 games. The top 8 are copied
unchanged, and the other 56 are mutated copies of parents picked at random from the top 16.
Each byte mutates with probability 1/16 by adding a value from -8 to 7 (saturating).

**The champion re-test.** Three games is a noisy judge: a mediocre genome can win a generation by
luck. So the 4 best genomes of every generation are re-tested on a fixed set of 40 other
games, and the best genome ever seen by that re-test is the champion. That is what gets
saved, replayed and printed in `champ`.

## Results

`bench.py` trains several seeds and scores each champion on 300 games that neither training
nor the re-test ever saw. The seed-to-seed spread is large (single seeds range from about
20 to 60 apples), so only compare averages over many seeds.

| Setup | Average apples per unseen game |
|---|---|
| First version: 6 senses, no re-test | about 40 |
| + champion re-test (6 senses, 6 hidden) | 46.6 (seeds 1-8) |
| **Default: + look-ahead senses, 8 hidden, 1000 generations** | **48.4 (seeds 1-8), 43.6 (held-out seeds 9-20)** |
| Same seeds, 6 senses, 1000 generations | 46.3 (seeds 1-8), 38.6 (held-out seeds 9-20) |

Over all 20 seeds that is about **45.5 against 41.7 apples**: the new senses help, but modestly,
and only with enough generations (they were not better at 300). The included `champion.bin`
(seed 1, 1000 generations) is a good run: 57.4 and 56.6 average apples on two sets of 300 unseen
games, best single game 75.

Things that were tried and did not help (average apples, seeds 1-8, 300 generations, against 46.6
for the baseline at the time): the raw number of free cells ahead instead of a 0/1 flag (14 to
40, depending on range), a bigger hidden layer with the old senses (34), 5 games per genome (31),
a lower mutation rate of 1/32 (31), fewer elites (38) or more parents (35). Less noisy selection
and gentler mutation seem to converge too fast to a mediocre snake.

## Build-time knobs

The net and the evolution can be changed without editing the source:

```
make clean && make NASMFLAGS="-DNHID=12 -DRAYBIN=0 -DRAY_CAP=3"
```

| Flag | Default | Meaning |
|---|---|---|
| `NHID` | 8 | hidden units |
| `RAYS` | 1 | `0` removes the three look-ahead senses |
| `RAYBIN` | 1 | `1`: look-ahead is 0/1 "obstacle within `RAY_CAP`"; `0`: number of free cells |
| `RAY_CAP` | 2 | how far the look-ahead senses look |
| `NGAMES` | 3 | games per genome per generation |
| `MUT_MASK` | 15 | a byte mutates when `rand & MUT_MASK == 0` |
| `ELITE`, `PARENTS` | 8, 16 | elites kept, size of the parent pool |

The genome size follows from these automatically. `ref.py` and `test.py` check the default
build; for another build, edit the matching constants at the top of `ref.py`.

## How it is verified

`ref.py` is an integer-exact Python model of the whole thing (game, senses, net, PRNG,
selection, mutation, the champion re-test and `eval`). `test.py` checks that:

- training stat lines and the bytes of `champion.bin` match the model exactly (several seeds),
- `eval` prints the same line as the model, including for a strong champion,
- `play` renders frames, and refuses missing files, wrong-sized files and bad arguments.

A separate 30-generation run was also identical line for line.

## Files

| File | What it is For |
|---|---|
| `snake.asm` | everything: game, senses, net, evolution, renderer, `play` and `eval` |
| `ref.py` | the Python model (this is the spec) |
| `test.py` | assembly versus model comparison |
| `bench.py` | train several seeds and score the champions on unseen games |
| `champion.bin` | a trained champion (seed 1, 1000 generations) |
| `Makefile` | `make`, `test`, `run`, `play`, `train`, `bench`, `clean` |

## Ideas for later - No promises :)

- Cross-over between parents, not only mutation.
- A free-space sense (how many cells the snake can still reach) to avoid trapping itself.
- A bigger population, which may let larger nets pay off.
- Smarter fitness so snakes stop looping to survive.
