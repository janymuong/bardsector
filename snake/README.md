# SSSurvival of the Fittest

Neuroevolution Snake in **pure x86-64 Linux assembly**. Again, no libc, no runtime.

A population of 64 tiny neural nets plays Snake. The best survive, mutated copies
replace the rest, and a few dozen generations later they are eating 40+ apples a game.
There is no backprop and no gradient: only selection. The game, the nets, the
evolution and the terminal renderer all live in `snake.asm`.

```
gen 3  apples 1  steps 11
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
make run             # train 200 generations, replay the champion every 20,
                     # then play it forever (Ctrl-C to stop)
make train           # headless: 300 generations, one stats line each
make test            # check the assembly against the Python model
```

```
./snake [generations=200] [seed=1] [watch=20]
```

| Argument | Meaning |
|---|---|
| `generations` | how many generations to evolve |
| `seed` | PRNG seed; the same seed always gives the same run |
| `watch` | `0`: headless, print stats and exit. `N`: also replay the champion every N generations, and play it forever after training |

Use a terminal at least 20 lines tall: each frame is 19 lines and is redrawn in place.

Each generation prints one line:

```
gen 150 best 22043 mean 7705 apples 188
```

`best` and `mean` are fitness over the generation's 3 games (`apples * 100 + steps survived`),
and `apples` is the best genome's total over those 3 games. The last generation's champion is
written to `champion.bin` (64 bytes).

## How it works

**The game.** 16x16 board with walls. The snake starts with length 3 heading east, and a
snake that goes 100 steps without eating starves. Food is placed with a per-game xorshift32
PRNG, so a game is fully determined by its seed.

**The senses.** Six binary inputs: is there danger (wall or body) one step to the left,
straight ahead and to the right, and is the food to the left, ahead or to the right.

**The brain.** 6 inputs, 6 ReLU hidden units, 3 outputs (turn left, go straight, turn right),
the largest output wins and ties go straight. All weights are signed bytes: 36 + 6 + 18 + 3 = 63
bytes plus one pad byte = a 64-byte genome.

**The evolution.** Every generation, all 64 genomes play the same 3 games. The top 8 are copied
unchanged, and the other 56 are mutated copies of parents picked at random from the top 16.
Each byte mutates with probability 1/16 by adding a value from -8 to 7 (saturating).
Using the same games for the whole population keeps the comparison fair.

**Speed.** 300 generations (about 57,000 games per 100 generations) take well under a second.

## Results

Saved champions averaged about 38-52 apples on 40 fresh games they never trained on
(best single game: 79 apples), for several seeds and for 150, 300 and 500 generations.
Training curves are noisy because each generation is judged on only 3 games.

## How it is verified

`ref.py` is an integer-exact Python model of the whole (game, net, PRNG, selection,
mutation). `test.py` runs `./snake` headless for several seeds and checks that every stat line
and the bytes of `champion.bin` match the model exactly. A separate 40-generation run was also
identical line for line.

## Files

| File | What it is |
|---|---|
| `snake.asm` | everything: game, net, evolution, renderer |
| `ref.py` | the Python model (this is the spec) |
| `test.py` | assembly versus model comparison |
| `Makefile` | `make`, `run`, `train`, `test`, `clean` |

## Ideas for later

- `./snake play champion.bin` to watch a saved champion without retraining.
- Judge each genome on more games, or re-test the champion on fresh games before saving it.
- Bigger senses (distance to walls, tail direction) or a bigger hidden layer.
- Cross-over between parents, not only mutation.
- Smarter fitness so snakes stop looping to survive.
