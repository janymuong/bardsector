# To Boot Or Not To Boot

> A neural network that fits in a **512-byte boot sector** and writes Shakespeare.

No OS, no libc, or runtime. The BIOS loads one sector, and that sector *is* the
language model: about 233 bytes of 16-bit x86 plus 243 bytes of 4-bit weights.
Type letters and the bard continues your thought.

```
TO BOOT, OR NOT TO BOOT:
ane she ind a inst the th i his th it thassar mate the is the th whase in a bas
wis ann ther the the innee thereng th the a you the ther ine hindend yrors thee
```

It is not good Shakespeare. A net this small only learns letter patterns, and which is
the joke.


## The model

```
h[k]     = ReLU( E1[prev][k] + E2[prev2][k] )     k = 0..5   (hidden layer, 6 units)
logit[j] = sum_k h[k] * W[j][k]                   j = 0..26  (space, a..z)
next     = argmax_j ( logit[j] + noise )          noise from a 16-bit LCG
```

- Context is the previous two characters (27 symbols: space and a-z).
- All weights are 4-bit, stored as `w + 8` so the assembly never handles signs.
- The noise is the temperature: it turns argmax into a sampler.
- A typed key overrides the net's choice, so you can steer it.
- Trained by `train.py` with quantization-aware training. Final loss is about
  3.24 bits/char (uniform guessing is 4.75, a full trigram table is about 2.8).

## Requirements

| Tooling | For |
|---|---|
| `nasm` | assembling the boot sector |
| `make`, `python` | build and tooling |
| `qemu-system-i386` | running the image |
| `numpy` | training |
| `unicorn` | headless emulator tests |


## **Setup (Arch):**
```
sudo pacman -S nasm make python python-numpy qemu-system-x86 python-unicorn
# or pip install unicorn or via virtual env - only needed for make test and emu.py demo
```


## **Run it:**

```
make                          # assemble bardsector.bin, print the size breakdown
make run                      # boot in QEMU; type to steer it
make test                     # emulator run compared against the Python model
python3 emu.py demo           # 600 characters, no QEMU needed
python3 emu.py demo 400 "to be or not to be"   # give it a prompt
```

To run it on real hardware, write the image to a USB stick (this destroys
whatever is on it, so check the device name twice): 
`sudo dd if=bardsector.bin of=/dev/sdX bs=512 count=1`, then boot it in
legacy/BIOS mode.


## Headless demo (no QEMU needed)

```
python3 emu.py demo                         # 600 chars of free-running output
python3 emu.py demo 400 "to be or not to be"  # type a prompt, then let it continue
python3 emu.py size                         # code / weights / free-bytes breakdown
```

## **Retrain it:**

```
mkdir -p data
curl -o data/shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

python3 explore.py                # compare model shapes (bits/char vs size)
python3 train.py 6 6000 6         # hidden units, steps, seeds -> model.npz, weights.inc
make                              # rebuild the sector with the new weights
make test
```

Training takes under a minute. The weights must stay at 243 bytes
(3 tables x 27 rows x 3 bytes) unless you also change the layout constants in
`bardsector.asm` and `NW` in `emu.py`.

## Files

| File | What it is |
|---|---|
| `bardsector.asm` | the boot sector: keyboard, forward pass, sampling, printing |
| `weights.inc` | generated 4-bit weight tables (E1, E2, W) |
| `train.py` | trains and exports the weights |
| `explore.py` | corpus loading and model-shape comparison |
| `emu.py` | Unicorn-based BIOS stand-in, Python reference model, tests, demo |
| `Makefile` | `make`, `make run`, `make test`, `make clean` |
| `data/shakespeare.txt` | training corpus (download it, see above) |

## How it is verified

`emu.py test` boots the real image in Unicorn with a stub BIOS (teletype,
keyboard, tick counter) and compares every output character with an independent
integer re-implementation of the net that reads its weights from the assembled
image. Five cases match exactly: two seeds, typed text, mixed-case and
punctuation input, and all key codes 0-127. It has not been run under real BIOS
firmware or QEMU in the build environment, so timing and keyboard behaviour
there are untested.

## Ideas for the 34 free bytes

- A third character of context (cost: one more weight table, so retrain smaller).
- Tune the noise shift (`shr ax, 10`) for more or less chaos.
- Print the banner in colour, or add a cursor.



---

**MORE**: Neuroevolution Snake. After that, Fake Shakespeare can get its own bigger version, since the boot sector is limited to letter patterns.