NASM ?= nasm
LD   ?= ld

snake: snake.o
	$(LD) -o $@ $<

snake.o: snake.asm
	$(NASM) -f elf64 -g -F dwarf -w+all $(NASMFLAGS) -o $@ $<

test: snake
	python3 test.py

run: snake
	./snake

train: snake
	./snake 1000 1 0

play: snake champion.bin
	./snake play champion.bin

champion.bin: snake
	./snake 1000 1 0

bench: snake
	python3 bench.py ./snake 1000

clean:
	rm -f snake snake.o

.PHONY: test run train play bench clean