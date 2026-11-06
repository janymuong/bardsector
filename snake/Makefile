NASM ?= nasm
LD   ?= ld

snake: snake.o
	$(LD) -o $@ $<

snake.o: snake.asm
	$(NASM) -f elf64 -g -F dwarf -w+all -o $@ $<

test: snake
	python3 test.py

run: snake
	./snake 200 1 20

train: snake
	./snake 300 1 0

clean:
	rm -f snake snake.o champion.bin

.PHONY: test run train clean
