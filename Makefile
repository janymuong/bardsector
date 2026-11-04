NASM ?= nasm
QEMU ?= qemu-system-i386

bardsector.bin: bardsector.asm weights.inc
	$(NASM) -f bin -w+all -l bardsector.lst -o $@ $<
	@python3 emu.py size

weights.inc model.npz: train.py explore.py data/shakespeare.txt
	python3 train.py 6 6000 6

test: bardsector.bin
	python3 emu.py test

run: bardsector.bin
	$(QEMU) -drive format=raw,file=bardsector.bin

clean:
	rm -f bardsector.bin bardsector.lst

.PHONY: test run clean
