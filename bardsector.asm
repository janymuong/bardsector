; =============================================================================
;   TO BOOT OR NOT TO BOOT
;   a neural network that fits in a 512-byte boot sector and writes Shakespeare
;
;   No OS, no libc, no runtime. The BIOS loads this sector, and this sector
;   *is* the language model: ~220 bytes of 16-bit x86 + 243 bytes of 4-bit weights.
;
;   The net (trained by train.py on ~1 MB of Shakespeare):
;       h[k]     = ReLU( E1[prev][k] + E2[prev2][k] - 16 )       k = 0..5
;       logit[j] = sum_k h[k] * W[j][k]                          j = 27 symbols
;       next     = argmax_j ( logit[j] + noise )                 noise = PRNG
;
;   Type letters and the bard continues your thought.
;
;   build: nasm -f bin bardsector.asm -o bardsector.bin
; =============================================================================
bits 16
org 0x7C00

H       equ 0x7E00              ; 6 bytes of hidden activations, just past the sector

start:
    xor     ax, ax
    mov     ds, ax
    mov     ss, ax
    mov     sp, 0x7C00
    sti                         ; hlt below needs interrupts on
    mov     al, 3               ; 80x25 text mode (also clears the screen)
    int     0x10

    mov     si, banner          ; title card
    xor     bx, bx              ; page 0 for the BIOS teletype
.b: lodsb
    test    al, al
    jz      .go
    mov     ah, 0x0E
    int     0x10
    jmp     .b
.go:
    xor     ah, ah              ; seed the PRNG from the BIOS tick counter
    int     0x1A
    mov     di, dx              ; di = PRNG state for the whole program
    push    E2                  ; context lives on the stack as two row pointers:
    push    E1                  ;   top = E1 row of prev char, next = E2 row of the one before

gen:
    ; ---- layer 1: embed the last two chars, add, ReLU ----------------------
    pop     si                  ; si -> E1 row (3 bytes = 6 nibbles)
    pop     bx                  ; bx -> E2 row
    mov     bp, H
    mov     cl, 3
.h: lodsb
    aam     16                  ; al = low nibble, ah = high nibble
    mov     dx, ax
    mov     al, [bx]
    inc     bx
    aam     16
    add     al, dl              ; nibbles are stored as w+8, so the sum has +16 in it
    sub     al, 16              ; subtract it; borrow means negative -> ReLU gives 0
    jnc     .p
    xor     al, al
.p: add     ah, dh
    sub     ah, 16
    jnc     .q
    xor     ah, ah
.q: mov     [bp], ax            ; h[2m], h[2m+1]
    inc     bp
    inc     bp
    dec     cl
    jnz     .h
    push    si                  ; = E1 row + 3, used to shift the context below

    ; ---- layer 2: 27 logits, add noise, keep the best as (score<<5 | symbol) ----
    mov     si, W
    mov     ch, 26              ; symbol number, counts down; W rows are stored z..space
    xor     dx, dx              ; dx = best key so far
.row:
    mov     bx, H
    xor     bp, bp              ; bp = dot product accumulator
    mov     cl, 3
.dot:
    lodsb
    aam     16
    push    ax
    mul     byte [bx]           ; low nibble  * h[2m]
    add     bp, ax
    pop     ax
    mov     al, ah
    mul     byte [bx+1]         ; high nibble * h[2m+1]
    add     bp, ax
    add     bx, 2
    dec     cl
    jnz     .dot
    imul    di, di, 25173       ; PRNG: 16-bit LCG, full period
    add     di, 13849
    mov     ax, di
    shr     ax, 10              ; 6 bits of noise: this is the "temperature"
    add     ax, bp
    shl     ax, 5
    or      al, ch              ; low 5 bits carry the symbol
    cmp     ax, dx
    jbe     .no
    mov     dx, ax
.no:dec     ch
    jns     .row

    mov     cl, dl
    and     cl, 31              ; cl = the net's choice

    ; ---- keyboard: a typed key overrides the choice (folded to 5 bits) -----
    mov     ah, 1
    int     0x16
    jz      .nk
    xor     ah, ah
    int     0x16
    and     al, 31              ; space -> 0, a..z -> 1..26 (so do A..Z)
    mov     cl, al
.nk:
    ; ---- print it -----------------------------------------------------------
    mov     al, cl
    test    al, al
    jnz     .nz
    mov     al, 0xC0            ; 0xC0 + 0x60 wraps to 0x20 = space
.nz:add     al, 0x60            ; 1..26 -> 'a'..'z'
    mov     ah, 0x0E
    xor     bx, bx
    int     0x10

    ; ---- shift the context: prev2 <- prev, prev <- this --------------------
    pop     ax                  ; E1 row of prev, + 3
    add     ax, E2 - E1 - 3     ; same row in the E2 table
    push    ax
    mov     al, cl
    mov     ah, 3
    mul     ah                  ; ax = 3 * symbol
    add     ax, E1
    push    ax
    hlt                         ; sleep until the next timer tick (~55 ms): typewriter pace
    jmp     gen

banner: db "TO BOOT, OR NOT TO BOOT:", 13, 10, 0

    times (510 - 243) - ($ - $$) db 0
%include "weights.inc"
    dw 0xAA55
