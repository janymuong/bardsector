; =============================================================================
;   SSSURVIVAL OF THE FITTEST
;   neuroevolution Snake in pure x86-64 Linux assembly (no libc)
;
;   usage: ./snake [generations=200] [seed=1] [watch=20]
;
;   A population of 64 tiny neural nets plays Snake. The best survive, mutated
;   copies replace the rest. No backprop, no gradients: just selection.
;   The game, the nets, the evolution and the renderer are all in this file.
;
;     watch = 0   headless: one stats line per generation, then exit
;     watch = N   also replay the champion live every N generations, and play
;                 it forever once training ends (Ctrl-C to stop)
;
;   The final champion's 64 bytes of weights are saved to champion.bin.
;   ref.py is a Python model of everything here; test.py checks they agree.
;
;   build: nasm -f elf64 snake.asm -o snake.o && ld snake.o -o snake
; =============================================================================

; ---- hyper-parameters: keep in sync with ref.py ----------------------------
%define POP        64
%define GSZ        64
%define NGAMES     3
%define HUNGER     100
%define SCORE_CAP  200
%define ELITE      8
%define PARENTS    16
%define MUT_MASK   15

%define FRAME_LINES 19
%define FRAME_NS    60000000            ; 60 ms per frame
%define GAP_NS      700000000           ; pause between demo games

%define SYS_write   1
%define SYS_open    2
%define SYS_close   3
%define SYS_nanosleep 35
%define SYS_exit    60
%define O_CREAT_WR_TRUNC 0x241

%macro PUTS 1                           ; copy a string literal to [rdi], advance rdi
    mov     esi, %1
    mov     ecx, %1_len
    rep movsb
%endmacro

section .data
dxs:        db 0, 1, 0, -1             ; N E S W
dys:        db -1, 0, 1, 0

s_up:       db 27, "[", "19A", 13
s_up_len    equ $ - s_up
s_gen:      db "gen "
s_gen_len   equ $ - s_gen
s_best:     db " best "
s_best_len  equ $ - s_best
s_mean:     db " mean "
s_mean_len  equ $ - s_mean
s_apl:      db " apples "
s_apl_len   equ $ - s_apl
s_aplf:     db "  apples "
s_aplf_len  equ $ - s_aplf
s_stp:      db "  steps "
s_stp_len   equ $ - s_stp
s_eol:      db 27, "[K", 10
s_eol_len   equ $ - s_eol
s_nl:       db 10
s_nl_len    equ $ - s_nl
s_border:   db "+"
            times 32 db "-"
            db "+", 10
s_border_len equ $ - s_border
c_empty:    db 27, "[90m. ", 27, "[0m"
c_empty_len equ $ - c_empty
c_body:     db 27, "[42m  ", 27, "[0m"
c_body_len  equ $ - c_body
c_head:     db 27, "[102m  ", 27, "[0m"
c_head_len  equ $ - c_head
c_food:     db 27, "[41m  ", 27, "[0m"
c_food_len  equ $ - c_food
s_saved:    db "saved champion.bin", 10
s_saved_len equ $ - s_saved
fname:      db "champion.bin", 0
ts_frame:   dq 0, FRAME_NS
ts_gap:     dq 0, GAP_NS

section .bss
rng:        resd 1                      ; evolution PRNG
grng:       resd 1                      ; per-game PRNG (food placement)
seeds:      resd NGAMES
p_gens:     resd 1
p_seed:     resd 1
p_watch:    resd 1
gen_no:     resd 1
render:     resd 1
up_flag:    resd 1

pop:        resb POP * GSZ
newpop:     resb POP * GSZ
fit:        resd POP
apl:        resd POP
keys:       resq POP
champ:      resb GSZ

; game state
g_gen:      resq 1                      ; genome pointer
g_occ:      resb 256
g_ring:     resb 256
g_head:     resd 1
g_tail:     resd 1
g_pos:      resd 1
g_dir:      resd 1
g_food:     resd 1
g_score:    resd 1
g_steps:    resd 1
g_hunger:   resd 1
g_x:        resd 6                      ; net inputs
g_h:        resd 6                      ; hidden activations
g_o:        resd 3                      ; outputs

line:       resb 128
frame:      resb 16384

section .text
global _start

; =============================================================================
; PRNGs: xorshift32. rand uses the evolution state, grand the game state.
; out: eax. clobbers edx.
; =============================================================================
rand:
    mov     eax, [rng]
    mov     edx, eax
    shl     edx, 13
    xor     eax, edx
    mov     edx, eax
    shr     edx, 17
    xor     eax, edx
    mov     edx, eax
    shl     edx, 5
    xor     eax, edx
    mov     [rng], eax
    ret

grand:
    mov     eax, [grng]
    mov     edx, eax
    shl     edx, 13
    xor     eax, edx
    mov     edx, eax
    shr     edx, 17
    xor     eax, edx
    mov     edx, eax
    shl     edx, 5
    xor     eax, edx
    mov     [grng], eax
    ret

; =============================================================================
; The game
; =============================================================================
place_food:                             ; random free cell -> g_food
.try:
    call    grand
    shr     eax, 8
    movzx   eax, al
    cmp     byte [g_occ + rax], 0
    jne     .try
    mov     [g_food], eax
    ret

game_init:
    mov     edi, g_occ
    xor     eax, eax
    mov     ecx, 256
    rep stosb
    mov     byte [g_ring], 134          ; snake starts at (6..8, 8) heading east
    mov     byte [g_ring + 1], 135
    mov     byte [g_ring + 2], 136
    mov     byte [g_occ + 134], 1
    mov     byte [g_occ + 135], 1
    mov     byte [g_occ + 136], 1
    mov     dword [g_tail], 0
    mov     dword [g_head], 3
    mov     dword [g_pos], 136
    mov     dword [g_dir], 1
    mov     dword [g_score], 0
    mov     dword [g_steps], 0
    mov     dword [g_hunger], 0
    jmp     place_food

; fill g_x[0..5]: danger left/straight/right, food left/ahead/right
compute_inputs:
    mov     eax, [g_pos]
    mov     r8d, eax
    and     r8d, 15                     ; x
    shr     eax, 4
    mov     r9d, eax                    ; y
    mov     r10d, [g_dir]
    xor     ecx, ecx
.dng:
    lea     eax, [r10 + rcx + 3]        ; heading after turning left/straight/right
    and     eax, 3
    movsx   edx, byte [dxs + rax]
    add     edx, r8d                    ; nx
    movsx   eax, byte [dys + rax]
    add     eax, r9d                    ; ny
    mov     esi, 1
    cmp     edx, 15                     ; unsigned: also catches negatives
    ja      .store
    cmp     eax, 15
    ja      .store
    shl     eax, 4
    add     eax, edx
    xor     esi, esi
    cmp     byte [g_occ + rax], 0
    je      .store
    mov     esi, 1
.store:
    mov     [g_x + rcx * 4], esi
    inc     ecx
    cmp     ecx, 3
    jb      .dng

    mov     eax, [g_food]
    mov     edx, eax
    and     edx, 15
    shr     eax, 4
    sub     edx, r8d                    ; food dx
    sub     eax, r9d                    ; food dy
    movsx   esi, byte [dxs + r10]
    imul    esi, edx
    movsx   edi, byte [dys + r10]
    imul    edi, eax
    add     esi, edi                    ; esi = food ahead component
    lea     ecx, [r10 + 1]
    and     ecx, 3                      ; heading to our right
    movsx   edi, byte [dxs + rcx]
    imul    edi, edx
    movsx   r11d, byte [dys + rcx]
    imul    r11d, eax
    add     edi, r11d                   ; edi = food right component
    xor     eax, eax
    test    edi, edi
    setl    al
    mov     [g_x + 12], eax             ; food left
    xor     eax, eax
    test    esi, esi
    setg    al
    mov     [g_x + 16], eax             ; food ahead
    xor     eax, eax
    test    edi, edi
    setg    al
    mov     [g_x + 20], eax             ; food right
    ret

; the net. in: rsi = genome. out: eax = 0 left, 1 straight, 2 right
forward:
    xor     ecx, ecx
.hid:
    movsx   eax, byte [rsi + 36 + rcx]  ; bias
    lea     rdx, [rcx + rcx * 2]
    add     rdx, rdx                    ; 6 * hidden unit
    add     rdx, rsi                    ; -> this unit's input weights
    xor     edi, edi
.hin:
    movsx   r8d, byte [rdx + rdi]
    imul    r8d, [g_x + rdi * 4]
    add     eax, r8d
    inc     edi
    cmp     edi, 6
    jb      .hin
    test    eax, eax                    ; ReLU
    jns     .keep
    xor     eax, eax
.keep:
    mov     [g_h + rcx * 4], eax
    inc     ecx
    cmp     ecx, 6
    jb      .hid

    xor     ecx, ecx
.out:
    movsx   eax, byte [rsi + 60 + rcx]
    lea     rdx, [rcx + rcx * 2]
    add     rdx, rdx
    lea     rdx, [rsi + rdx + 42]       ; -> this output's hidden weights
    xor     edi, edi
.oin:
    movsx   r8d, byte [rdx + rdi]
    imul    r8d, [g_h + rdi * 4]
    add     eax, r8d
    inc     edi
    cmp     edi, 6
    jb      .oin
    mov     [g_o + rcx * 4], eax
    inc     ecx
    cmp     ecx, 3
    jb      .out

    mov     eax, 1                      ; argmax, ties favour straight
    mov     edx, [g_o + 4]
    mov     ecx, [g_o]
    cmp     ecx, edx
    jle     .n0
    mov     edx, ecx
    xor     eax, eax
.n0:
    mov     ecx, [g_o + 8]
    cmp     ecx, edx
    jle     .n2
    mov     eax, 2
.n2:
    ret

; play one game. in: rsi = genome, grng seeded, [render] set.
; out: eax = fitness (apples*100 + steps survived), edx = apples
; uses only memory for game state, so r12-r15 and rbp belong to the caller.
play:
    mov     [g_gen], rsi
    call    game_init
.loop:
    cmp     dword [render], 0
    je      .nr
    call    draw_frame
.nr:
    call    compute_inputs
    mov     rsi, [g_gen]
    call    forward
    mov     ecx, [g_dir]
    lea     ecx, [rcx + rax + 3]
    and     ecx, 3
    mov     [g_dir], ecx
    mov     eax, [g_pos]
    mov     edx, eax
    and     edx, 15
    shr     eax, 4
    movsx   esi, byte [dxs + rcx]
    add     edx, esi                    ; new x
    movsx   esi, byte [dys + rcx]
    add     eax, esi                    ; new y
    cmp     edx, 15
    ja      .dead
    cmp     eax, 15
    ja      .dead
    shl     eax, 4
    add     eax, edx                    ; new cell
    cmp     byte [g_occ + rax], 0
    jne     .dead
    inc     dword [g_steps]
    inc     dword [g_hunger]
    cmp     eax, [g_food]
    jne     .nofood
    inc     dword [g_score]
    mov     dword [g_hunger], 0
    mov     esi, 1                      ; esi = ate
    jmp     .grow
.nofood:
    xor     esi, esi
    mov     edx, [g_tail]               ; tail cell becomes free
    movzx   ecx, byte [g_ring + rdx]
    mov     byte [g_occ + rcx], 0
    inc     edx
    and     edx, 255
    mov     [g_tail], edx
.grow:
    mov     byte [g_occ + rax], 1
    mov     edx, [g_head]
    mov     [g_ring + rdx], al
    inc     edx
    and     edx, 255
    mov     [g_head], edx
    mov     [g_pos], eax
    test    esi, esi
    jz      .nf
    cmp     dword [g_score], SCORE_CAP
    jae     .dead
    call    place_food
.nf:
    cmp     dword [g_hunger], HUNGER
    jb      .loop
.dead:
    cmp     dword [render], 0
    je      .ret
    call    draw_frame
.ret:
    mov     edx, [g_score]
    imul    eax, edx, 100
    add     eax, [g_steps]
    ret

; =============================================================================
; Renderer: builds one ANSI frame in `frame`, writes it, sleeps.
; Redraws in place by moving the cursor up FRAME_LINES lines (not the first time).
; =============================================================================
draw_frame:
    mov     edi, frame
    cmp     dword [up_flag], 0
    je      .noup
    PUTS    s_up
.noup:
    mov     dword [up_flag], 1
    PUTS    s_gen
    mov     eax, [gen_no]
    call    utoa
    PUTS    s_aplf
    mov     eax, [g_score]
    call    utoa
    PUTS    s_stp
    mov     eax, [g_steps]
    call    utoa
    PUTS    s_eol
    PUTS    s_border
    xor     r9d, r9d                    ; cell index 0..255
.row:
    mov     byte [rdi], '|'
    inc     rdi
.cell:
    cmp     r9d, [g_pos]
    je      .head
    cmp     byte [g_occ + r9], 0
    jne     .body
    cmp     r9d, [g_food]
    je      .food
    PUTS    c_empty
    jmp     .next
.head:
    PUTS    c_head
    jmp     .next
.body:
    PUTS    c_body
    jmp     .next
.food:
    PUTS    c_food
.next:
    inc     r9d
    test    r9d, 15
    jnz     .cell
    mov     byte [rdi], '|'
    mov     byte [rdi + 1], 10
    add     rdi, 2
    cmp     r9d, 256
    jb      .row
    PUTS    s_border
    mov     esi, frame
    mov     rdx, rdi
    sub     rdx, rsi
    mov     edi, 1
    mov     eax, SYS_write
    syscall
    mov     edi, ts_frame
    xor     esi, esi
    mov     eax, SYS_nanosleep
    syscall
    ret

; utoa: write rax as decimal at [rdi], advance rdi. Clobbers rax, rcx, rdx.
utoa:
    push    rbx
    mov     ebx, 10
    xor     ecx, ecx
.div:
    xor     edx, edx
    div     rbx
    add     dl, '0'
    push    rdx
    inc     ecx
    test    rax, rax
    jnz     .div
.pop:
    pop     rax
    mov     [rdi], al
    inc     rdi
    dec     ecx
    jnz     .pop
    pop     rbx
    ret

; atoi: rsi -> decimal string, eax = value (stops at the first non-digit)
atoi:
    xor     eax, eax
.l:
    movzx   edx, byte [rsi]
    sub     edx, '0'
    cmp     edx, 9
    ja      .d
    imul    eax, eax, 10
    add     eax, edx
    inc     rsi
    jmp     .l
.d:
    ret

; =============================================================================
; main: r12 = genome index, r13 = inner counter, r14 = generation, r15 = fitness sum
;       rbp = apple sum
; =============================================================================
_start:
    mov     dword [p_gens], 200
    mov     dword [p_seed], 1
    mov     dword [p_watch], 20
    cmp     qword [rsp], 2
    jb      .args_done
    mov     rsi, [rsp + 16]
    call    atoi
    mov     [p_gens], eax
    cmp     qword [rsp], 3
    jb      .args_done
    mov     rsi, [rsp + 24]
    call    atoi
    mov     [p_seed], eax
    cmp     qword [rsp], 4
    jb      .args_done
    mov     rsi, [rsp + 32]
    call    atoi
    mov     [p_watch], eax
.args_done:
    cmp     dword [p_gens], 0
    jne     .g_ok
    mov     dword [p_gens], 1
.g_ok:
    mov     eax, [p_seed]
    test    eax, eax
    jnz     .s_ok
    mov     eax, 1
.s_ok:
    mov     [rng], eax

    ; ---- random initial population: small weights in -32..31 ---------------
    xor     r13d, r13d
.init:
    mov     eax, r13d
    and     eax, 63
    cmp     eax, 63
    jne     .rnd
    xor     eax, eax                    ; pad byte
    jmp     .st
.rnd:
    call    rand
    shr     eax, 8
    and     eax, 63
    sub     eax, 32
.st:
    mov     [pop + r13], al
    inc     r13d
    cmp     r13d, POP * GSZ
    jb      .init

    mov     r14d, 1
.gen:
    mov     [gen_no], r14d
    call    rand
    mov     [seeds], eax
    call    rand
    mov     [seeds + 4], eax
    call    rand
    mov     [seeds + 8], eax

    ; ---- evaluate: every genome plays the same NGAMES games -----------------
    xor     r12d, r12d
.ev_i:
    xor     r15d, r15d
    xor     ebp, ebp
    xor     r13d, r13d
.ev_g:
    mov     eax, [seeds + r13 * 4]
    mov     [grng], eax
    mov     dword [render], 0
    mov     esi, pop
    mov     rax, r12
    shl     rax, 6
    add     rsi, rax
    call    play
    add     r15d, eax
    add     ebp, edx
    inc     r13d
    cmp     r13d, NGAMES
    jb      .ev_g
    mov     [fit + r12 * 4], r15d
    mov     [apl + r12 * 4], ebp
    mov     eax, r15d
    shl     rax, 8
    or      rax, r12
    mov     [keys + r12 * 8], rax       ; key = fitness << 8 | index
    inc     r12d
    cmp     r12d, POP
    jb      .ev_i

    ; ---- sort keys, best first (insertion sort; keys are unique) ------------
    mov     ecx, 1
.si:
    cmp     ecx, POP
    jae     .sorted
    mov     rax, [keys + rcx * 8]
    mov     rdx, rcx
    dec     rdx
.sj:
    test    rdx, rdx
    js      .place
    mov     rsi, [keys + rdx * 8]
    cmp     rsi, rax
    jae     .place
    mov     [keys + rdx * 8 + 8], rsi
    dec     rdx
    jmp     .sj
.place:
    mov     [keys + rdx * 8 + 8], rax
    inc     ecx
    jmp     .si
.sorted:

    ; ---- stats line: gen N best B mean M apples A ---------------------------
    xor     r8d, r8d
    xor     ecx, ecx
.sum:
    mov     eax, [fit + rcx * 4]
    add     r8, rax
    inc     ecx
    cmp     ecx, POP
    jb      .sum
    shr     r8, 6                       ; mean = sum / 64
    mov     r9, [keys]
    mov     r10, r9
    shr     r9, 8                       ; best fitness
    and     r10d, 255                   ; its index
    mov     edi, line
    PUTS    s_gen
    mov     eax, r14d
    call    utoa
    PUTS    s_best
    mov     rax, r9
    call    utoa
    PUTS    s_mean
    mov     rax, r8
    call    utoa
    PUTS    s_apl
    mov     eax, [apl + r10 * 4]
    call    utoa
    PUTS    s_nl
    mov     esi, line
    mov     rdx, rdi
    sub     rdx, rsi
    mov     edi, 1
    mov     eax, SYS_write
    syscall

    ; ---- remember this generation's champion --------------------------------
    mov     esi, pop
    shl     r10, 6
    add     rsi, r10
    mov     edi, champ
    mov     ecx, GSZ
    rep movsb

    ; ---- optional live replay of the champion -------------------------------
    mov     ecx, [p_watch]
    test    ecx, ecx
    jz      .nowatch
    mov     eax, r14d
    xor     edx, edx
    div     ecx
    test    edx, edx
    jnz     .nowatch
    mov     eax, [seeds]
    mov     [grng], eax
    mov     dword [render], 1
    mov     dword [up_flag], 0
    mov     esi, champ
    call    play
    mov     dword [render], 0
.nowatch:
    cmp     r14d, [p_gens]
    jae     .done

    ; ---- breed: elites unchanged, the rest mutated copies of the top PARENTS -
    xor     r13d, r13d
.el:
    mov     rax, [keys + r13 * 8]
    and     eax, 255
    shl     eax, 6
    mov     esi, pop
    add     rsi, rax
    mov     edi, r13d
    shl     edi, 6
    add     edi, newpop
    mov     ecx, GSZ
    rep movsb
    inc     r13d
    cmp     r13d, ELITE
    jb      .el
.ch:
    call    rand
    shr     eax, 8
    and     eax, PARENTS - 1
    mov     rax, [keys + rax * 8]
    and     eax, 255
    shl     eax, 6
    mov     esi, pop
    add     rsi, rax
    mov     edi, r13d
    shl     edi, 6
    add     edi, newpop
    mov     ecx, GSZ
    rep movsb
    sub     rdi, GSZ                    ; back to the start of the child
    xor     ecx, ecx
.mu:
    call    rand
    test    eax, MUT_MASK
    jnz     .nm
    shr     eax, 8
    and     eax, 15
    sub     eax, 8
    movsx   edx, byte [rdi + rcx]
    add     eax, edx
    cmp     eax, 127
    jle     .c1
    mov     eax, 127
.c1:
    cmp     eax, -128
    jge     .c2
    mov     eax, -128
.c2:
    mov     [rdi + rcx], al
.nm:
    inc     ecx
    cmp     ecx, GSZ - 1
    jb      .mu
    inc     r13d
    cmp     r13d, POP
    jb      .ch

    mov     esi, newpop
    mov     edi, pop
    mov     ecx, POP * GSZ
    rep movsb
    inc     r14d
    jmp     .gen

    ; ---- save the champion --------------------------------------------------
.done:
    mov     eax, SYS_open
    mov     edi, fname
    mov     esi, O_CREAT_WR_TRUNC
    mov     edx, 420                    ; 0644
    syscall
    test    rax, rax
    js      .nosave
    mov     r12, rax
    mov     eax, SYS_write
    mov     rdi, r12
    mov     esi, champ
    mov     edx, GSZ
    syscall
    mov     eax, SYS_close
    mov     rdi, r12
    syscall
.nosave:
    mov     eax, SYS_write
    mov     edi, 1
    mov     esi, s_saved
    mov     edx, s_saved_len
    syscall

    ; ---- endless demo -------------------------------------------------------
    cmp     dword [p_watch], 0
    je      .exit
    mov     dword [up_flag], 0
.demo:
    call    rand
    mov     [grng], eax
    mov     dword [render], 1
    mov     esi, champ
    call    play
    mov     edi, ts_gap
    xor     esi, esi
    mov     eax, SYS_nanosleep
    syscall
    jmp     .demo
.exit:
    xor     edi, edi
    mov     eax, SYS_exit
    syscall
