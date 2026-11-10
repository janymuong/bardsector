; =============================================================================
;   SSSURVIVAL OF THE FITTEST
;   neuroevolution Snake in pure x86-64 Linux assembly (no libc)
;
;   snake [generations=500] [seed=1] [watch=100]    evolve a population
;   snake play [file=champion.bin] [seed]           watch a saved genome play
;   snake eval [file=champion.bin] [games=100] [seed=1]
;                                                   score a saved genome
;
;   A population of 64 tiny neural nets plays Snake. The best survive, mutated
;   copies replace the rest. No backprop, no gradients: just selection.
;   The game, the nets, the evolution and the renderer are all in this file.
;
;   Training: every generation, the top few genomes are re-tested on a fixed set
;   of fresh games, and the best genome ever seen (by that re-test) is the
;   champion, so one lucky generation can't become the saved result.
;     watch = 0   headless: one stats line per generation, then exit
;     watch = N   also replay the champion live every N generations, and play it
;                 forever once training ends (Ctrl-C to stop)
;   The champion's weights are saved to champion.bin.
;
;   ref.py is a Python model of everything here; test.py checks they agree.
;
;   build: nasm -f elf64 snake.asm -o snake.o && ld snake.o -o snake
;   Build-time knobs (see README): -DNHID -DRAYS -DRAYBIN -DRAY_CAP -DNGAMES
;   -DMUT_MASK -DELITE -DPARENTS. ref.py must be given the same values.
; =============================================================================

; ---- hyper-parameters: keep in sync with ref.py ----------------------------
%ifndef NHID
%define NHID       8                    ; hidden units
%endif
%ifndef RAYS
%define RAYS       1                    ; 1 = three extra "look ahead" senses
%endif
%ifndef RAYBIN
%define RAYBIN     1                    ; 1: sense = 0/1 "obstacle within RAY_CAP cells"
%endif                                  ; 0: sense = free cells before the obstacle
%ifndef RAY_CAP
%define RAY_CAP    2                    ; how far the senses look, in cells
%endif
%ifndef NGAMES
%define NGAMES     3                    ; games per genome per generation
%endif
%ifndef MUT_MASK
%define MUT_MASK   15                   ; a byte mutates when (rand & MUT_MASK) == 0
%endif
%ifndef ELITE
%define ELITE      8                    ; best genomes copied unchanged
%endif
%ifndef PARENTS
%define PARENTS    16                   ; children are bred from this many top genomes
%endif
%define POP        64
%define HUNGER     100
%define SCORE_CAP  200
%define TEST_GAMES 40                   ; fresh games used to re-test candidates
%define RETEST_TOP 4                    ; how many of a generation's best are re-tested
%define TEST_STATE 0x2545F491           ; seeds the fixed re-test games
%define EVAL_MAX   4096
%define REPLAY_STEPS 300                ; training-time replays stop after this many steps

; ---- genome layout (signed bytes) -------------------------------------------
;   W1[NHID][NIN] | B1[NHID] | W2[3][NHID] | B2[3] | padding up to GSZ
%define NIN        (6 + 3*RAYS)
%define OFF_B1     (NHID*NIN)
%define OFF_W2     (OFF_B1 + NHID)
%define OFF_B2     (OFF_W2 + 3*NHID)
%define GUSED      (OFF_B2 + 3)
%if GUSED <= 64
%define GSHIFT 6
%elif GUSED <= 128
%define GSHIFT 7
%elif GUSED <= 256
%define GSHIFT 8
%else
%define GSHIFT 9
%endif
%define GSZ        (1 << GSHIFT)

%define FRAME_LINES 19
%define FRAME_NS    60000000            ; 60 ms per frame
%define GAP_NS      700000000           ; pause between demo games

%define SYS_read    0
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
s_champlbl: db "champion"
s_champlbl_len equ $ - s_champlbl
s_best:     db " best "
s_best_len  equ $ - s_best
s_mean:     db " mean "
s_mean_len  equ $ - s_mean
s_apl:      db " apples "
s_apl_len   equ $ - s_apl
s_champ:    db " champ "
s_champ_len equ $ - s_champ
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
s_eval:     db "eval "
s_eval_len  equ $ - s_eval
s_colon:    db ": "
s_colon_len equ $ - s_colon
s_games:    db " games, apples mean "
s_games_len equ $ - s_games
s_min:      db " min "
s_min_len   equ $ - s_min
s_max:      db " max "
s_max_len   equ $ - s_max
s_usage:    db "usage: snake [generations=500] [seed=1] [watch=100]  evolve", 10
            db "       snake play [file=champion.bin] [seed]         watch a saved genome", 10
            db "       snake eval [file=champion.bin] [games=100] [seed=1]", 10
            db "arguments are plain numbers: snake 64 1 20, not generations=64", 10
s_usage_len equ $ - s_usage
e_open:     db "error: cannot open the genome file", 10
e_open_len  equ $ - e_open
e_size:     db "error: not a genome file for this build (wrong size)", 10
e_size_len  equ $ - e_size
fname:      db "champion.bin", 0
ts_frame:   dq 0, FRAME_NS
ts_gap:     dq 0, GAP_NS

section .bss
rng:        resd 1                      ; evolution PRNG
grng:       resd 1                      ; per-game PRNG (food placement)
seeds:      resd NGAMES
tseeds:     resd TEST_GAMES             ; fixed re-test games
eseeds:     resd EVAL_MAX
p_gens:     resd 1
p_seed:     resd 1
p_watch:    resd 1
p_games:    resd 1
fn_ptr:     resq 1
gen_no:     resd 1
render:     resd 1
up_flag:    resd 1
mode_play:  resd 1
champ_total: resd 1
ev_min:     resd 1
ev_max:     resd 1

pop:        resb POP * GSZ
newpop:     resb POP * GSZ
fit:        resd POP
apl:        resd POP
keys:       resq POP
champ:      resb GSZ
ldbuf:      resb GSZ + 1

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
g_x:        resd NIN                    ; net inputs
g_h:        resd NHID                   ; hidden activations
g_o:        resd 3                      ; outputs

line:       resb 512
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

; fill_seeds: in rdi = dword array, ecx = count (>0), eax = xorshift32 state.
; Independent of the global generators.
fill_seeds:
.l:
    mov     edx, eax
    shl     edx, 13
    xor     eax, edx
    mov     edx, eax
    shr     edx, 17
    xor     eax, edx
    mov     edx, eax
    shl     edx, 5
    xor     eax, edx
    mov     [rdi], eax
    add     rdi, 4
    dec     ecx
    jnz     .l
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

; fill g_x: danger left/straight/right, food left/ahead/right,
;           then (RAYS) look-ahead left/straight/right: is there a wall or body
;           within RAY_CAP cells (RAYBIN=1) or how many cells are free (RAYBIN=0)
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

%if RAYS
    xor     ecx, ecx
.ray:
    lea     eax, [r10 + rcx + 3]
    and     eax, 3
    movsx   esi, byte [dxs + rax]       ; step dx
    movsx   edi, byte [dys + rax]       ; step dy
    mov     edx, r8d                    ; ray x
    mov     eax, r9d                    ; ray y
    xor     r11d, r11d                  ; free cells so far
.rstep:
    add     edx, esi
    add     eax, edi
    cmp     edx, 15
    ja      .rdone                      ; wall
    cmp     eax, 15
    ja      .rdone
    mov     ebx, eax
    shl     ebx, 4
    add     ebx, edx
    cmp     byte [g_occ + rbx], 0
    jne     .rdone                      ; body
    inc     r11d
    cmp     r11d, RAY_CAP
    jb      .rstep
.rdone:
%if RAYBIN
    xor     eax, eax                    ; 1 = an obstacle within RAY_CAP cells
    cmp     r11d, RAY_CAP
    setb    al
    mov     [g_x + 24 + rcx * 4], eax
%else
    mov     [g_x + 24 + rcx * 4], r11d  ; free cells before the obstacle
%endif
    inc     ecx
    cmp     ecx, 3
    jb      .ray
%endif
    ret

; the net. in: rsi = genome. out: eax = 0 left, 1 straight, 2 right
forward:
    xor     ecx, ecx
.hid:
    movsx   eax, byte [rsi + OFF_B1 + rcx]  ; bias
    imul    edx, ecx, NIN
    add     rdx, rsi                    ; -> this unit's input weights
    xor     edi, edi
.hin:
    movsx   r8d, byte [rdx + rdi]
    imul    r8d, [g_x + rdi * 4]
    add     eax, r8d
    inc     edi
    cmp     edi, NIN
    jb      .hin
    test    eax, eax                    ; ReLU
    jns     .keep
    xor     eax, eax
.keep:
    mov     [g_h + rcx * 4], eax
    inc     ecx
    cmp     ecx, NHID
    jb      .hid

    xor     ecx, ecx
.out:
    movsx   eax, byte [rsi + OFF_B2 + rcx]
    imul    edx, ecx, NHID
    lea     rdx, [rsi + rdx + OFF_W2]   ; -> this output's hidden weights
    xor     edi, edi
.oin:
    movsx   r8d, byte [rdx + rdi]
    imul    r8d, [g_h + rdi * 4]
    add     eax, r8d
    inc     edi
    cmp     edi, NHID
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
    cmp     dword [render], 0           ; a live replay during training is cut short
    je      .nocut                      ; (the final demo / `snake play` is not)
    cmp     dword [mode_play], 0
    jne     .nocut
    cmp     dword [g_steps], REPLAY_STEPS
    jae     .dead
.nocut:
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

; evaluate: play one genome on `count` fixed games.
; in:  rsi = genome, rdx = dword array of game seeds, ecx = count (>0)
; out: eax = total apples; [ev_min]/[ev_max] = fewest/most apples in a game
; preserves rbx, rbp, r12-r15.
evaluate:
    push    r12
    push    r13
    push    r14
    push    r15
    mov     r12, rsi
    mov     r13, rdx
    mov     r14d, ecx
    xor     r15d, r15d
    mov     dword [ev_min], 0x7FFFFFFF
    mov     dword [ev_max], 0
    mov     dword [render], 0
.g:
    mov     eax, [r13]
    mov     [grng], eax
    mov     rsi, r12
    call    play
    add     r15d, edx
    cmp     edx, [ev_min]
    jge     .nomin
    mov     [ev_min], edx
.nomin:
    cmp     edx, [ev_max]
    jle     .nomax
    mov     [ev_max], edx
.nomax:
    add     r13, 4
    dec     r14d
    jnz     .g
    mov     eax, r15d
    pop     r15
    pop     r14
    pop     r13
    pop     r12
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
    cmp     dword [mode_play], 0
    je      .lbl_gen
    PUTS    s_champlbl
    jmp     .lbl_done
.lbl_gen:
    PUTS    s_gen
    mov     eax, [gen_no]
    call    utoa
.lbl_done:
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

; print_tenths: eax = value * 10; writes "I.D" at [rdi]. Clobbers rax, rcx, rdx.
print_tenths:
    xor     edx, edx
    mov     ecx, 10
    div     ecx                         ; eax = integer part, edx = tenths digit
    push    rdx
    call    utoa
    mov     byte [rdi], '.'
    inc     rdi
    pop     rax
    add     al, '0'
    mov     [rdi], al
    inc     rdi
    ret

; atoi: rsi -> decimal string. eax = value, CF=1 if it does not start with a digit
atoi:
    movzx   edx, byte [rsi]
    sub     edx, '0'
    cmp     edx, 9
    ja      .none
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
    clc
    ret
.none:
    stc
    ret

; die: write rsi/rdx to stderr, exit(1)
die:
    mov     edi, 2
    mov     eax, SYS_write
    syscall
    mov     edi, 1
    mov     eax, SYS_exit
    syscall

; load_genome: rsi -> file name. Reads exactly GSZ bytes into `champ`, or dies.
; clobbers r12, r13.
load_genome:
    mov     eax, SYS_open
    mov     rdi, rsi
    xor     esi, esi                    ; O_RDONLY
    xor     edx, edx
    syscall
    test    rax, rax
    js      .noopen
    mov     r12, rax
    mov     eax, SYS_read
    mov     rdi, r12
    mov     esi, ldbuf
    mov     edx, GSZ + 1                ; one extra byte detects a too-long file
    syscall
    mov     r13, rax
    mov     eax, SYS_close
    mov     rdi, r12
    syscall
    cmp     r13, GSZ
    jne     .badsize
    mov     esi, ldbuf
    mov     edi, champ
    mov     ecx, GSZ
    rep movsb
    ret
.noopen:
    mov     esi, e_open
    mov     edx, e_open_len
    jmp     die
.badsize:
    mov     esi, e_size
    mov     edx, e_size_len
    jmp     die

; =============================================================================
; main: r12 = genome index, r13 = inner counter, r14 = generation, r15 = fitness sum
;       rbp = apple sum
; =============================================================================
_start:
    mov     dword [p_gens], 500
    mov     dword [p_seed], 1
    mov     dword [p_watch], 100
    mov     dword [p_games], 100
    mov     qword [fn_ptr], fname
    cmp     qword [rsp], 2
    jb      .train
    mov     rsi, [rsp + 16]
    cmp     byte [rsi], 'p'
    je      .mode_play
    cmp     byte [rsi], 'e'
    je      .mode_eval
    call    atoi
    jc      .usage
    mov     [p_gens], eax
    cmp     qword [rsp], 3
    jb      .train
    mov     rsi, [rsp + 24]
    call    atoi
    jc      .usage
    mov     [p_seed], eax
    cmp     qword [rsp], 4
    jb      .train
    mov     rsi, [rsp + 32]
    call    atoi
    jc      .usage
    mov     [p_watch], eax
    jmp     .train

.usage:
    mov     esi, s_usage
    mov     edx, s_usage_len
    jmp     die

; ---- snake play [file] [seed] -------------------------------------------------
.mode_play:
    mov     dword [p_seed], 0
    cmp     qword [rsp], 3
    jb      .p_go
    mov     rax, [rsp + 24]
    mov     [fn_ptr], rax
    cmp     qword [rsp], 4
    jb      .p_go
    mov     rsi, [rsp + 32]
    call    atoi
    jc      .usage
    mov     [p_seed], eax
.p_go:
    mov     rsi, [fn_ptr]
    call    load_genome
    mov     eax, [p_seed]
    test    eax, eax
    jnz     .p_seeded
    rdtsc                               ; no seed given: use the CPU cycle counter
    xor     eax, edx
    jnz     .p_seeded
    mov     eax, 1
.p_seeded:
    mov     [rng], eax
    jmp     .demo_start

; ---- snake eval [file] [games] [seed] ----------------------------------------
.mode_eval:
    cmp     qword [rsp], 3
    jb      .e_go
    mov     rax, [rsp + 24]
    mov     [fn_ptr], rax
    cmp     qword [rsp], 4
    jb      .e_go
    mov     rsi, [rsp + 32]
    call    atoi
    jc      .usage
    mov     [p_games], eax
    cmp     qword [rsp], 5
    jb      .e_go
    mov     rsi, [rsp + 40]
    call    atoi
    jc      .usage
    mov     [p_seed], eax
.e_go:
    cmp     dword [p_games], 0
    jne     .e_g1
    mov     dword [p_games], 1
.e_g1:
    cmp     dword [p_games], EVAL_MAX
    jbe     .e_g2
    mov     dword [p_games], EVAL_MAX
.e_g2:
    cmp     dword [p_seed], 0
    jne     .e_g3
    mov     dword [p_seed], 1
.e_g3:
    mov     rsi, [fn_ptr]
    call    load_genome
    mov     edi, eseeds
    mov     ecx, [p_games]
    mov     eax, [p_seed]
    call    fill_seeds
    mov     esi, champ
    mov     edx, eseeds
    mov     ecx, [p_games]
    call    evaluate
    mov     r8d, eax                    ; total apples
    mov     edi, line
    PUTS    s_eval
    mov     rsi, [fn_ptr]               ; echo the file name
.e_cp:
    lodsb
    test    al, al
    jz      .e_cpd
    stosb
    jmp     .e_cp
.e_cpd:
    PUTS    s_colon
    mov     eax, [p_games]
    call    utoa
    PUTS    s_games
    mov     eax, r8d
    imul    eax, eax, 10
    xor     edx, edx
    div     dword [p_games]
    call    print_tenths
    PUTS    s_min
    mov     eax, [ev_min]
    call    utoa
    PUTS    s_max
    mov     eax, [ev_max]
    call    utoa
    PUTS    s_nl
    mov     esi, line
    mov     rdx, rdi
    sub     rdx, rsi
    mov     edi, 1
    mov     eax, SYS_write
    syscall
    jmp     .exit

; ---- snake [generations] [seed] [watch]: evolve ------------------------------
.train:
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
    mov     edi, tseeds                 ; the fixed games used to re-test candidates
    mov     ecx, TEST_GAMES
    mov     eax, TEST_STATE
    call    fill_seeds
    mov     dword [champ_total], -1
    mov     dword [mode_play], 0

    ; ---- random initial population: small weights in -32..31 ---------------
    xor     r13d, r13d
.init:
    mov     eax, r13d
    and     eax, GSZ - 1
    cmp     eax, GUSED
    jb      .rnd
    xor     eax, eax                    ; padding byte
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
    xor     r13d, r13d
.mkseeds:
    call    rand
    mov     [seeds + r13 * 4], eax
    inc     r13d
    cmp     r13d, NGAMES
    jb      .mkseeds

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
    shl     rax, GSHIFT
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

    ; ---- hall of fame: re-test this generation's best on the fixed fresh games
    xor     r13d, r13d
.hof:
    mov     rax, [keys + r13 * 8]
    and     eax, 255
    shl     eax, GSHIFT
    mov     esi, pop
    add     rsi, rax
    mov     edx, tseeds
    mov     ecx, TEST_GAMES
    call    evaluate
    cmp     eax, [champ_total]
    jle     .hof_next
    mov     [champ_total], eax          ; a new champion
    mov     rax, [keys + r13 * 8]
    and     eax, 255
    shl     eax, GSHIFT
    mov     esi, pop
    add     rsi, rax
    mov     edi, champ
    mov     ecx, GSZ
    rep movsb
.hof_next:
    inc     r13d
    cmp     r13d, RETEST_TOP
    jb      .hof

    ; ---- stats line: gen N best B mean M apples A champ C.D -----------------
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
    PUTS    s_champ
    mov     eax, [champ_total]
    imul    eax, eax, 10
    xor     edx, edx
    mov     ecx, TEST_GAMES
    div     ecx                         ; champion's apples per game, in tenths
    call    print_tenths
    PUTS    s_nl
    mov     esi, line
    mov     rdx, rdi
    sub     rdx, rsi
    mov     edi, 1
    mov     eax, SYS_write
    syscall

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
    shl     eax, GSHIFT
    mov     esi, pop
    add     rsi, rax
    mov     edi, r13d
    shl     edi, GSHIFT
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
    shl     eax, GSHIFT
    mov     esi, pop
    add     rsi, rax
    mov     edi, r13d
    shl     edi, GSHIFT
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
    cmp     ecx, GUSED
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

    cmp     dword [p_watch], 0
    je      .exit

    ; ---- endless demo (after training, and for `snake play`) ----------------
.demo_start:
    mov     dword [mode_play], 1
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
