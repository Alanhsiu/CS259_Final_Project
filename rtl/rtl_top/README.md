# Single-Head Self-Attention Accelerator — RTL Design

## Overview

This directory implements a complete **single-head self-attention** accelerator in SystemVerilog targeting an 8×8 systolic array.  The design computes:

```
Out = softmax( Q · Kᵀ / sqrt(d_k) ) · V
```

where Q, K, V are all **N×N INT8** matrices (N=8, d_k=8, seq\_len=8).  
K must be pre-transposed by software before loading.

---

## Architecture

### Block Diagram

```
  Host
   │  load Q / Kᵀ / V  (load_en, load_sel, load_row, load_col, load_data)
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        attention_top                                  │
│                                                                       │
│  Q_buf [8×8]   ──┐                                                   │
│  KT_buf[8×8]   ──┤──► skew mux ──► systolic_array_8x8 ──► c_data   │
│  V_buf [8×8]   ──┘  (phase sel)      (8×8 PE grid)         │        │
│                          ▲                                   │ Phase1 │
│                          │ reload in                         ▼        │
│  attn_rf[8×8] ──(Q0.6)──┘ Phase 2            score_rf[8×8] ◄──────  │
│  (Q1.15 softmax output)                        (>>>1 scaled)  │       │
│                                                               │       │
│                                          softmax_unit ◄───────┘       │
│                                          (N=8, row-wise)     │        │
│                                                               ▼        │
│                                                      attn_rf[8×8]     │
│                                                      (Q1.15)  │       │
│                                                      →Q0.6 truncate   │
│                                                               │        │
│  attention_fsm ──────────────────────────── controls all ────┘        │
│                                                                        │
│  buffer_output ──► out_data / out_valid / out_ready ──► Host          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## File Inventory

| File | Role | Source |
|------|------|--------|
| `pe.sv` | Processing element: signed 8×8 MAC, 20-bit accumulator | copied from rtl_8x8 |
| `systolic_array_8x8.sv` | 8×8 grid of PEs with horizontal/vertical skew buses | copied from rtl_8x8 |
| `softmax_unit.sv` | Row-wise softmax with LUT-based exp(), restoring divider | copied from rtl; used with WIN=20 |
| `buffer_output.sv` | Captures 64-element result array, streams one word per cycle | copied from rtl_8x8 |
| `attention_fsm.sv` | **New** — multi-phase master sequencer | new |
| `attention_top.sv` | **New** — top-level integration with all internal memories | new |
| `attention_tb.sv` | Two-test simulation suite | new |

---

## Compute Phases

The design time-multiplexes one systolic array across two matrix multiplications, separated by a row-wise softmax pass.

### Phase 1 — Q · Kᵀ  (22 cycles)

```
a_data[r] = Q_buf [r][cycle_cnt - r]       (row-skewed)
b_data[c] = KT_buf[cycle_cnt - c][c]       (col-skewed)
```

The systolic wavefront takes 15 cycles to fill (cycles 0–14) and 7 more drain cycles (15–21).  
On cycle 21, `c_data[r*8+c]` holds the final accumulation for every output element.

After MUL1: `score_rf[r][c] ← c_data[r*8+c] >>> 1`  
The arithmetic right-shift by 1 approximates division by `sqrt(8) ≈ 2.83`.  
(Hardware cost: zero — pure wiring.)

### Phase 2 — Row-wise Softmax  (8 rows × ~22 cycles = ~176 cycles)

The `softmax_unit` is reused sequentially for all 8 rows of `score_rf`.  
The unit implements numerically stable softmax via:

1. **Max subtraction** — subtract row maximum before exponentiation (prevents overflow).
2. **Exp approximation** — `exp(-z) = EXP_ROM[w_frac] >> w_int`  
   using `w = z · log₂(e)` decomposed into integer and fractional parts (256-entry Q1.15 LUT).
3. **Reciprocal** — 9-cycle right-shift restoring divider computes `2²³ / sum_exp`.
4. **Normalization** — sequential multiply of each `exp_val[i] × quotient`, output in Q1.15.

Output: `attn_rf[row][0..7]` — 8×8 attention weights, Q1.15 format.

### Quantization Bridge (Q1.15 → 8-bit for Phase 3)

The PE expects signed 8-bit inputs and sign-extends `a_in[7]`.  
To prevent negative weight interpretation, attention weights are converted to **Q0.6** (7-bit unsigned):

```
attn_q07[r][c] = attn_rf[r][c][15] ? 8'h7F          // saturate at 1.0
               : {1'b0, attn_rf[r][c][14:8]}          // bits [14:8], MSB=0
```

This forces `a_in[7] = 0` so PE sign-extension is always a zero-extension.  
Scale factor: `1/128` — recover INT8 scale by right-shifting final output by 7.

### Phase 3 — attn\_q07 · V  (22 cycles)

Same systolic skew pattern, now drawing from `attn_q07` and `V_buf`:

```
a_data[r] = attn_q07[r][cycle_cnt - r]
b_data[c] = V_buf   [cycle_cnt - c][c]
```

Output: 20-bit signed accumulator values, streamed through `buffer_output`.

---

## FSM State Sequence

```
IDLE
 └─► MUL1_CLR   (1 cy)   clear PE accumulators
      └─► MUL1_COMP  (22 cy)  compute Q · Kᵀ
           └─► SCORE_CAP  (1 cy)   capture + scale c_data → score_rf
                └─► SOFT_START (1 cy) ┐  repeated
                     └─► SOFT_WAIT   ─┘  for rows 0..7
                          └─► MUL2_CLR  (1 cy)   clear PE accumulators
                               └─► MUL2_COMP (22 cy)  compute attn · V
                                    └─► RESULT_CAP (1 cy)  capture → buffer_output
                                         └─► STREAM  (64 cy)  stream results
                                              └─► DONE_ST (1 cy)
```

Total latency ≈ **286 cycles** (excluding load time).

---

## Loading Interface

All three matrices are loaded before asserting `start`.  
Each element is written one byte at a time:

| `load_sel` | Target buffer |
|-----------|--------------|
| `2'b00` | Q\_buf |
| `2'b01` | KT\_buf (K must be pre-transposed by software) |
| `2'b10` | V\_buf |

Loading 3 × 64 = 192 elements takes 192 cycles at one write per cycle.

---

## Output Format

`out_data` is a 20-bit signed word representing one element of `attn · V`.  
Due to Q0.6 quantization of the attention weights, the accumulator is scaled by 128.  
To recover INT8-range values: `result_int8 = out_data >>> 7`.

---

## Key Design Decisions

### Why share one systolic array for both matmuls?

Area is the primary constraint.  A second 8×8 array would double the PE count (128 PEs instead of 64).  Time-multiplexing costs latency but saves ~50% of the datapath area.

### Why pre-transpose K in software?

A hardware transpose buffer for 8×8 INT8 would require 64 registers and routing.  Since K is loaded from an external memory whose contents can be pre-arranged, the cost is zero in hardware.

### Why Q0.6 (7-bit) instead of Q0.7 (8-bit) for attention weights?

The PE's sign-extension logic (`a_ext = {{8{a_in[7]}}, a_in}`) assumes the input is signed.  A Q0.7 value with the top bit set (weights ≥ 0.5 mapped to 128–254) would be sign-extended to a negative number, producing wrong products.  Forcing `a_in[7] = 0` (Q0.6, values 0–127) eliminates this at the cost of 1 bit of weight precision.

### Why `>>> 1` for scaling instead of a divider?

`sqrt(8) ≈ 2.83`.  An arithmetic right-shift by 1 divides by 2, introducing a ~41% scaling error.  However, softmax is order-preserving: it depends only on **relative** score differences, not absolute magnitudes.  A uniform scaling factor shifts all softmax inputs identically, changing the sharpness slightly but not the argmax or the overall attention pattern.  This saves the area of a pipelined divider.

### Softmax numerical stability

Standard softmax `exp(x)/Σexp(x)` overflows for large positive inputs.  The unit subtracts the row maximum before exponentiation (`exp(x - max)`), which is mathematically equivalent and keeps all inputs ≤ 0, preventing overflow in the Q1.15 representation.

---

## Simulation

Compile and run with Xcelium (or any IEEE 1800-2009 compatible simulator):

```bash
xmverilog attention_tb.sv attention_top.sv attention_fsm.sv \
           pe.sv systolic_array_8x8.sv buffer_output.sv softmax_unit.sv \
           +access+r
```

### Test 1 — Uniform attention (closed-form)

Q = K = all-1s, V\[r\]\[c\] = r.  Expected: all 64 outputs = **448** (exact).

### Test 2 — Real data

Reads `real_q_tile.hex` and `real_k_tile.hex` (first 8×8 tile, row-major INT8).  
Transposes K in the testbench.  
Verifies `score_rf` against a software-computed golden model built from the same tile data.  
Prints `c_expected` (full d\_k=64 accumulated reference) for context.

---

## Limitations

| Limitation | Impact |
|-----------|--------|
| No tiling | Only handles seq\_len = d\_k = 8 directly; larger matrices require external tile accumulation |
| Single head | No multi-head parallelism; heads must be serialized |
| Q0.6 weight precision | 7-bit weight resolution instead of 8-bit; final output needs `>>> 7` correction |
| Scale approximation | `/2` instead of `/sqrt(8)`; softmax sharpness differs slightly from float reference |
| Sequential softmax | 8 rows processed serially (176 cycles); parallel rows would reduce this to 22 cycles at 8× area cost |
