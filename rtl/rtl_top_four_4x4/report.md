# Comparison Report: 1×8×8 vs 4×4×4 vs 16×2×2 Systolic Array Attention Designs

All three designs implement the same single-head self-attention accelerator
(`Out = softmax(Q·Kᵀ / sqrt(d_k)) · V`, seq_len=8, d_k=64, NTILES=8)
using 64 PEs total, differing only in how the PEs are partitioned.

Synthesis: Design Vision `compile_ultra`, same library and constraints for all three.
All runs report `clock_network = 0`, so power figures are directly comparable.

---

## 1. Cycle Count

### Per-matmul-phase timing

Each matmul phase feeds a skewed wavefront into the systolic array.
With inner dimension K=8 per tile, the last accumulation occurs at cycle
`t = 2(N_ARRAY−1) + (K−1)`.

| | 1× 8×8 | 4× 4×4 | 16× 2×2 |
|-|--------|--------|---------|
| Array side (N\_ARRAY) | 8 | 4 | 2 |
| COMP\_DONE (last cycle) | 21 | 13 | 9 |
| MUL\_COMP duration | **22 cy** | **14 cy** | **10 cy** |
| FEED\_LAST | 14 | 10 | 8 |
| Savings vs 8×8 | — | −8 cy (−36%) | −12 cy (−55%) |

### Full pipeline breakdown (NTILES=8, hardware cycles only)

| FSM phase | 1× 8×8 | 4× 4×4 | 16× 2×2 |
|-----------|--------|--------|---------|
| Phase 1 · Q·Kᵀ (×8 tiles) | 8×24 = **192** | 8×16 = **128** | 8×12 = **96** |
| SCORE\_CAP | 1 | 1 | 1 |
| Row-wise softmax (×8 rows, 22 cy/row) | **176** | **176** | **176** |
| Phase 3 · attn·V | **24** | **16** | **12** |
| STREAM + DONE\_ST | **65** | **65** | **65** |
| **Total** | **458** | **386** | **350** |

Softmax (176 cy, 4+WRECIP+N = 4+9+8 = 21 cy/row × 8 rows) is
identical in all three designs and is the dominant bottleneck.

| | vs 1× 8×8 | vs 4× 4×4 |
|--|-----------|-----------|
| 4× 4×4 | **−72 cy (−15.7%)** | — |
| 16× 2×2 | **−108 cy (−23.6%)** | −36 cy (−9.3%) |

---

## 2. Area (synthesis results)

### Full attention\_top synthesis results

All three full `attention_top` designs (register files, softmax, FSM, buffers included).

| Metric | 1× 8×8 | 4× 4×4 | 16× 2×2 |
|--------|--------|--------|---------|
| Number of cells | 54,589 | 56,226 | 62,200 |
| Combinational cells | 44,594 | 46,266 | 52,229 |
| Sequential cells | 9,491 | 9,491 | 9,491 |
| Buf/Inv count | 8,343 | 8,616 | 10,619 |
| Combinational area (μm²) | 648,208 | 629,998 | 658,978 |
| Buf/Inv area (μm²) | 64,540 | 52,748 | 76,736 |
| Noncombinational area (μm²) | 314,431 | 313,420 | 313,381 |
| **Total cell area (μm²)** | **962,640** | **943,418** | **972,359** |
| Net interconnect area (μm²) | 6,958,222 | 7,111,616 | 7,586,685 |
| **Total area (μm²)** | **7,920,862** | **8,055,035** | **8,559,044** |

### Key observations

Sequential cell count is **identical** (9,491) across all three designs — the register files,
FSM, softmax unit, and output buffer are shared and unchanged. All area differences
come from combinational logic and routing:

- More arrays → more input multiplexing logic → more buf/inv → more combinational cells
- More arrays → more inter-array wires → larger net interconnect area
- The 8×8 single-array design has the **least routing overhead** (6,958,222 μm² net)
- The 16×2×2 sixteen-array design has the **most routing overhead** (7,586,685 μm², +9.0%)

| | vs 1× 8×8 total area | vs 4× 4×4 total area |
|--|----------------------|----------------------|
| 4× 4×4 | **+1.7%** | baseline |
| 16× 2×2 | **+8.1%** | +6.3% |

---

## 3. Power (synthesis results)

All three synthesis runs use the same methodology (`clock_network = 0` in all runs).
Figures are directly comparable.

### Raw data

| Power component | 1× 8×8 | 4× 4×4 | 16× 2×2 |
|----------------|--------|--------|---------|
| Cell Internal | 20.6285 mW (93%) | 20.5618 mW (93%) | 20.5735 mW (93%) |
| Net Switching | 1.4702 mW (7%) | 1.5087 mW (7%) | 1.5565 mW (7%) |
| **Total Dynamic** | **22.0988 mW** | **22.0705 mW** | **22.1300 mW** |
| Cell Leakage | 0.9379 mW | 0.9081 mW | 0.9224 mW |
| **Total Power** | **23.0365 mW** | **22.9785 mW** | **23.0523 mW** |

### Power by group

| Group | 1× 8×8 | 4× 4×4 | 16× 2×2 |
|-------|--------|--------|---------|
| clock\_network | 0.0000 mW | 0.0000 mW | 0.0000 mW |
| register | 20.7662 mW (90.1%) | 20.6890 mW (90.0%) | 20.6341 mW (89.5%) |
| combinational | 2.2702 mW (9.9%) | 2.2894 mW (10.0%) | 2.4183 mW (10.5%) |

### Power comparison

Total power is nearly identical across all three designs (within 0.3%):

| | vs 1× 8×8 | vs 4× 4×4 |
|--|-----------|-----------|
| 4× 4×4 | **−0.25%** | baseline |
| 16× 2×2 | **+0.07%** | +0.32% |

Register power dominates (~90%) and is driven by the shared register files
(Q\_buf, KT\_buf, V\_buf, accum\_rf, score\_rf, attn\_rf), which are identical
in all three designs. Combinational power increases slightly with more arrays
due to additional mux and interconnect buffering logic.

---

## 4. Summary

| Metric | 1× 8×8 | 4× 4×4 | 16× 2×2 |
|--------|--------|--------|---------|
| PE count | 64 | 64 | 64 |
| MUL\_COMP cycles | 22 | 14 | 10 |
| **Total HW cycles** | **458** | **386** | **350** |
| Cycle reduction vs 8×8 | — | −15.7% | −23.6% |
| **Total cell area (μm²)** | **962,640** | **943,418** | **972,359** |
| **Total area (μm²)** | **7,920,862** | **8,055,035** | **8,559,044** |
| Area increase vs 8×8 | baseline | +1.7% | +8.1% |
| **Total Power (mW)** | **23.04** | **22.98** | **23.05** |
| Power difference vs 8×8 | baseline | −0.25% | +0.07% |

### Efficiency: cycles × total area (area–latency product)

Lower is better — measures how much silicon is occupied per unit of computation time.

| | Cycles | Total area (μm²) | Cycles × Area | Normalised |
|--|--------|-----------------|--------------|-----------|
| 1× 8×8 | 458 | 7,920,862 | 3.628 × 10⁹ | 1.21× |
| 4× 4×4 | 386 | 8,055,035 | 3.109 × 10⁹ | 1.04× |
| **16× 2×2** | **350** | **8,559,044** | **2.996 × 10⁹** | **1.00×** |

### Efficiency: cycles × total power (energy per computation)

Lower is better — measures total energy consumed per attention computation.

| | Cycles | Total Power (mW) | Cycles × Power | Normalised |
|--|--------|-----------------|---------------|-----------|
| 1× 8×8 | 458 | 23.04 | 10,552 | 1.31× |
| 4× 4×4 | 386 | 22.98 | 8,870 | 1.10× |
| **16× 2×2** | **350** | **23.05** | **8,068** | **1.00×** |

---

## 5. Conclusion

| Criterion | Best design | Notes |
|-----------|------------|-------|
| Fewest HW cycles | **16× 2×2** (350 cy) | Shortest matmul phase (10 cy) |
| Lowest total area | **1× 8×8** (7,920,862 μm²) | Single array has least routing overhead |
| Lowest active power | **4× 4×4** (22.98 mW) | Marginal; all three within 0.3% |
| Best area–latency efficiency | **16× 2×2** (1.00×) | Cycle savings outweigh the 8% area overhead |
| Best energy efficiency | **16× 2×2** (1.00×) | 23% fewer cycles at essentially identical power |

The updated synthesis data shows that all three designs have **nearly identical active power**
(within 0.3%), making power a non-differentiating factor. All register files, the softmax
unit, FSM, and buffers are shared, so the dominant ~90% register power term is constant.

The real trade-off is **area vs. cycle count**:

- **1× 8×8**: Lowest total area (7.92 Mμm²) due to minimal routing overhead, but slowest
  (458 cy). The right choice when die area is the primary constraint.
- **4× 4×4**: +1.7% area, −15.7% cycles vs 8×8. The lowest-risk parallelisation step —
  small area penalty, meaningful cycle reduction.
- **16× 2×2**: +8.1% area, −23.6% cycles vs 8×8. Best energy efficiency and area–latency
  product; the routing overhead of sixteen small arrays is real but worthwhile for throughput.

**Recommended design:**
- **16× 2×2** when throughput or energy efficiency is the priority (best on both metrics).
- **4× 4×4** when die area is tightly constrained (only +1.7% area for −15.7% cycle saving).
- **1× 8×8** only if area must be minimised at all costs.

The softmax phase (176 cycles, 38–50% of total runtime depending on design) is unchanged
across all three and is the true bottleneck. Eliminating or pipelining softmax would benefit
all designs equally and unlock much larger throughput gains than any systolic array repartitioning.
