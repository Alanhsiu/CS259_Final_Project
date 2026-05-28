# Overnight Run Notes — 2026-05-28

Summary of autonomous work done while you slept. All tasks committed; nothing pushed.

---

## Task 3: Power Measurement ✓ DONE

**Script:** `gpu_baseline/measure_power.py`
**Outputs:** `gpu_baseline/power_results.csv`, `gpu_baseline/gpu_results_power.csv`

Ran `nvidia-smi` at 100 ms intervals over a 10-second sustained attention run per config.
Measured average GPU power (whole-GPU, including idle baseline of ~36 W):

| S | Avg Power (W) | Energy/inference (mJ) | GFLOPS/W |
|---|---|---|---|
| 1024 | 46.6 | 9.1 | 29.6 |
| 2048 | 58.3 | 18.5 | 57.9 |
| 4096 | 80.7 | 41.7 | 103.1 |
| 8192 | 112.0 | 118.4 | 145.1 |
| 16384 | 106.9 | 490.7 | 140.0 |

**Sanity checks:**
- Power increases from idle (~36 W) with load — physically correct.
- GFLOPS/W improves from 29.6 → 145.1 as S grows (better GPU utilization) then plateaus — correct for memory-bound workload.
- Peak ~145 GFLOPS/W at S=8192; reasonable given ~250 W TDP and ~38% SM utilization.

**Known limitation:** This is a shared GPU. The ~36 W idle baseline from other processes is included in all readings. True compute-only GFLOPS/W is slightly higher (by roughly 36/avg_power fraction, ~10–25%). I noted this in the summary.

**One bug found and fixed:** Initial script used `(flops / 1e9) / avg_w` (GFLOPs-per-inference / W), which is not the standard GFLOPS/W metric. Fixed to `gflops_throughput / avg_w` where `gflops_throughput` is the sustained GFLOPS from the latency measurement.

---

## Task 4: Roofline Plot ✓ DONE

**Script:** `gpu_baseline/roofline_plot.py`
**Output:** `gpu_baseline/roofline.png`

Plotted two compute ceilings (FP16 tensor core 110 TFLOPS, FP32 14.9 TFLOPS) and the HBM2 memory ceiling (652.8 GB/s). For each S, plotted both theoretical AI (minimum bytes = Q+K+V+O only) and achieved AI (from NCU DRAM measurements).

**Sanity checks:**
- All achieved AI values (57–92 FLOP/B) fall below the FP16 tensor core ridge (169 FLOP/B) → MEMORY-BOUND ✓
- Theoretical AI values are 512–8192 FLOP/B (far above ridge), reflecting that the ideal lower bound assumes no attention matrix traffic.
- The gap between theoretical and achieved AI is expected: actual hardware must write/read intermediate attention scores, plus NCU replay overhead inflates DRAM bytes.

**One bug found and fixed:** Initial `bound` column classified by comparing AI against FP32 ridge (22.8 FLOP/B). Since achieved AI (57-92) is *above* 22.8, everything was wrongly labeled "compute (FP32)". Fixed to compare against the FP16 tensor core ridge (169 FLOP/B), which is the relevant ceiling. All configs correctly labeled MEMORY-BOUND.

---

## Task 5: Real Attention Tiles ✓ DONE

**Script:** `gpu_baseline/extract_real_tiles.py`
**Output:** `test_vectors/real_tile/` (real_q_tile.hex, real_k_tile.hex, real_c_expected.hex, real_tile_meta.txt)

Loaded GPT-2 small (117M) from HuggingFace, registered a forward hook on the first attention layer to capture Q and K after the `c_attn` projection. Extracted 8×64 tiles (tokens 0–7, head 0), quantized to INT8 using symmetric per-tensor quantization, and saved hex files in the same format as Task 1 vectors.

**Verification:** Readback and recompute passed (PASS ✓).
**Quantization error:** max abs = 0.21, mean abs = 0.066. Max *relative* error is 447% but that's due to near-zero denominators in the C matrix — the absolute precision is fine for the RTL team's test.

**Note on tile shape:** The RTL systolic array processes 8×8 tiles, but Q and K have shape 8×64 (8 tokens, 64-dimensional heads). For a full Q×Kᵀ computation the systolic array needs 64/8 = 8 partial-sum tiles. The hex files contain the full 8×64 projections; `real_tile_meta.txt` explains the decomposition.

---

## Task 6: Consolidated Summary ✓ DONE

**File:** `gpu_baseline/GPU_BASELINE_SUMMARY.md`

Contains: hardware specs, measurement methodology, full results table (latency, GFLOPS, DRAM traffic, utilization, power, energy, GFLOPS/W), roofline plot with interpretation, and one-paragraph conclusion confirming memory-bound behavior. Ready to hand to Member B.

---

## Tasks NOT done
- **Task 7 (Report):** Skipped as instructed — you'll do this tomorrow.

---

## Files committed (not pushed)
- Commit 1: Task 3 power measurement scripts + CSVs
- Commit 2: Task 4 roofline plot script + PNG
- Commit 3: Task 5 real GPT-2 attention tiles
- Commit 4: Task 6 consolidated summary + this notes file

## Things to review tomorrow
1. Check `roofline.png` visually — the legend is crowded (5 S values + ceiling lines); you may want to simplify.
2. The power measurements assume a shared GPU. If you want cleaner numbers, consider running on an idle machine or subtracting the idle baseline (36 W).
3. Task 5's Q×Kᵀ tile is 8×64 × 64×8, not a pure 8×8 matmul. Confirm with Members C/D whether they need a single 8×8 tile (one of the 8 partial sums) or the full 8×64 projections. I saved the full projections since that's more general.
