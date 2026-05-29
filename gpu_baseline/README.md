# gpu_baseline — Member A scripts

Single-head FP16 attention (D=64, batch=1) on NVIDIA TITAN V.

## Directory layout

```
gpu_baseline/
  data/              CSV outputs (inputs for downstream scripts)
  figures/           PNG plots
  *.py               Scripts (described below)
  README.md
  GPU_BASELINE_SUMMARY.md
```

---

## Scripts

### `gpu_baseline.py` — Task 1 (latency + GFLOPS baseline)

Sweeps S ∈ {1024, 2048, 4096, 8192, 16384}, runs 100 warmup + 100 timed
`torch.nn.functional.scaled_dot_product_attention` iterations, records
mean/std latency and sustained GFLOPS.

| | |
|---|---|
| **Reads** | *(none)* |
| **Writes** | `data/gpu_results_base.csv` |
| **Dependencies** | *(none — run first)* |

```bash
~/myenv/bin/python gpu_baseline.py
```

---

### `ncu_profile.py` — Task 2 (Nsight Compute memory profiling)

Profiles the SDPA kernel with `ncu` for each S config. Captures L1→L2 read
bytes, L1→L2 write bytes, SM throughput %, DRAM throughput %, and warp
active %. Computes achieved arithmetic intensity.

**Warning:** ncu replays every kernel — expect 10–30 min. Run in tmux.

| | |
|---|---|
| **Reads** | `sdpa_one_pass.py` (invoked as subprocess) |
| **Writes** | `data/gpu_results_ncu.csv` |
| **Dependencies** | *(none — standalone)* |

```bash
~/myenv/bin/python ncu_profile.py
```

---

### `merge_results.py` — merges base + ncu into a clean summary table

Joins `gpu_results_base.csv` and `gpu_results_ncu.csv` on S key, converts
bytes to MB, and writes a concise merged CSV.

| | |
|---|---|
| **Reads** | `data/gpu_results_base.csv`, `data/gpu_results_ncu.csv` |
| **Writes** | `data/gpu_results_full.csv` |
| **Dependencies** | `gpu_baseline.py`, `ncu_profile.py` |

```bash
~/myenv/bin/python merge_results.py
```

---

### `measure_power.py` — Task 3 (power, energy, GFLOPS/W)

Runs nvidia-smi at 100 ms intervals while sustaining attention for ~10 s
per S config. Parses timestamps to window readings to each config's active
period. Computes avg/min/max power, energy (mJ = W × ms), and GFLOPS/W.
Merges everything into the final results CSV.

Note: `power_log.csv` is gitignored (transient raw log).

| | |
|---|---|
| **Reads** | `data/gpu_results_base.csv`, `data/gpu_results_ncu.csv` |
| **Writes** | `data/power_results.csv`, `data/gpu_results_final.csv` |
| **Dependencies** | `gpu_baseline.py`, `ncu_profile.py` |

```bash
~/myenv/bin/python measure_power.py
```

---

### `roofline_plot.py` — Task 4 (roofline chart)

Draws TITAN V memory and compute ceilings (FP16 tensor core 110 TFLOPS,
FP32 14.9 TFLOPS, HBM2 652.8 GB/s). Plots theoretical AI (Q+K+V+O only)
and achieved AI (measured DRAM bytes) for each S config.

| | |
|---|---|
| **Reads** | `data/gpu_results_base.csv`, `data/gpu_results_ncu.csv` |
| **Writes** | `figures/roofline.png` |
| **Dependencies** | `gpu_baseline.py`, `ncu_profile.py` |

```bash
~/myenv/bin/python roofline_plot.py
```

---

### `extract_real_tiles.py` — Task 5 (real GPT-2 attention tiles for RTL team)

Loads GPT-2 small from HuggingFace, hooks the first attention layer to
capture Q/K projections, extracts an 8×64 tile (tokens 0–7, head 0),
quantizes to INT8 (symmetric per-tensor), and verifies INT32 matmul
readback matches expected output.

| | |
|---|---|
| **Reads** | HuggingFace `gpt2` weights (downloaded on first run) |
| **Writes** | `../test_vectors/real_tile/real_q_tile.hex`, `real_k_tile.hex`, `real_c_expected.hex`, `real_tile_meta.txt` |
| **Dependencies** | `transformers` package (`pip install transformers`) |

```bash
~/myenv/bin/python extract_real_tiles.py
```

---

## Recommended run order

```
1. gpu_baseline.py        # ~1 min
2. ncu_profile.py         # ~10-30 min in tmux
3. merge_results.py       # <1 s
4. measure_power.py       # ~1 hr (sustained GPU run)
5. roofline_plot.py       # <1 s
6. extract_real_tiles.py  # ~1 min (downloads ~500 MB on first run)
```

## Key results (TITAN V, FP16, D=64, batch=1)

All configs are **memory-bound**: achieved AI 57–92 FLOP/B < FP16 tensor core ridge 169 FLOP/B.
Peak efficiency: **145 GFLOPS/W** at S=8192.

See `GPU_BASELINE_SUMMARY.md` for the full results table and roofline analysis.
