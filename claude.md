# CS259 Final Project — GPU Baseline (Member A)

## My Role
I am Member A. My responsibilities are:
1. GPU baseline measurement (cuDNN/cuBLAS attention performance)
2. Test vector generation (for the RTL team's Verilog testbench)
3. Writing the GPU baseline section of the report

Other members: B does the analytical model, C and D build a custom systolic
array accelerator in RTL. I do NOT work on RTL, gem5, or the analytical model.

## Project Goal
Compare a custom systolic array accelerator (built by C/D in RTL) against a
GPU baseline (my job) running the same attention workload. The headline metric
is energy efficiency (GFLOPS/W) and area efficiency. The professor approved:
GPU baseline via cuDNN/cuBLAS + custom RTL + analytical model. gem5 and
hand-written CUDA kernels were explicitly dropped.

## Repository Structure
This is a shared team repo. My work (Member A, GPU baseline) lives entirely in
`gpu_baseline/`. Test vectors are kept in a separate top-level folder
`test_vectors/` so the RTL team (C, D) can access them independently without
touching my code. Do NOT create files outside these two folders unless I
explicitly ask.

## Git Workflow
- After completing each task and verifying it works, automatically stage and
  commit the relevant files with a clear, descriptive commit message.
- Use conventional commit style, e.g.:
  - "Add INT8 matmul test vectors for RTL team"
  - "Add ncu profiling script and merge DRAM metrics into results CSV"
- Only commit files within `gpu_baseline/` and `test_vectors/` (my own work).
- Do NOT push automatically — I will push manually after reviewing.
- Do NOT commit large data files like raw ncu reports (*.ncu-rep) or
  power logs; add them to .gitignore instead.

## Environment
- Do NOT use `source ~/myenv/bin/activate` in commands. It triggers a
  permission prompt every time because source evaluates shell code. Instead, call the venv's python directly by its absolute path:
- Hardware: NVIDIA TITAN V, 12 GB HBM2, CUDA 12.4
- PyTorch: 2.6.0+cu124
- TITAN V key specs (verify before using in roofline):
  - HBM2 bandwidth: ~652 GB/s
  - FP32 peak: ~14.9 TFLOPS
  - FP16 tensor core peak: ~110 TFLOPS (deep learning)
  - 80 SMs

## Workload
- Single-head attention prefill, D=64, FP16, batch=1
- Primary config: S=4096; sweep S = [1024, 2048, 4096, 8192, 16384]
- FLOP count formula: 4 * S * S * D
  (Q*K^T = 2*S*S*D, Attn*V = 2*S*S*D)

## Conventions
- No Chinese comments in code. English only.
- Always run scripts after writing to verify they work.
- Save all measurement data to CSV.
- Save plots as high-res PNG (dpi=150+).
- For long-running tasks (ncu, power logging), remind me to use tmux.

## Current Status (as of now)
DONE:
- gpu_baseline.py: sweeps S, measures latency (mean/std over 100 runs),
  computes GFLOPS, peak memory; writes gpu_results.csv
- plot_results.py: plots latency vs S and GFLOPS vs S; saves gpu_results.png

NOT DONE (the task list below):
- ncu profiling (DRAM traffic, SM/DRAM utilization)
- power measurement (for GFLOPS/W)
- roofline plot
- test vector generation (RTL team is blocked on this — high priority)
- report section

---

# TASK LIST (do these in order; ask me before destructive ops)

## Task 1 — Test vectors for RTL team (HIGHEST PRIORITY)
The RTL team (C, D) is blocked waiting on this. They want to sweep multiple
systolic array sizes (4x4, 8x8, 16x16), so test vectors must be generated for
each size. Write `generate_test_vectors.py` that:

- Uses np.random.seed(42) for reproducibility
- For each tile_size in [4, 8, 16]:
  - Generates 10 random tile_size x tile_size INT8 matrix pairs (A, B) in
    range [-128, 127]
  - Computes C = A @ B in INT32 (use .astype(np.int32) before multiply)
  - Also generates 3 edge cases for that size: all-zeros, identity matrix,
    all-127 (overflow test)
- Saves each matrix as a hex file, one number per line, row-major order:
  - A, B as INT8 -> 2 hex chars per line
  - C as INT32 -> 8 hex chars per line
- Organizes output by size into subfolders:
```
  test-vectors/
    size_4/
      random/  tile_00_A.hex, tile_00_B.hex, tile_00_C_expected.hex, ...
      edge/    zeros_A.hex, zeros_B.hex, zeros_C_expected.hex,
               identity_*.hex, maxval_*.hex
    size_8/
      random/  ...
      edge/    ...
    size_16/
      random/  ...
      edge/    ...
```
- Writes a single README.md at the test-vectors/ root explaining the format,
  the size subfolders, and a Verilog $readmemh usage snippet
- After generating, verify correctness: read back one hex file per size,
  reconstruct the matrices in numpy, recompute A@B, and assert it matches the
  saved C_expected. Print a PASS/FAIL summary per size.

## Task 2 — ncu profiling (~1-2 hrs incl. ncu runtime)
ncu replays kernels and is slow; remind me to run this in tmux.
For each S in the sweep, collect these metrics on the SDPA kernel:
- dram__bytes_read.sum
- dram__bytes_write.sum
- sm__throughput.avg.pct_of_peak_sustained_elapsed
- gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
- sm__warps_active.avg.pct_of_peak_sustained_active
Write a script that runs ncu per config, parses the output (CSV or text),
and merges these columns into gpu_results.csv (or a new gpu_results_ncu.csv).
Note: ncu may need to profile a single forward pass per config to keep runtime
reasonable — adjust the script so the profiled run does few iterations.
From dram bytes, compute achieved arithmetic intensity = FLOPs / total_bytes
and add it as a column.

## Task 3 — Power measurement (~1 hr)
Measure GPU power during sustained attention runs to compute energy and
GFLOPS/W. Approach:
- Launch nvidia-smi power logging in the background:
  nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu --format=csv -lms 100
- Run gpu_baseline.py in a loop long enough (e.g., several seconds per S) to
  get stable power readings
- Parse power_log.csv, compute average power during the active window per config
- Compute energy_mj = avg_power_w * latency_ms, and gflops_per_watt
- Add power_w, energy_mj, gflops_per_watt columns to the results CSV
Write this as a script that automates the logging + run + parse + merge.

## Task 4 — Roofline plot (~1 hr)
Write roofline_plot.py that produces a proper roofline chart:
- X axis: arithmetic intensity (FLOPs/byte), log scale
- Y axis: GFLOPS, log scale
- Draw the memory ceiling (diagonal: AI * peak_bandwidth) and compute ceiling
  (horizontal: peak_FLOPS) using TITAN V specs
- Plot two sets of points for each S config:
  (a) theoretical arithmetic intensity (FLOPs / minimum bytes to read inputs
      and write outputs once)
  (b) achieved arithmetic intensity (FLOPs / measured DRAM bytes from Task 2)
- Label each point with its S value
- Annotate whether each config is compute-bound or memory-bound
- Save as roofline.png (dpi=150)
Use FP16 tensor core peak (~110 TFLOPS) and standard FP16 peak (~31 TFLOPS)
as two reference compute ceilings if helpful, and explain the choice.

## Task 5 — Real attention tiles (OPTIONAL, nice-to-have, ~1 hr)
If time permits, dump real Q/K/V attention tiles from a small pretrained model
(e.g., a HuggingFace BERT or GPT-2 layer), quantize to INT8, and extract one
8x8 tile from a Q*K^T computation as an additional realistic test case for the
RTL team. This strengthens the report ("validated on real attention tiles").
Skip if other tasks are not done yet.

## Task 6 — Consolidate results summary (~30 min)
Create GPU_BASELINE_SUMMARY.md for the team containing:
- Setup (hardware, software, workload, methodology)
- A results table (latency, GFLOPS, DRAM traffic, utilization, power,
  energy, GFLOPS/W for each S)
- The roofline plot
- A one-paragraph conclusion: is the GPU compute-bound or memory-bound for
  attention, and what is its energy efficiency
This is what I hand to Member B (for comparison) and the report writer.

## Task 7 — Report section (~10 hrs, I will mostly write prose myself)
Help me draft the GPU Baseline section of the report (target 3-4 pages):
- 3.1 Methodology: hardware specs, cuDNN/SDPA, measurement protocol
- 3.2 Performance characterization: latency scaling, GFLOPS, utilization
- 3.3 Roofline analysis: compute vs memory bound, theoretical vs achieved AI
- 3.4 Energy efficiency: power, energy per inference, GFLOPS/W
Produce clean figures and a results table in LaTeX or Markdown. I will refine
the writing; you produce the structure, figures, and data-driven sentences.

## Optional improvement (mention but don't auto-apply)
The current timing uses time.perf_counter() + synchronize() per call, which
includes some CPU launch overhead, especially for small S. A more precise
alternative is CUDA events (torch.cuda.Event with elapsed_time). Suggest this
as an option but ask before changing the existing measurement logic, since the
current numbers are already usable.