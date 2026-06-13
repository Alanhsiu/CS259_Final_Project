# Design Space Exploration of a Single-Head Attention Accelerator

> *Where do the cycles actually go when you build attention in silicon?*

A CS259 project that takes single-head INT8 scaled dot-product attention from a
GPU roofline all the way down to synthesized RTL — and asks a concrete architectural
question: given a **fixed budget of 64 processing elements**, how should you arrange
them, and where does the time really get spent?

📄 The full write-up lives in the paper. This repo holds everything behind it:
the GPU measurements, the analytical model, three synthesizable accelerator variants,
and the test vectors that validate them.

---

## TL;DR

- A GPU is *bored* doing single-head attention. On an NVIDIA TITAN V, SM utilization
  never tops **38%** and achieved arithmetic intensity (57–92 FLOP/B) stays well below
  the FP16 tensor-core ridge (169 FLOP/B). The workload is **data-movement limited**,
  not compute-bound — exactly the gap a fixed-function datapath can exploit.
- Splitting one `8×8` systolic array into sixteen `2×2` arrays shrinks the per-tile
  compute window from **22 → 10 cycles**, cutting total hardware cycles by **23.6%**
  (458 → 350) for just **+8.1% area** and **<0.3% power**.
- The twist: **softmax is the wall.** Fixed at **176 cycles** regardless of how you
  slice the array, it grows from 38.4% to **50.3%** of total runtime as the matmul
  phases shrink. Repartitioning the array eventually stops mattering — the divider and
  LUT do.

---

## What's Inside

We sweep two orthogonal design axes:

| Axis | Variants | What changes |
|------|----------|--------------|
| **Array size** | `4×4`, `8×8`, `16×16` | PE count grows quadratically, wavefront latency only linearly → utilization drops (57% → 36% → 21%) |
| **PE partitioning** (fixed 64 PEs) | `1×8×8`, `4×4×4`, `16×2×2` | Same silicon, shorter wavefront path to the corner PE → fewer cycles |

Everything is anchored to **Synopsys Design Vision synthesis at TSMC 0.13 µm, 100 MHz**,
and cross-checked against a **measured** TITAN V baseline (PyTorch SDPA, FP16).

---

## Headline Numbers

**PE partitioning (64 PEs, S=8, d_k=64):**

| Design | Compute window | Total cycles | Latency | Area vs 1×8×8 | Power |
|--------|---------------:|-------------:|--------:|--------------:|------:|
| `1×8×8`  | 22 cy | 458 | 4.58 µs | baseline | 23.04 mW |
| `4×4×4`  | 14 cy | 386 | 3.86 µs | +1.7% | 22.98 mW |
| `16×2×2` | 10 cy | **350** | **3.50 µs** | +8.1% | 23.05 mW |

`16×2×2` wins on both normalized **area–latency product** and **energy proxy** — the
cycle savings outweigh the routing overhead, and power is essentially flat (the shared
register files dominate ~90% of it).

**GPU baseline (TITAN V, FP16, D=64):**

| S | Latency | GFLOPS | SM util | GFLOPS/W |
|---|--------:|-------:|--------:|---------:|
| 1,024 | 0.195 ms | 1,380 | 5.1% | 29.6 |
| 4,096 | 0.516 ms | 8,322 | 21.0% | 103.1 |
| 8,192 | 1.057 ms | 16,250 | 38.0% | **145.1** |

---

## Repository Layout

```
gpu_baseline/            GPU measurement: latency/GFLOPS sweep, Nsight Compute
                         profiling, power/energy, roofline, GPT-2 real-tile extraction
analysis/                Analytical model — turns RTL cycle counts into latency,
                         area–latency product, and an energy proxy
rtl/
  rtl_4x4/               Standalone 4×4 systolic array + unit testbenches
  rtl_8x8/               Standalone 8×8 systolic array + testbench
  rtl_16x16/             Standalone 16×16 systolic array
  rtl_top/               Full single-head attention accelerator (1×8×8 baseline)
  rtl_top_four_4x4/      Four-4×4 partitioning variant
  rtl_top_sixteen_2x2/   Sixteen-2×2 partitioning variant
  comparison/            RTL-vs-GPU comparison scripts and figures
test_vectors/            INT8 matmul test vectors (sizes 4/8/16) + a real
                         quantized GPT-2 attention tile for validation
```

---

## Quick Start

### 1. GPU baseline *(needs an NVIDIA GPU + PyTorch/CUDA)*

```bash
cd gpu_baseline
python gpu_baseline.py        # latency / GFLOPS sweep over S
python ncu_profile.py         # Nsight Compute memory profiling — run in tmux, it's slow
python measure_power.py       # nvidia-smi power logging → energy, GFLOPS/W
python roofline_plot.py       # roofline chart
```

### 2. Test vectors for the RTL team

```bash
python generate_test_vectors.py    # 10 random + 3 edge tiles per size, into test_vectors/
```

Each tile ships as `$readmemh`-ready hex (`_A`, `_B`, `_C_expected`). There's also a
**real GPT-2 small** attention tile (head 0, tokens 0–7, per-tensor INT8 quantized) so
the design is validated on more than synthetic data.

### 3. Analytical model

```bash
cd analysis/member_b_final
python analytical_model.py    # reproduces the FSM cycle breakdown, derives PPA metrics
```

### 4. RTL simulation

```bash
# Full attention accelerator (IEEE 1800-2009 compatible simulator)
cd rtl/rtl_top
<simulator> attention_tb.sv attention_top.sv attention_fsm.sv \
    pe.sv systolic_array_8x8.sv buffer_output.sv softmax_unit.sv
```

The testbench runs two checks: a closed-form **uniform-attention** sanity test (every
output should equal 448) and a **real-data** test validating `score_rf` against an
external golden `Q·Kᵀ` reference.

---

## How the Accelerator Works

```
Out = softmax( Q·Kᵀ / √d_k ) · V
```

One `8×8` systolic array is **time-multiplexed** across both matmuls — a deliberate
area-vs-latency trade (a second array would double the PE count). Between them sits a
fixed-point softmax unit: row-max subtraction for stability, a 256-entry LUT for `exp`,
and a 9-bit restoring divider for normalization. The `1/√d_k` scaling is a free
right-shift instead of a real divider. A 12-state FSM sequences the whole pipeline —
load, tiled `Q·Kᵀ`, softmax, `attn·V`, stream — and the *same* FSM drives all three
partitioning variants by changing only `N_ARRAY`, `COMP_DONE`, and `FEED_LAST`.

---

## Workload

- Single-head scaled dot-product attention, `D = 64`, batch = 1
- **GPU:** FP16, prefill, `S ∈ {1024, 2048, 4096, 8192, 16384}`
- **Accelerator:** INT8 inputs / ~20-bit accumulation, fixed `S = 8` tile, `d_k = 64`
  decomposed into 8 K-tiles

---

## Author

**Cheng-Hsiu Hsieh** — University of California, Los Angeles
Project topic, design-space scope, GPU baseline, and RTL test infrastructure.

*With teammates Anqi Yang (RTL design), Ching-Yu Cheng (analytical model & PPA),
and Chi-Yu Chen (PE optimization & sub-array partitioning).*

---

## License

For coursework and research use.