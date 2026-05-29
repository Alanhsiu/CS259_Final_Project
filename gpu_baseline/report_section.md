# Section 3: GPU Baseline

## 3.1 Methodology

### Hardware platform

All GPU baseline measurements were collected on an NVIDIA TITAN V GPU.
Table 1 summarises the relevant hardware parameters used to construct the
roofline model.

**Table 1: NVIDIA TITAN V hardware specifications.**

| Parameter | Value |
|-----------|-------|
| Architecture | Volta (GV100, sm_70) |
| HBM2 peak bandwidth | 652.8 GB/s |
| FP32 peak throughput | 14.9 TFLOPS |
| FP16 tensor core peak | 110 TFLOPS |
| Streaming multiprocessors | 80 |
| L2 cache | 4.5 MB |
| VRAM | 12 GB HBM2 |
| Thermal design power | ~250 W |

### Workload

The workload is single-head scaled dot-product attention (SDPA) in prefill
mode, matching the inner computation performed by the systolic array
accelerator built by Members C and D. The operation is:

```
Attention(Q, K, V) = softmax(Q Kᵀ / √D) · V
```

with batch size 1, one attention head, and head dimension D = 64, using
FP16 precision throughout. The sequence length S is swept over
{1024, 2048, 4096, 8192, 16384}. The theoretical FLOPs count is
4·S²·D (two matrix multiplications of cost 2·S²·D each).

### Software

PyTorch 2.6.0+cu124 was used, invoking
`torch.nn.functional.scaled_dot_product_attention`. On Volta GPUs PyTorch
dispatches to the memory-efficient attention kernel from the xformers
library (`fmha_cutlassF_f16_aligned_*_sm70`), which implements a
FlashAttention-style tiled algorithm that avoids materialising the full
S×S attention matrix in DRAM.

### Latency measurement

For each S, the kernel is executed 100 times after 100 warmup iterations.
Each iteration is bracketed by `torch.cuda.synchronize()` calls and timed
with `time.perf_counter()`. The mean and standard deviation of the 100
timed iterations are reported. This approach includes CPU launch overhead
(estimated at ~10–20 µs per call), which is significant only for the
smallest S=1024 config.

### Memory traffic measurement

DRAM traffic was captured using NVIDIA Nsight Compute (`ncu`). Because ncu
replays every kernel to collect counter data, a single forward pass per S
config was profiled to keep runtime manageable. The metrics collected were:

- `l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld.sum` — bytes read from L2
  into the L1 cache (global load traffic)
- `l1tex__m_l1tex2xbar_write_bytes_mem_lg_op_st.sum` — bytes written from
  L1 to L2 (global store traffic)
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — SM utilisation
- `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` — DRAM
  utilisation as seen by the memory controller

Arithmetic intensity (AI) is computed as FLOPs divided by the sum of L1
read and write bytes. This L1→L2 traffic metric captures actual kernel
data movement and is appropriate for flash-attention-style kernels, which
deliberately avoid evicting the attention matrix to DRAM.

### Power measurement

GPU power was logged at 100 ms intervals using
`nvidia-smi --query-gpu=timestamp,power.draw --format=csv -lms 100`. For
each S config the attention kernel was run in a tight loop for
approximately ten seconds; the power readings whose timestamps fall within
that window are averaged to obtain the sustained average power. Energy per
inference is then computed as E = P_avg · t_lat (watts × milliseconds =
millijoules). The GPU is shared with other lab users; the idle baseline
(~36 W) is included in all readings, so reported power figures represent a
conservative (slightly pessimistic) estimate of the attention kernel's true
power draw.

---

## 3.2 Performance Characterisation

### Full results table

**Table 2: GPU baseline results for single-head FP16 SDPA (D=64, batch=1).**

| S | FLOPs (G) | Latency (ms) | Std (ms) | Throughput (GFLOPS) | L1 traffic (MB) | Achieved AI (FLOP/B) | SM util (%) | Avg power (W) | Energy (mJ) | GFLOPS/W |
|---|---|---|---|---|---|---|---|---|---|---|
| 1,024 | 0.27 | 0.195 | 0.051 | 1,380 | 4.70 | 57.1 | 5.1 | 46.6 | 9.1 | 29.6 |
| 2,048 | 1.07 | 0.318 | 0.031 | 3,376 | 17.7 | 60.7 | 10.4 | 58.3 | 18.5 | 57.9 |
| 4,096 | 4.29 | 0.516 | 0.007 | 8,323 | 70.3 | 61.1 | 21.0 | 80.7 | 41.7 | 103.1 |
| 8,192 | 17.18 | 1.057 | 0.010 | 16,250 | 186.7 | 92.0 | 38.0 | 112.0 | 118.4 | 145.1 |
| 16,384 | 68.72 | 4.590 | 0.164 | 14,973 | 941.9 | 73.0 | 32.9 | 106.9 | 490.7 | 140.0 |

*L1 traffic = sum of L1→L2 read and write bytes measured by Nsight Compute.*
*Achieved AI = FLOPs / L1 traffic.*

### Latency and throughput scaling

Latency grows sub-quadratically with sequence length: the 16× increase in
S from 1,024 to 16,384 produces only a 23.6× increase in latency, far
below the 256× that would be expected if the GPU were fully utilised at all
sizes. The explanation is straightforward: at S=1,024, SM utilisation is
only 5.1%, meaning the GPU is heavily underloaded and execution time is
dominated by kernel-launch overhead and pipeline fill. As S grows,
utilisation rises (10% at S=2,048; 21% at S=4,096; 38% at S=8,192),
reflecting better occupancy and warp-level parallelism, so each doubling of
S yields progressively less than a 4× latency increase.

Sustained throughput peaks at **16,250 GFLOPS** (16.3 TFLOPS) for S=8,192.
This is 109% of the FP32 scalar peak (14.9 TFLOPS), confirming that tensor
cores are active for this configuration; it is nevertheless only 14.8% of
the theoretical FP16 tensor core peak (110 TFLOPS). The throughput drops
slightly to 14,973 GFLOPS at S=16,384, consistent with increased memory
pressure as the L1 read traffic grows to 938 MB—a regime where cache
thrashing limits effective reuse.

The standard deviation of latency falls from 0.051 ms at S=1,024 (26% of
mean) to 0.007 ms at S=4,096 (1.4%) and remains below 4% for S≥4,096,
indicating stable, repeatable measurements for the practically important
mid-to-large configurations.

### SM and DRAM utilisation

SM utilisation as reported by Nsight Compute rises monotonically from 5% to
38% over the tested range and then falls back to 33% at S=16,384, tracking
the throughput curve. DRAM utilisation (as seen by the memory controller)
stays below 0.6% across all configs. This low DRAM utilisation is not
indicative of poor bandwidth use; rather, it reflects that the
FlashAttention-style kernel issues DRAM traffic in short, high-bandwidth
bursts between which the controller is idle, so the time-averaged
utilisation appears low even though the kernel is bandwidth-limited.

---

## 3.3 Roofline Analysis

![Roofline chart for TITAN V single-head attention](figures/roofline.png)

**Figure 1:** Roofline chart for the TITAN V. The solid blue line is the
FP16 tensor core roof (110 TFLOPS, ridge at 169 FLOP/byte); the dashed
green line is the FP32 scalar roof (14.9 TFLOPS, ridge at 22.8 FLOP/byte).
Open circles show the theoretical arithmetic intensity (inputs + output
only); filled squares show the achieved AI from Nsight Compute. All
measured operating points fall below the tensor core ridge, placing the
workload in the memory-bound regime.

### Theoretical vs. achieved arithmetic intensity

The theoretical AI lower bound assumes that only the four FP16 arrays Q, K,
V, and O (each of size S×D) are transferred between DRAM and the chip:

```
AI_theo = 4·S²·D / (4·S·D·2)  =  S / 2  [FLOP/byte]
```

This gives AI_theo = 512–8,192 FLOP/byte for the tested range—well above
the tensor core ridge—suggesting that a perfectly cache-oblivious
implementation would be compute-bound.

The achieved AI measured by Nsight Compute ranges from **57 to 92 FLOP/byte**
across all configurations (Table 2), roughly 10–90× below the theoretical
bound. The gap reflects traffic that the theoretical bound ignores: (i) the
tiled accumulation of partial softmax numerators and denominators requires
intermediate loads/stores across the tile boundary; (ii) the Q, K, V blocks
that do not fit in the L1 scratchpad must be reloaded on each outer loop
iteration; (iii) Nsight Compute's kernel replay can inflate measured bytes
by re-issuing cache-resident loads that would normally hit in L2 on a single
real pass.

All achieved AI values (57–92 FLOP/byte) lie between the FP32 ridge
(22.8 FLOP/byte) and the FP16 tensor core ridge (169 FLOP/byte). Because
the kernel actively uses tensor cores—measured throughput at S=8,192
reaches 16.3 TFLOPS, exceeding the 14.9 TFLOPS FP32 scalar peak—the
binding ceiling is the tensor core roof. The workload is therefore
**memory-bound with respect to the tensor core ceiling** for all tested
configurations.

### Binding constraint

The performance gap between the measured operating points and the tensor
core roof is 6–80×, depending on S. At S=8,192, where throughput peaks at
16,250 GFLOPS and AI is 92 FLOP/byte, the memory ceiling predicts
92 × 652.8 GB/s = **60 TFLOPS**; the achieved 16.3 TFLOPS is 3.7× below
even that ceiling. This gap reflects imperfect memory access efficiency:
L2-hit traffic is counted at full bandwidth, but partial cache misses,
bank conflicts, and warp stalls reduce effective throughput. Nonetheless,
the roofline analysis confirms that no amount of pure compute optimisation
(e.g., selecting a faster matmul tile size) would substantially improve
performance; reducing memory traffic—or increasing arithmetic reuse per
byte fetched—is the necessary lever.

---

## 3.4 Energy Efficiency

### Power scaling

Average GPU power during sustained attention runs scales from 46.6 W at
S=1,024 to a peak of 112 W at S=8,192, then drops slightly to 107 W at
S=16,384 (Table 2). The idle baseline is approximately 36 W (measured before
any kernel activity). The active increment above idle therefore ranges from
~11 W (S=1,024) to ~76 W (S=8,192), consistent with the SM utilisation
trend: a 5% utilised GPU draws little incremental power, while a 38%
utilised GPU draws ~76 W of dynamic power on top of the static component.

The slight power decrease from S=8,192 to S=16,384 (112 W → 107 W) despite
higher total FLOPs is explained by the lower SM utilisation at S=16,384
(33% vs. 38%), as threads spend more time stalled on memory rather than
executing compute instructions.

### Energy per inference

Energy per inference E = P_avg · t_lat grows with S because both power and
latency increase. It ranges from **9.1 mJ** at S=1,024 to **491 mJ** at
S=16,384—a 54× increase over a 16× increase in S. The super-linear energy
scaling reflects the combination of longer latency (23.6×) and higher power
(2.3×). For workloads where latency is acceptable but energy is constrained
(e.g., battery-powered edge devices), shorter sequences are substantially
more efficient on a per-inference basis.

### GFLOPS/W efficiency

**Figure 2** (Table 2, rightmost column) shows GFLOPS/W as a function of S.
Efficiency improves monotonically from **29.6 GFLOPS/W** at S=1,024 to a
peak of **145 GFLOPS/W** at S=8,192, then holds at **140 GFLOPS/W** at
S=16,384. The 4.9× improvement from smallest to largest S is driven by two
compounding factors: throughput grows 11.8× while power grows only 2.4×,
so the ratio improves substantially.

The plateau at S≥8,192 indicates that the workload has entered a regime
where both the GPU's memory subsystem and its power delivery are
simultaneously saturated; further increases in S raise latency and energy
proportionally without improving efficiency. The peak value of 145 GFLOPS/W
represents roughly 33% of the theoretical maximum (~440 GFLOPS/W at 110
TFLOPS / 250 W TDP), a reasonable figure for a memory-bound kernel on a
GPU that was not designed specifically for attention.

### Comparison context for the RTL accelerator

These figures establish the GPU efficiency baseline that the custom systolic
array (Members C and D) must be compared against. For the primary S=4,096
config:

- GPU throughput: **8,323 GFLOPS**
- GPU energy per inference: **41.7 mJ**
- GPU efficiency: **103 GFLOPS/W**

A systolic array occupying a fraction of the TITAN V die area and targeting
this single operation can reasonably be expected to achieve higher
GFLOPS/W, since it eliminates the large static power of a general-purpose
GPU (DRAM controllers, video encoders, raster pipelines, etc.) that
contribute to the ~36 W idle baseline. Whether it achieves comparable raw
throughput on the larger sequence lengths (S=8,192, 16,384) will depend on
the on-chip buffer capacity relative to the O(S²) attention matrix.
