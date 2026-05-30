<!--
Member A's contribution to the final report (GPU baseline + test vectors).
Merge guide:
  - "GPU Baseline — Methodology"  -> into the paper's Methodology section
  - "GPU Baseline — Evaluation"   -> into the paper's Evaluation section
  - "Statement of Work (Member A)"-> into the global Statement of Work
  - "References (this section)"   -> fold into the global References list
Figure path assumes the repo layout: gpu_baseline/figures/roofline.png
-->

# GPU Baseline — Methodology

*(Merge into the paper's Methodology section.)*

## Platform

All GPU baseline measurements were collected on a single NVIDIA TITAN V (Volta, GV100, `sm_70`) with 12 GB of HBM2. The hardware parameters used throughout the analysis are summarized in Table 1. The software stack was PyTorch 2.6.0 with CUDA 12.4; attention was executed through `torch.nn.functional.scaled_dot_product_attention` (SDPA), which on Volta dispatches to the memory-efficient FlashAttention-style kernel (`fmha_cutlassF_f16_aligned_*_sm70`) [1, 2]. Kernel-level memory and utilization counters were collected with NVIDIA Nsight Compute (`ncu`) [3]. The GPU was shared with other users during measurement; the implications are noted where relevant.

**Table 1. NVIDIA TITAN V parameters used for the baseline and roofline model.**

| Parameter | Value |
|---|---|
| Architecture | Volta (GV100, `sm_70`) |
| HBM2 peak bandwidth | 652.8 GB/s |
| FP32 peak | 14.9 TFLOPS |
| FP16 tensor-core peak | 110 TFLOPS |
| Streaming multiprocessors | 80 |
| L2 cache | 4.5 MB |
| Thermal design power | ~250 W |

## Workload

The baseline workload is single-head scaled dot-product attention in the prefill setting — the same inner computation the custom systolic-array accelerator targets — with batch size 1, one head, head dimension D = 64, and FP16 precision. The sequence length S is swept over {1024, 2048, 4096, 8192, 16384}. We count the two matrix multiplications (QKᵀ and the attention–value product) as 4·S²·D FLOPs and exclude the softmax, following the usual convention for attention FLOP accounting.

## Measurement protocol

**Latency and throughput.** For each S the kernel is run for 100 timed iterations after 100 warm-up iterations, with each iteration bracketed by `torch.cuda.synchronize()` and timed using `time.perf_counter()`. We report the mean and standard deviation, and define sustained throughput as 4·S²·D divided by mean latency. This protocol includes a small per-call CPU launch overhead (tens of microseconds) that is only significant at the smallest S.

**Memory traffic.** Because Nsight Compute replays each kernel to read hardware counters, a single forward pass per S was profiled. We collected the global load/store traffic between the L2 and L1 caches (`l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld.sum` and `l1tex__m_l1tex2xbar_write_bytes_mem_lg_op_st.sum`), SM throughput (`sm__throughput.avg.pct_of_peak_sustained_elapsed`), and memory-controller / DRAM throughput (`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`). We report L1↔L2 traffic rather than `dram__bytes_*` because for the smaller sequence lengths the working set is cache-resident and DRAM writes are effectively zero, which makes a DRAM-based arithmetic intensity ill-defined. The L1↔L2 figure is therefore an upper bound on the data the kernel actually moves through the memory hierarchy. Achieved arithmetic intensity (AI) is computed as FLOPs divided by total L1↔L2 bytes.

**Power and energy.** GPU power was logged at 100 ms intervals with `nvidia-smi --query-gpu=timestamp,power.draw --format=csv -lms 100` while the kernel ran in a tight loop for roughly ten seconds per S. The samples whose timestamps fall inside each config's active window are averaged to obtain mean power. Energy per inference is E = P_avg · t_latency (W · ms = mJ), and efficiency is reported as sustained GFLOPS divided by mean power. Because the GPU was shared, the readings include an idle baseline of ~36 W from other activity; the reported power and energy are therefore conservative (slightly pessimistic) estimates of the kernel's own draw.

---

# GPU Baseline — Evaluation

*(Merge into the paper's Evaluation section.)*

Table 2 reports the full baseline. We organize the discussion around three questions: how performance scales with sequence length, what limits the kernel, and how energy-efficient it is.

**Table 2. GPU baseline for single-head FP16 attention (D = 64, batch = 1) on the TITAN V.**

| S | FLOPs (G) | Latency (ms) | Std (ms) | Throughput (GFLOPS) | L1↔L2 traffic (MB) | Achieved AI (FLOP/B) | SM util (%) | DRAM util (%) | Power (W) | Energy (mJ) | GFLOPS/W |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1,024 | 0.27 | 0.195 | 0.051 | 1,380 | 4.70 | 57.1 | 5.1 | 0.46 | 46.6 | 9.1 | 29.6 |
| 2,048 | 1.07 | 0.318 | 0.031 | 3,376 | 17.70 | 60.7 | 10.4 | 0.50 | 58.3 | 18.5 | 57.9 |
| 4,096 | 4.29 | 0.516 | 0.007 | 8,322 | 70.26 | 61.1 | 21.0 | 0.52 | 80.7 | 41.7 | 103.1 |
| 8,192 | 17.18 | 1.057 | 0.010 | 16,250 | 186.71 | 92.0 | 38.0 | 0.55 | 112.0 | 118.4 | 145.1 |
| 16,384 | 68.72 | 4.590 | 0.164 | 14,973 | 941.93 | 73.0 | 32.9 | 0.53 | 106.9 | 490.7 | 140.0 |

## Performance scaling

Latency grows sub-quadratically: a 16× increase in S (1,024 → 16,384) raises latency only 23.6×, far below the 256× expected if the GPU were saturated at every size. The reason is visible in the utilization column. At S = 1,024 only 5.1% of SM throughput is used and execution is dominated by launch overhead and pipeline fill; as S grows, utilization rises (10% → 21% → 38%) and each doubling of S costs progressively less than the naive 4×. Sustained throughput peaks at 16,250 GFLOPS (16.3 TFLOPS) at S = 8,192 — about 15% of the FP16 tensor-core peak — before falling back to 14,973 GFLOPS at S = 16,384. The drop coincides with the L1↔L2 traffic jumping to 942 MB: at S = 16,384 the Q/K/V operands occupy 6.3 MB and no longer fit in the 4.5 MB L2, so reuse degrades. Measurement variance is small for the practically relevant sizes (latency standard deviation is 1.4% of the mean at S = 4,096 and below 4% for S ≥ 4,096).

## What limits the kernel

We use the roofline in Figure 1 together with the utilization counters to characterize the bottleneck. Three observations matter. First, the kernel is clearly **not compute-bound**: peak throughput reaches only ~15% of the tensor-core roof and SM utilization never exceeds 38%. Second, it is also **not HBM-bandwidth-bound**: measured DRAM utilization stays below 1% for every configuration, and for S ≤ 8,192 the entire Q/K/V working set fits within the L2 cache, so the kernel runs almost entirely out of on-chip memory. Third, the achieved arithmetic intensity measured against L1↔L2 traffic (57–92 FLOP/B) sits below the FP16 tensor-core ridge (169 FLOP/B), confirming that performance is governed by data movement through the cache hierarchy and by occupancy rather than by raw FLOP throughput.

Taken together, the binding constraint for this single-head, D = 64 shape is **low SM occupancy / utilization** — aggravated by launch overhead at small S and by L2 capacity pressure at the largest S — rather than either the compute roof or HBM bandwidth. This is precisely the source of inefficiency a specialized datapath can exploit: a design that keeps a small attention tile resident on chip and feeds a dense systolic array can convert this under-utilized, data-movement-limited regime into near-peak utilization on a fraction of the silicon.

*Note on Figure 1.* The roofline plots the per-S operating points against the TITAN V's HBM (652.8 GB/s) and tensor-core (110 TFLOPS) ceilings. The arithmetic-intensity axis uses L1↔L2 traffic as the denominator, so the diagonal should be read as a memory-traffic reference rather than a strict HBM-bandwidth bound; the sub-1% DRAM utilization in Table 2 is the direct evidence that the configurations are cache-resident.

![Roofline for TITAN V single-head attention. Open markers: theoretical AI (inputs + output only); filled markers: achieved AI from L1↔L2 traffic. All operating points lie below the tensor-core roof.](figures/roofline.png)

**Figure 1.** Roofline chart for the TITAN V single-head attention workload.

## Energy efficiency

Energy efficiency improves strongly with sequence length, from 29.6 GFLOPS/W at S = 1,024 to a peak of 145 GFLOPS/W at S = 8,192, then holds near 140 GFLOPS/W at S = 16,384. The 4.9× improvement is driven by throughput rising 11.8× while mean power rises only 2.4×: a lightly-loaded GPU still pays most of its static power, so efficiency is poor until utilization climbs. The plateau at S ≥ 8,192 reflects the saturation point identified above. Energy per inference rises from 9.1 mJ to 490.7 mJ across the sweep — super-linear in S because both latency and power grow. The peak of 145 GFLOPS/W is roughly one-third of a naive ceiling (~440 GFLOPS/W from 110 TFLOPS / 250 W), a reasonable figure for an under-utilized general-purpose kernel and, again, consistent with the static-power overhead of a large shared GPU.

## Baseline for the accelerator comparison

For the primary S = 4,096 configuration the GPU baseline is **8,322 GFLOPS** of sustained throughput, **41.7 mJ** per inference, and **103 GFLOPS/W**. These are the figures the custom systolic array (Members C/D) is compared against. Because the GPU spends its energy budget largely on static power and under-used general-purpose machinery, a fixed-function datapath occupying a small fraction of the die area can plausibly improve energy per operation even if it does not match the GPU's raw throughput at the largest sequence lengths, where on-chip buffer capacity relative to the O(S²) attention matrix becomes the deciding factor.

---

# Statement of Work (Member A)

*(Merge into the paper's Statement of Work section.)*

**Member A** was responsible for the GPU baseline and the RTL test vectors. This included: implementing the latency/throughput sweep over sequence length using PyTorch SDPA; profiling kernel memory traffic and utilization with Nsight Compute and computing achieved arithmetic intensity; measuring sustained power to derive energy-per-inference and GFLOPS/W; producing the roofline analysis; generating the INT8 matrix-multiply test vectors for the 4×4, 8×8, and 16×16 systolic-array sizes (ten random tiles plus zero/identity/overflow edge cases per size) together with a quantized real attention tile extracted from GPT-2 for the RTL testbench; and writing this GPU Baseline section of the report.

---

# References (this section)

*(Fold into the global References list and renumber as needed.)*

[1] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

[2] PyTorch, "torch.nn.functional.scaled_dot_product_attention," PyTorch 2.6 documentation.

[3] NVIDIA, "Nsight Compute Profiling Guide," NVIDIA Developer Documentation.

[4] NVIDIA, "NVIDIA Tesla V100 GPU Architecture (Volta GV100) Whitepaper," 2017.
