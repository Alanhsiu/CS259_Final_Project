# System-Level Analytical Model and GPU Comparison

## Methodology

Member B was responsible for the system-level analytical model and the cross-layer comparison between the custom systolic-array accelerator and the GPU baseline. Following the project scope clarification, we removed gem5 from the main evaluation path and focused on an analytical accelerator model driven by RTL-derived area, power, and timing parameters.

The modeled workload is the matmul portion of single-head transformer attention with sequence length `S = 4096` and head dimension `D = 64`. The accelerator model includes the two dominant matrix multiplications:

```text
QK^T:  [S x D] x [D x S] -> [S x S]
AttnV: [S x S] x [S x D] -> [S x D]
```

Softmax is excluded from the accelerator timing model. For operation counting, one multiply-accumulate is counted as two operations, so each matmul has:

```text
2 * S * S * D operations
```

The full matmul portion of attention therefore has:

```text
4 * S * S * D = 4.29 GOPS for S=4096, D=64
```

The analytical model tiles each matrix multiplication over an `R x C` systolic array. For an `M x K` by `K x N` multiplication:

```text
tiles_M = ceil(M / R)
tiles_N = ceil(N / C)
cycles_per_output_tile = K + fill/drain overhead
total_cycles = tiles_M * tiles_N * cycles_per_output_tile
latency = total_cycles / frequency
```

When measured RTL timing is available, the model can replace the estimated fill/drain overhead with RTL-derived cycle counts. Area and power are taken from the RTL synthesis reports and converted into `mm^2` and watts before computing area efficiency and energy efficiency.

## RTL-Informed Accelerator Results

The proposal targets an `8x8` systolic array, so we use the `8x8` RTL-informed result as the main accelerator configuration. We also evaluate `4x4` and `16x16` arrays as a sensitivity study. The RTL implementation uses INT8 inputs and approximately 19-20 bit accumulation. The reported frequency is based on a `10 ns` cycle time, corresponding to `100 MHz`.

Table 1 summarizes the full-workload projection using measured RTL PPA values for each array size.

**Table 1. RTL-informed accelerator projection for attention matmuls (`S=4096`, `D=64`).**

| Array | Total Cycles | Latency (ms) | Throughput (GOPS) | GOPS/W | GOPS/mm^2 |
|---|---:|---:|---:|---:|---:|
| `4x4` | 140,607,488 | 1406.075 | 3.055 | 724.252 | 43.637 |
| `8x8` | 37,281,792 | 372.818 | 11.520 | 923.344 | 5.051 |
| `16x16` | 10,385,408 | 103.854 | 41.356 | 1033.703 | 4.537 |

Increasing the array size significantly reduces latency because more output elements are computed in parallel. The `16x16` design is about `3.6x` faster than the `8x8` design and about `13.5x` faster than the `4x4` design. Energy efficiency also improves with larger arrays because the additional compute resources increase utilization faster than power increases.

However, area efficiency does not improve monotonically. The measured `4x4` design has the highest GOPS/mm^2, while `8x8` and `16x16` are lower. This is because the synthesized total area grows quickly once interconnect and control overhead are included. This suggests that scaling the array improves latency but introduces nontrivial area overhead.

## Memory-Stall Sensitivity

We also ran a sensitivity study on the `8x8` RTL-informed model to estimate the effect of memory or buffer stalls. This sweep holds frequency, area, and power fixed to the measured `8x8` values and increases the cycle count by a stall fraction.

**Table 2. Memory-stall sensitivity for the `8x8` analytical model.**

| Memory Stall | Total Cycles | Latency (ms) | Throughput (GOPS) |
|---:|---:|---:|---:|
| 0% | 37,281,792 | 372.818 | 11.520 |
| 10% | 41,062,400 | 410.624 | 10.460 |
| 20% | 44,843,008 | 448.430 | 9.578 |

As expected, stalls directly increase latency and reduce throughput. A 20% stall fraction reduces throughput from `11.520 GOPS` to `9.578 GOPS`. This reinforces the importance of keeping the systolic array fed with input and weight tiles from local buffers.

## GPU Baseline

Member A provided the GPU baseline using PyTorch scaled dot-product attention on an NVIDIA TITAN V. The baseline uses FP16 precision and PyTorch's optimized SDPA path. For the main `S=4096`, `D=64` configuration, the GPU achieves:

**Table 3. GPU baseline for `S=4096`, `D=64`.**

| Metric | Value |
|---|---:|
| GPU | NVIDIA TITAN V |
| Precision | FP16 |
| Latency | 0.516 ms |
| Throughput | 8322 GFLOPS |
| Power | 80.7 W |
| Energy | 41.7 mJ |
| Efficiency | 103.1 GFLOPS/W |

## Accelerator vs. GPU Comparison

Table 4 compares the main `8x8` RTL-informed systolic array result against the TITAN V GPU baseline.

**Table 4. Accelerator vs. GPU comparison for `S=4096`, `D=64`.**

| System | Scope | Precision | Latency (ms) | Throughput | Power | Efficiency |
|---|---|---|---:|---:|---:|---:|
| `8x8` systolic array | Matmul only | INT8 input, ~20-bit accumulation | 372.818 | 11.520 GOPS | 12.477 mW | 923.344 GOPS/W |
| NVIDIA TITAN V SDPA | Full SDPA | FP16 | 0.516 | 8322 GFLOPS | 80.7 W | 103.1 GFLOPS/W |

The GPU is orders of magnitude faster in raw latency and throughput because it has far more parallel compute resources and uses a highly optimized PyTorch SDPA kernel. The `8x8` systolic array does not compete with the GPU on absolute latency.

The custom accelerator, however, operates at much lower power. In the RTL-informed projection, the `8x8` array achieves higher operation-level energy efficiency than the GPU baseline. This result should be interpreted carefully: the accelerator model targets INT8 matmul only, while the GPU baseline uses FP16 SDPA. Therefore, the comparison is a cross-layer reference point rather than an iso-precision, identical-kernel comparison.

## Discussion

The main takeaway is that a small systolic-array accelerator can be attractive for energy efficiency but is not competitive with a full GPU in raw performance for large attention workloads. The `8x8` array is the proposal-matching design point and provides a useful middle ground: it is much faster than `4x4`, lower-area than `16x16`, and has measured RTL PPA support.

The sensitivity results also show that array scaling alone is not enough. Area grows quickly due to interconnect and control overhead, and memory stalls can noticeably reduce throughput. A complete accelerator would therefore need careful buffer sizing and dataflow scheduling to keep the array busy.

Overall, the analytical model connects the RTL design to the end-to-end attention workload. It allows small synthesized systolic-array blocks to be projected onto full `QK^T` and `AttnV` computations, and it provides a fair way to discuss latency, energy efficiency, and area efficiency relative to a strong GPU baseline.

## Statement of Work

Member B developed the analytical performance model for the systolic-array accelerator, including operation counting, tiling, cycle estimation, latency, throughput, energy-efficiency, and area-efficiency calculations. Member B integrated measured RTL PPA results from the RTL team for `4x4`, `8x8`, and `16x16` designs, generated array-size and memory-stall sensitivity tables, and compared the main `8x8` accelerator configuration against Member A's PyTorch/TITAN V GPU baseline.
