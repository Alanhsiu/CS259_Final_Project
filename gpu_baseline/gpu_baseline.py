"""GPU baseline: sweep S, measure SDPA latency and GFLOPS, write gpu_results.csv."""
import csv
import time

import torch
import torch.nn.functional as F

SEQ_LENS = [1024, 2048, 4096, 8192, 16384]
D = 64
WARMUP = 100
ITERS = 100
OUT_CSV = "gpu_results.csv"


def measure(S: int) -> dict:
    Q = torch.randn(1, 1, S, D, device="cuda", dtype=torch.float16)
    K = torch.randn(1, 1, S, D, device="cuda", dtype=torch.float16)
    V = torch.randn(1, 1, S, D, device="cuda", dtype=torch.float16)

    # warmup
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    # timed runs
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)  # ms

    mean_ms = sum(times) / len(times)
    std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
    flops = 4 * S * S * D
    gflops = flops / (mean_ms * 1e-3) / 1e9
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

    return {
        "S": S,
        "latency_mean_ms": round(mean_ms, 4),
        "latency_std_ms": round(std_ms, 4),
        "gflops": round(gflops, 2),
        "peak_mem_mb": round(peak_mem_mb, 2),
    }


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    rows = []
    for S in SEQ_LENS:
        torch.cuda.reset_peak_memory_stats()
        r = measure(S)
        rows.append(r)
        print(f"  S={S:6d}  lat={r['latency_mean_ms']:.2f}±{r['latency_std_ms']:.2f} ms  "
              f"{r['gflops']:.1f} GFLOPS  peak={r['peak_mem_mb']:.0f} MB")

    fieldnames = ["S", "latency_mean_ms", "latency_std_ms", "gflops", "peak_mem_mb"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {OUT_CSV}")


if __name__ == "__main__":
    main()
