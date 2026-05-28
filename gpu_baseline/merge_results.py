"""Merge gpu_results.csv and gpu_results_ncu.csv into gpu_results_full.csv."""
import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_CSV = os.path.join(SCRIPT_DIR, "gpu_results.csv")
NCU_CSV  = os.path.join(SCRIPT_DIR, "gpu_results_ncu.csv")
OUT_CSV  = os.path.join(SCRIPT_DIR, "gpu_results_full.csv")


def read_csv(path: str) -> dict[int, dict]:
    with open(path, newline="") as f:
        return {int(row["S"]): row for row in csv.DictReader(f)}


def main():
    base = read_csv(BASE_CSV)
    ncu  = read_csv(NCU_CSV)

    base_s = set(base)
    ncu_s  = set(ncu)
    if base_s != ncu_s:
        only_base = base_s - ncu_s
        only_ncu  = ncu_s  - base_s
        if only_base:
            print(f"WARNING: S values only in {BASE_CSV}: {sorted(only_base)}")
        if only_ncu:
            print(f"WARNING: S values only in {NCU_CSV}: {sorted(only_ncu)}")
    common = sorted(base_s & ncu_s)
    print(f"S values aligned: {common}  ({len(common)} rows)")

    fieldnames = [
        "S", "latency_mean_ms", "gflops",
        "l1_read_mb", "l1_write_mb",
        "arithmetic_intensity",
        "sm_util_pct", "dram_util_pct",
    ]

    rows = []
    for S in common:
        b, n = base[S], ncu[S]
        rows.append({
            "S":                   S,
            "latency_mean_ms":     float(b["latency_mean_ms"]),
            "gflops":              float(b["gflops"]),
            "l1_read_mb":          round(int(n["l1_read_bytes"])  / 1e6, 4),
            "l1_write_mb":         round(int(n["l1_write_bytes"]) / 1e6, 4),
            "arithmetic_intensity": float(n["achieved_ai_flops_per_byte"]),
            "sm_util_pct":         float(n["sm_throughput_pct"]),
            "dram_util_pct":       float(n["dram_throughput_pct"]),
        })

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Written: {OUT_CSV}\n")
    header = f"{'S':>6}  {'lat_ms':>8}  {'GFLOPS':>8}  {'l1_r_MB':>8}  {'l1_w_MB':>8}  {'AI':>8}  {'SM%':>6}  {'DRAM%':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['S']:>6}  {r['latency_mean_ms']:>8.3f}  {r['gflops']:>8.1f}  "
              f"{r['l1_read_mb']:>8.3f}  {r['l1_write_mb']:>8.3f}  "
              f"{r['arithmetic_intensity']:>8.1f}  {r['sm_util_pct']:>6.1f}  "
              f"{r['dram_util_pct']:>6.1f}")


if __name__ == "__main__":
    main()
