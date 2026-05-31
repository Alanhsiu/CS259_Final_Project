#!/usr/bin/env python3
"""
Create a compact comparison between the RTL-informed accelerator model and the
GPU baseline reported by Member A.
"""

from __future__ import annotations

import csv
from pathlib import Path


RESULTS_DIR = Path("results")


def read_total_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            if row["name"] == "Total matmul":
                return row
    raise ValueError(f"No Total matmul row found in {path}")


def main() -> None:
    rtl = read_total_row(RESULTS_DIR / "rtl_8x8_measured_ppa.csv")

    rows = [
        {
            "system": "8x8 systolic array RTL-informed model",
            "workload": "S=4096, D=64, matmul only",
            "precision": "INT8 input, ~20-bit accumulation",
            "latency_ms": f"{float(rtl['latency_us']) / 1000.0:.3f}",
            "throughput_gflops_or_gops": f"{float(rtl['tops']) * 1000.0:.3f}",
            "power_w": "0.012477",
            "efficiency_gflops_or_gops_per_w": f"{float(rtl['tops_per_watt']) * 1000.0:.3f}",
            "area_mm2": "2.281",
            "notes": "Uses measured 8x8 RTL PPA; softmax excluded",
        },
        {
            "system": "NVIDIA TITAN V PyTorch SDPA",
            "workload": "S=4096, D=64, full SDPA",
            "precision": "FP16",
            "latency_ms": "0.516",
            "throughput_gflops_or_gops": "8322.000",
            "power_w": "80.700",
            "efficiency_gflops_or_gops_per_w": "103.100",
            "area_mm2": "N/A",
            "notes": "Member A GPU baseline; includes optimized PyTorch SDPA path",
        },
    ]

    out = RESULTS_DIR / "accelerator_gpu_comparison.csv"
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("| System | Latency (ms) | Throughput | Power (W) | Efficiency | Area (mm^2) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['system']} | {row['latency_ms']} | "
            f"{row['throughput_gflops_or_gops']} | {row['power_w']} | "
            f"{row['efficiency_gflops_or_gops_per_w']} | {row['area_mm2']} |"
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
