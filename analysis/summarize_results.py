#!/usr/bin/env python3
"""
Summarize analytical model CSV outputs into compact report tables.
"""

from __future__ import annotations

import csv
from pathlib import Path


RESULTS_DIR = Path("results")


def read_total_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as csvfile:
        rows = list(csv.DictReader(csvfile))
    for row in rows:
        if row["name"] == "Total matmul":
            return row
    raise ValueError(f"No Total matmul row found in {path}")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_array_sweep() -> list[dict[str, str]]:
    rows = []
    paths = sorted(
        RESULTS_DIR.glob("array_*x*.csv"),
        key=lambda p: int(p.stem.replace("array_", "").split("x")[0]),
    )
    for path in paths:
        total = read_total_row(path)
        label = path.stem.replace("array_", "")
        rows.append(
            {
                "array": label,
                "total_cycles": total["total_cycles"],
                "latency_ms": f"{float(total['latency_us']) / 1000.0:.3f}",
                "tops": total["tops"],
                "tops_per_watt": total["tops_per_watt"],
                "gops_per_mm2": total["gops_per_mm2"],
            }
        )
    return rows


def summarize_stall_sweep() -> list[dict[str, str]]:
    rows = []
    paths = sorted(
        RESULTS_DIR.glob("stall_*.csv"),
        key=lambda p: int(p.stem.replace("stall_", "")),
    )
    for path in paths:
        total = read_total_row(path)
        label = path.stem.replace("stall_", "")
        stall_percent = int(label)
        rows.append(
            {
                "memory_stall_percent": str(stall_percent),
                "total_cycles": total["total_cycles"],
                "latency_ms": f"{float(total['latency_us']) / 1000.0:.3f}",
                "tops": total["tops"],
                "tops_per_watt": total["tops_per_watt"],
                "gops_per_mm2": total["gops_per_mm2"],
            }
        )
    return rows


def summarize_rtl_ppa() -> list[dict[str, str]]:
    rows = []
    paths = sorted(
        RESULTS_DIR.glob("rtl_*x*_measured_ppa.csv"),
        key=lambda p: int(p.stem.split("_")[1].replace("x", "")),
    )
    for path in paths:
        total = read_total_row(path)
        label = path.stem.split("_")[1]
        rows.append(
            {
                "array": label,
                "total_cycles": total["total_cycles"],
                "latency_ms": f"{float(total['latency_us']) / 1000.0:.3f}",
                "tops": total["tops"],
                "tops_per_watt": total["tops_per_watt"],
                "gops_per_mm2": total["gops_per_mm2"],
            }
        )
    return rows


def print_markdown_table(title: str, rows: list[dict[str, str]], columns: list[str]) -> None:
    print(title)
    print()
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(row[column] for column in columns) + " |")
    print()


def main() -> None:
    array_rows = summarize_array_sweep()
    stall_rows = summarize_stall_sweep()
    rtl_rows = summarize_rtl_ppa()

    write_csv(
        RESULTS_DIR / "summary_array_sweep.csv",
        array_rows,
        ["array", "total_cycles", "latency_ms", "tops", "tops_per_watt", "gops_per_mm2"],
    )
    write_csv(
        RESULTS_DIR / "summary_stall_sweep.csv",
        stall_rows,
        ["memory_stall_percent", "total_cycles", "latency_ms", "tops", "tops_per_watt", "gops_per_mm2"],
    )
    write_csv(
        RESULTS_DIR / "summary_rtl_ppa.csv",
        rtl_rows,
        ["array", "total_cycles", "latency_ms", "tops", "tops_per_watt", "gops_per_mm2"],
    )

    print_markdown_table(
        "Array Size Sweep",
        array_rows,
        ["array", "total_cycles", "latency_ms", "tops", "tops_per_watt", "gops_per_mm2"],
    )
    print_markdown_table(
        "Memory Stall Sweep",
        stall_rows,
        ["memory_stall_percent", "total_cycles", "latency_ms", "tops", "tops_per_watt", "gops_per_mm2"],
    )
    print_markdown_table(
        "RTL Measured PPA",
        rtl_rows,
        ["array", "total_cycles", "latency_ms", "tops", "tops_per_watt", "gops_per_mm2"],
    )

    print("Wrote results/summary_array_sweep.csv")
    print("Wrote results/summary_stall_sweep.csv")
    print("Wrote results/summary_rtl_ppa.csv")


if __name__ == "__main__":
    main()
