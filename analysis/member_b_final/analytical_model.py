#!/usr/bin/env python3
"""CS259 final project: Member B analytical RTL model.

This version matches the final RTL comparison report in
rtl/rtl_top_four_4x4/report.md.  The model compares three designs with the
same total number of PEs (64), but different systolic-array partitioning:

    1 x 8x8, 4 x 4x4, and 16 x 2x2.

The goal is not to predict an unrelated RTL design from scratch.  Instead, it
reproduces the RTL report's cycle accounting, then derives latency and PPA
tradeoff metrics from the synthesis numbers.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


FREQ_MHZ = 100.0


@dataclass(frozen=True)
class PartitionDesign:
    name: str
    pe_organization: str
    num_arrays: int
    array_side: int
    total_pes: int
    total_cell_area_um2: int
    total_area_um2: int
    total_power_mw: float


DESIGNS = [
    PartitionDesign(
        name="single_8x8",
        pe_organization="1 x 8x8",
        num_arrays=1,
        array_side=8,
        total_pes=64,
        total_cell_area_um2=962_640,
        total_area_um2=7_920_862,
        total_power_mw=23.0365,
    ),
    PartitionDesign(
        name="four_4x4",
        pe_organization="4 x 4x4",
        num_arrays=4,
        array_side=4,
        total_pes=64,
        total_cell_area_um2=943_418,
        total_area_um2=8_055_035,
        total_power_mw=22.9785,
    ),
    PartitionDesign(
        name="sixteen_2x2",
        pe_organization="16 x 2x2",
        num_arrays=16,
        array_side=2,
        total_pes=64,
        total_cell_area_um2=972_359,
        total_area_um2=8_559_044,
        total_power_mw=23.0523,
    ),
]


@dataclass(frozen=True)
class CycleBreakdown:
    matmul_compute_cycles: int
    qk_cycles: int
    score_capture_cycles: int
    softmax_cycles: int
    attnv_cycles: int
    stream_done_cycles: int
    total_cycles: int


def cycles_for_design(design: PartitionDesign) -> CycleBreakdown:
    """Return FSM cycle accounting for one S=8, D=64 attention tile.

    Values follow the RTL report:
    - matmul compute window: 22 / 14 / 10 cycles
    - QK phase: 8 passes, each with clear + compute + capture
    - softmax: fixed 176 cycles for all designs
    - stream + done: fixed 65 cycles
    """
    matmul_compute_cycles = 2 * design.array_side + 6
    matmul_phase_cycles = 1 + matmul_compute_cycles + 1

    qk_cycles = 8 * matmul_phase_cycles
    score_capture_cycles = 1
    softmax_cycles = 176
    attnv_cycles = matmul_phase_cycles
    stream_done_cycles = 65
    total_cycles = (
        qk_cycles
        + score_capture_cycles
        + softmax_cycles
        + attnv_cycles
        + stream_done_cycles
    )

    return CycleBreakdown(
        matmul_compute_cycles=matmul_compute_cycles,
        qk_cycles=qk_cycles,
        score_capture_cycles=score_capture_cycles,
        softmax_cycles=softmax_cycles,
        attnv_cycles=attnv_cycles,
        stream_done_cycles=stream_done_cycles,
        total_cycles=total_cycles,
    )


def model_row(design: PartitionDesign, baseline: PartitionDesign) -> dict[str, object]:
    cycles = cycles_for_design(design)
    baseline_cycles = cycles_for_design(baseline)

    latency_us = cycles.total_cycles / FREQ_MHZ
    cycles_x_area = cycles.total_cycles * design.total_area_um2
    cycles_x_power = cycles.total_cycles * design.total_power_mw

    best_cycles_x_area = min(
        cycles_for_design(d).total_cycles * d.total_area_um2 for d in DESIGNS
    )
    best_cycles_x_power = min(
        cycles_for_design(d).total_cycles * d.total_power_mw for d in DESIGNS
    )

    return {
        "design": design.name,
        "pe_organization": design.pe_organization,
        "total_pes": design.total_pes,
        "matmul_compute_cycles": cycles.matmul_compute_cycles,
        "qk_cycles": cycles.qk_cycles,
        "score_capture_cycles": cycles.score_capture_cycles,
        "softmax_cycles": cycles.softmax_cycles,
        "attnv_cycles": cycles.attnv_cycles,
        "stream_done_cycles": cycles.stream_done_cycles,
        "total_cycles": cycles.total_cycles,
        "latency_us": latency_us,
        "total_cell_area_um2": design.total_cell_area_um2,
        "total_area_um2": design.total_area_um2,
        "total_power_mw": design.total_power_mw,
        "cycle_change_vs_1x8x8_pct": (
            (cycles.total_cycles / baseline_cycles.total_cycles - 1.0) * 100.0
        ),
        "total_area_change_vs_1x8x8_pct": (
            (design.total_area_um2 / baseline.total_area_um2 - 1.0) * 100.0
        ),
        "power_change_vs_1x8x8_pct": (
            (design.total_power_mw / baseline.total_power_mw - 1.0) * 100.0
        ),
        "cycles_x_area": cycles_x_area,
        "cycles_x_power": cycles_x_power,
        "normalized_cycles_x_area": cycles_x_area / best_cycles_x_area,
        "normalized_cycles_x_power": cycles_x_power / best_cycles_x_power,
    }


def validation_rows() -> list[dict[str, object]]:
    """Compare analytical cycle totals with the RTL report totals."""
    rtl_report_cycles = {
        "single_8x8": 458,
        "four_4x4": 386,
        "sixteen_2x2": 350,
    }
    rows: list[dict[str, object]] = []
    for design in DESIGNS:
        predicted = cycles_for_design(design).total_cycles
        expected = rtl_report_cycles[design.name]
        error = abs(predicted - expected) / expected * 100.0
        rows.append(
            {
                "design": design.name,
                "pe_organization": design.pe_organization,
                "rtl_report_cycles": expected,
                "analytical_model_cycles": predicted,
                "error_percent": error,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    print("\nRTL partition analytical model")
    print("=" * 78)
    print(
        f"{'Design':<14} {'Cycles':>8} {'Latency(us)':>12} "
        f"{'Area change':>12} {'Power change':>13} {'Norm CxA':>10} {'Norm CxP':>10}"
    )
    print("-" * 78)
    for row in rows:
        print(
            f"{row['pe_organization']:<14} "
            f"{int(row['total_cycles']):>8} "
            f"{float(row['latency_us']):>12.2f} "
            f"{float(row['total_area_change_vs_1x8x8_pct']):>11.1f}% "
            f"{float(row['power_change_vs_1x8x8_pct']):>12.2f}% "
            f"{float(row['normalized_cycles_x_area']):>10.3f} "
            f"{float(row['normalized_cycles_x_power']):>10.3f}"
        )

    print("\nValidation against RTL report")
    print("-" * 78)
    for row in validation_rows():
        print(
            f"{row['pe_organization']:<14} "
            f"RTL={int(row['rtl_report_cycles']):>3}  "
            f"model={int(row['analytical_model_cycles']):>3}  "
            f"error={float(row['error_percent']):.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for generated CSV files.",
    )
    args = parser.parse_args()

    baseline = DESIGNS[0]
    rows = [model_row(design, baseline) for design in DESIGNS]

    write_csv(args.out_dir / "rtl_partition_comparison.csv", rows)
    write_csv(args.out_dir / "rtl_partition_validation.csv", validation_rows())
    print_summary(rows)


if __name__ == "__main__":
    main()
