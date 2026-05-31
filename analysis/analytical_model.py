#!/usr/bin/env python3
"""
Analytical performance model for the CS259 final project.

This models only the matmul portion of attention:
  1. Q x K^T:  [S x D] x [D x S] -> [S x S]
  2. Attn x V: [S x S] x [S x D] -> [S x D]

Softmax is intentionally out of scope for the accelerator model.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AcceleratorConfig:
    array_rows: int = 8
    array_cols: int = 8
    frequency_mhz: float = 200.0
    power_mw: float = 50.0
    area_mm2: float = 1.0
    drain_cycles: int | None = None
    cycles_per_output_tile: int | None = None
    memory_stall_fraction: float = 0.0

    @property
    def effective_drain_cycles(self) -> int:
        if self.drain_cycles is not None:
            return self.drain_cycles
        return self.array_rows + self.array_cols - 2


@dataclass(frozen=True)
class MatmulShape:
    name: str
    m: int
    k: int
    n: int

    @property
    def macs(self) -> int:
        return self.m * self.n * self.k

    @property
    def ops(self) -> int:
        return 2 * self.macs


@dataclass(frozen=True)
class MatmulEstimate:
    name: str
    m: int
    k: int
    n: int
    tiles_m: int
    tiles_n: int
    cycles_per_output_tile: int
    total_cycles: int
    latency_us: float
    ops: int
    tops: float
    tops_per_watt: float
    gops_per_mm2: float


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def attention_shapes(sequence_length: int, head_dim: int) -> list[MatmulShape]:
    return [
        MatmulShape("QK^T", sequence_length, head_dim, sequence_length),
        MatmulShape("AttnV", sequence_length, sequence_length, head_dim),
    ]


def estimate_matmul(shape: MatmulShape, config: AcceleratorConfig) -> MatmulEstimate:
    tiles_m = ceil_div(shape.m, config.array_rows)
    tiles_n = ceil_div(shape.n, config.array_cols)

    # One output tile streams across the full K dimension. The drain term captures
    # systolic fill/drain overhead. If RTL provides a measured cycles-per-output-
    # tile value for the same tile semantics, use it directly.
    if config.cycles_per_output_tile is None:
        base_cycles = shape.k + config.effective_drain_cycles
    else:
        base_cycles = config.cycles_per_output_tile
    stalled_cycles = math.ceil(base_cycles * (1.0 + config.memory_stall_fraction))
    total_cycles = tiles_m * tiles_n * stalled_cycles

    latency_us = total_cycles / config.frequency_mhz
    seconds = latency_us * 1e-6
    tops = shape.ops / seconds / 1e12 if seconds > 0 else 0.0
    watts = config.power_mw / 1000.0
    tops_per_watt = tops / watts if watts > 0 else 0.0
    gops_per_mm2 = (tops * 1000.0) / config.area_mm2 if config.area_mm2 > 0 else 0.0

    return MatmulEstimate(
        name=shape.name,
        m=shape.m,
        k=shape.k,
        n=shape.n,
        tiles_m=tiles_m,
        tiles_n=tiles_n,
        cycles_per_output_tile=stalled_cycles,
        total_cycles=total_cycles,
        latency_us=latency_us,
        ops=shape.ops,
        tops=tops,
        tops_per_watt=tops_per_watt,
        gops_per_mm2=gops_per_mm2,
    )


def estimate_all(shapes: Iterable[MatmulShape], config: AcceleratorConfig) -> list[MatmulEstimate]:
    return [estimate_matmul(shape, config) for shape in shapes]


def total_row(estimates: list[MatmulEstimate], config: AcceleratorConfig) -> MatmulEstimate:
    total_cycles = sum(item.total_cycles for item in estimates)
    total_ops = sum(item.ops for item in estimates)
    latency_us = total_cycles / config.frequency_mhz
    seconds = latency_us * 1e-6
    tops = total_ops / seconds / 1e12 if seconds > 0 else 0.0
    watts = config.power_mw / 1000.0
    tops_per_watt = tops / watts if watts > 0 else 0.0
    gops_per_mm2 = (tops * 1000.0) / config.area_mm2 if config.area_mm2 > 0 else 0.0

    return MatmulEstimate(
        name="Total matmul",
        m=0,
        k=0,
        n=0,
        tiles_m=0,
        tiles_n=0,
        cycles_per_output_tile=0,
        total_cycles=total_cycles,
        latency_us=latency_us,
        ops=total_ops,
        tops=tops,
        tops_per_watt=tops_per_watt,
        gops_per_mm2=gops_per_mm2,
    )


def write_csv(path: Path, rows: list[MatmulEstimate]) -> None:
    fieldnames = [
        "name",
        "m",
        "k",
        "n",
        "tiles_m",
        "tiles_n",
        "cycles_per_output_tile",
        "total_cycles",
        "latency_us",
        "ops",
        "tops",
        "tops_per_watt",
        "gops_per_mm2",
    ]
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row.name,
                    "m": row.m,
                    "k": row.k,
                    "n": row.n,
                    "tiles_m": row.tiles_m,
                    "tiles_n": row.tiles_n,
                    "cycles_per_output_tile": row.cycles_per_output_tile,
                    "total_cycles": row.total_cycles,
                    "latency_us": f"{row.latency_us:.6f}",
                    "ops": row.ops,
                    "tops": f"{row.tops:.6f}",
                    "tops_per_watt": f"{row.tops_per_watt:.6f}",
                    "gops_per_mm2": f"{row.gops_per_mm2:.6f}",
                }
            )


def print_table(rows: list[MatmulEstimate], config: AcceleratorConfig) -> None:
    print("Accelerator config")
    print(f"  array: {config.array_rows}x{config.array_cols} INT8 MACs")
    print(f"  frequency: {config.frequency_mhz:.3f} MHz")
    print(f"  power: {config.power_mw:.3f} mW")
    print(f"  area: {config.area_mm2:.3f} mm^2")
    print(f"  memory stall fraction: {config.memory_stall_fraction:.3f}")
    print()
    print(
        f"{'Matmul':<12} {'Shape':<22} {'Cycles':>14} {'Latency (us)':>14} "
        f"{'TOPS':>10} {'TOPS/W':>10} {'GOPS/mm2':>12}"
    )
    print("-" * 98)
    for row in rows:
        shape = "-" if row.m == 0 else f"{row.m}x{row.k} * {row.k}x{row.n}"
        print(
            f"{row.name:<12} {shape:<22} {row.total_cycles:>14,d} "
            f"{row.latency_us:>14.3f} {row.tops:>10.3f} "
            f"{row.tops_per_watt:>10.3f} {row.gops_per_mm2:>12.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS259 systolic-array analytical model")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--array-rows", type=int, default=8)
    parser.add_argument("--array-cols", type=int, default=8)
    parser.add_argument("--frequency-mhz", type=float, default=200.0)
    parser.add_argument("--power-mw", type=float, default=50.0)
    parser.add_argument("--area-mm2", type=float, default=1.0)
    parser.add_argument("--drain-cycles", type=int, default=None)
    parser.add_argument("--cycles-per-output-tile", type=int, default=None)
    parser.add_argument("--measured-tile-k", type=int, default=None)
    parser.add_argument("--measured-tile-cycles", type=int, default=None)
    parser.add_argument("--memory-stall-fraction", type=float, default=0.0)
    parser.add_argument("--csv", type=Path, default=Path("results/analytical_model.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drain_cycles = args.drain_cycles
    if args.measured_tile_k is not None or args.measured_tile_cycles is not None:
        if args.measured_tile_k is None or args.measured_tile_cycles is None:
            raise SystemExit("--measured-tile-k and --measured-tile-cycles must be used together")
        drain_cycles = args.measured_tile_cycles - args.measured_tile_k
        if drain_cycles < 0:
            raise SystemExit("Measured tile cycles must be greater than or equal to measured tile K")

    config = AcceleratorConfig(
        array_rows=args.array_rows,
        array_cols=args.array_cols,
        frequency_mhz=args.frequency_mhz,
        power_mw=args.power_mw,
        area_mm2=args.area_mm2,
        drain_cycles=drain_cycles,
        cycles_per_output_tile=args.cycles_per_output_tile,
        memory_stall_fraction=args.memory_stall_fraction,
    )
    rows = estimate_all(attention_shapes(args.sequence_length, args.head_dim), config)
    rows.append(total_row(rows, config))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.csv, rows)
    print_table(rows, config)
    print()
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
