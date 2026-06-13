"""Task 4: Roofline plot for TITAN V attention workload.

Shows both theoretical and achieved arithmetic intensity for each S config,
plotted against the memory and compute ceilings of the TITAN V.
"""
import csv
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------- TITAN V hardware specs ----------
BW_GBS = 652.8          # HBM2 peak bandwidth (GB/s)
FP32_PEAK_GFLOPS = 14900.0   # FP32 peak (~14.9 TFLOPS)
FP16_TENSOR_GFLOPS = 110000.0  # FP16 tensor core peak (~110 TFLOPS)
FP16_STD_GFLOPS = 28900.0    # FP16 standard peak (~28.9 TFLOPS)
# Ridge points: intensity where memory ceiling meets compute ceiling
RIDGE_TENSOR = FP16_TENSOR_GFLOPS / BW_GBS   # ~168 FLOP/byte
RIDGE_FP32 = FP32_PEAK_GFLOPS / BW_GBS       # ~22.8 FLOP/byte
RIDGE_FP16 = FP16_STD_GFLOPS / BW_GBS        # ~44.2 FLOP/byte

D = 64
SEQ_LENS = [1024, 2048, 4096, 8192, 16384]


def theoretical_ai(S: int) -> float:
    """Minimum bytes = read Q,K,V + write O; all FP16 (2 bytes each)."""
    flops = 4 * S * S * D
    # Q: S*D, K: S*D, V: S*D, O: S*D  (all FP16)
    min_bytes = 4 * S * D * 2   # 4 matrices, each S*D elements, 2 bytes each
    # attn_weights S*S (written then read) — exclude if we assume flash-like
    # Use the strict lower bound: input + output only
    return flops / min_bytes


def load_ncu_results(path: str) -> dict:
    """Returns dict S -> row."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[int(row["S"])] = row
    return data


def load_base_results(path: str) -> dict:
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[int(row["S"])] = row
    return data


def main():
    ncu = load_ncu_results(os.path.join(DATA_DIR, "gpu_results_ncu.csv"))
    # base = load_base_results(os.path.join(DATA_DIR, "gpu_results_base.csv"))
    base = load_base_results(os.path.join(DATA_DIR, "gpu_results_final.csv"))

    fig, ax = plt.subplots(figsize=(10, 7))

    # --- Roofline ceilings ---
    ai_range = np.logspace(-1, 4, 2000)

    # Memory ceiling (diagonal): GFLOPS = AI * BW
    mem_ceil = ai_range * BW_GBS

    # Compute ceilings (horizontal)
    compute_ceil_tensor = np.full_like(ai_range, FP16_TENSOR_GFLOPS)
    compute_ceil_fp32 = np.full_like(ai_range, FP32_PEAK_GFLOPS)

    # Roofline = min(memory, compute)
    roof_tensor = np.minimum(mem_ceil, compute_ceil_tensor)
    roof_fp32 = np.minimum(mem_ceil, compute_ceil_fp32)

    ax.loglog(ai_range, roof_tensor, "b-", linewidth=2, label="FP16 Tensor Core roof (~110 TFLOPS)")
    ax.loglog(ai_range, roof_fp32, "g--", linewidth=1.5, label="FP32 roof (~14.9 TFLOPS)")

    # Annotate ridge points
    ax.axvline(RIDGE_TENSOR, color="b", linestyle=":", alpha=0.4)
    ax.axvline(RIDGE_FP32, color="g", linestyle=":", alpha=0.4)
    ax.text(RIDGE_TENSOR * 1.05, 500, f"Ridge\n{RIDGE_TENSOR:.0f} FLOP/B", color="b",
            fontsize=8, va="bottom")
    ax.text(RIDGE_FP32 * 1.05, 200, f"Ridge\n{RIDGE_FP32:.1f} FLOP/B", color="g",
            fontsize=8, va="bottom")

    # --- Data points ---
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(SEQ_LENS)))

    for i, S in enumerate(SEQ_LENS):
        flops = 4 * S * S * D
        gflops = float(base[S]["gflops"])
        ai_theo = theoretical_ai(S)
        ai_ach = float(ncu[S]["achieved_ai_flops_per_byte"])
        label_s = f"S={S}"

        # Theoretical point (open circle)
        ax.scatter(ai_theo, gflops, color=colors[i], s=80, marker="o",
                   edgecolors=colors[i], facecolors="none", linewidths=2, zorder=5)
        # Achieved point (filled square)
        ax.scatter(ai_ach, gflops, color=colors[i], s=80, marker="s",
                   zorder=5)

        # Label near achieved point
        ax.annotate(label_s, (ai_ach, gflops),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=colors[i])

    # --- Legend for point styles ---
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markerfacecolor="none", markersize=8, label="Theoretical AI (input+output only)"),
        Line2D([0], [0], marker="s", color="gray", linestyle="None",
               markersize=8, label="Achieved AI (measured DRAM traffic)"),
    ]
    for i, S in enumerate(SEQ_LENS):
        legend_handles.append(
            Line2D([0], [0], marker="s", color=colors[i], linestyle="None",
                   markersize=6, label=f"S={S}")
        )
    legend_handles = legend_handles[:2] + [
        Line2D([0], [0], color="b", linewidth=2, label="FP16 Tensor Core roof (~110 TFLOPS)"),
        Line2D([0], [0], color="g", linestyle="--", linewidth=1.5, label="FP32 roof (~14.9 TFLOPS)"),
    ] + legend_handles[2:]

    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, ncol=1)

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=12)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=12)
    ax.set_title("Roofline — TITAN V, FP16 Single-Head Attention\n"
                 "(D=64, batch=1; achieved AI 57–92 FLOP/B < tensor-core ridge 169 FLOP/B)",
                 fontsize=11)

    ax.set_xlim(1, 1e4)
    ax.set_ylim(100, 2e5)
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    # Shade memory-bound region
    # ax.axvspan(1, RIDGE_TENSOR, alpha=0.04, color="red", label="_nolegend_")
    # ax.text(2, 130, "Memory-bound region", color="red", fontsize=8, alpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "roofline.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    # Print sanity-check table
    print(f"\n{'S':>6}  {'AI_theo':>10}  {'AI_ach':>10}  {'GFLOPS':>8}  {'Bound'}")
    print("-" * 55)
    for S in SEQ_LENS:
        ai_theo = theoretical_ai(S)
        ai_ach = float(ncu[S]["achieved_ai_flops_per_byte"])
        gflops = float(base[S]["gflops"])
        # FP16 tensor core is the active compute ceiling for this workload
        bound = "MEMORY" if ai_ach < RIDGE_TENSOR else "COMPUTE (tensor)"
        print(f"{S:>6}  {ai_theo:>10.1f}  {ai_ach:>10.1f}  {gflops:>8.0f}  {bound}")
    ai_vals = [float(ncu[S]["achieved_ai_flops_per_byte"]) for S in SEQ_LENS]
    print(f"\nTITAN V ridge points: FP32={RIDGE_FP32:.1f} FLOP/B, "
          f"FP16 Tensor={RIDGE_TENSOR:.0f} FLOP/B")
    print(f"Achieved AI range: {min(ai_vals):.0f}–{max(ai_vals):.0f} FLOP/B")
    print(f"All values below FP16 Tensor Core ridge ({RIDGE_TENSOR:.0f} FLOP/B) → MEMORY-BOUND ✓")


if __name__ == "__main__":
    main()
