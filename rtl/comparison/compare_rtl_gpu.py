"""
compare_rtl_gpu.py
Compare RTL synthesis results (attention_top) vs GPU baseline (TITAN V).

Sources:
  rtl/rtl_top_syn/compile_area.rpt
  rtl/rtl_top_syn/compile_power.rpt
  rtl/rtl_top_syn/compile_timing.rpt
  gpu_baseline/data/gpu_results_final.csv
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE    = Path(__file__).parent                              # rtl/comparison/
RTL_DIR = BASE.parent / "rtl_top_syn"                       # rtl/rtl_top_syn/
GPU_CSV = BASE.parent.parent / "gpu_baseline" / "data" / "gpu_results_final.csv"
OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)
(BASE / "data").mkdir(exist_ok=True)
CMP_CSV = BASE / "data" / "rtl_gpu_comparison.csv"

# ── 1.  Parse synthesis reports ──────────────────────────────────────────────

def parse_area(path: Path):
    text = path.read_text()
    m  = re.search(r'Total cell area:\s+([\d.]+)', text)
    m2 = re.search(r'Total area:\s+([\d.]+)', text)
    return float(m.group(1)), float(m2.group(1))   # cell_um2, total_um2

def parse_power(path: Path):
    text = path.read_text()
    m  = re.search(r'Total Dynamic Power\s*=\s*([\d.]+)\s*mW', text)
    m2 = re.search(r'Cell Leakage Power\s*=\s*([\d.]+)\s*uW', text)
    dyn_mw  = float(m.group(1))
    leak_uw = float(m2.group(1))
    return dyn_mw, leak_uw, dyn_mw + leak_uw / 1e3   # dynamic_mw, leakage_uw, total_mw

def parse_timing(path: Path):
    text = path.read_text()
    # "clock CLK (rise edge)  10.00  10.00" -- last match carries the period
    matches = re.findall(r'clock CLK \(rise edge\)\s+([\d.]+)\s+([\d.]+)', text)
    period_ns = float(matches[-1][0])
    slack_m   = re.search(r'slack \(MET\)\s+([\d.]+)', text)
    slack_ns  = float(slack_m.group(1)) if slack_m else 0.0
    return period_ns, 1e3 / period_ns, slack_ns     # period_ns, freq_MHz, slack_ns

cell_area_um2, total_area_um2 = parse_area(RTL_DIR / "compile_area.rpt")
dyn_mw, leak_uw, total_mw     = parse_power(RTL_DIR / "compile_power.rpt")
period_ns, freq_mhz, slack_ns = parse_timing(RTL_DIR / "compile_timing.rpt")

print("=== RTL Synthesis Results ===")
print(f"  Cell area          : {cell_area_um2:>12,.0f} um2  ({cell_area_um2/1e6:.4f} mm2)")
print(f"  Total area (w/nets): {total_area_um2:>12,.0f} um2  ({total_area_um2/1e6:.4f} mm2)")
print(f"  Dynamic power      : {dyn_mw:.4f} mW")
print(f"  Leakage power      : {leak_uw:.2f} uW  ({leak_uw/1e3:.4f} mW)")
print(f"  Total power        : {total_mw:.4f} mW")
print(f"  Clock period       : {period_ns:.2f} ns  -> {freq_mhz:.1f} MHz")
print(f"  Timing slack       : {slack_ns:.2f} ns  (MET)")

# ── 2.  RTL cycle-accurate performance model ──────────────────────────────────
#
# attention_top: N=8 systolic array, NTILES=8, D=N*NTILES=64, INT8 inputs.
# Processes one 8x8 attention block (S=8 tokens, D=64 head-dim) per invocation.
#
# Host protocol (per invocation):
#   Preload  : Load Q(8x8) + V(8x8) + KT_tile_0(8x8) = 192 cycles
#   Tile loop: For tiles 0..NTILES-2 (7 non-last tiles):
#                MUL1_CLR(1) + MUL1_COMP(22) + TILE_CAP(1) + KT_reload(64) = 88 cy
#              Last tile: MUL1_CLR(1) + MUL1_COMP(22) + TILE_CAP(1)        = 24 cy
#   Post     : SCORE_CAP(1)
#              + 8 rows * [SOFT_START(1) + SOFT_WAIT(21)] = 176 cy
#                (softmax latency N=8: 4+WRECIP+N = 4+9+8 = 21 cy, per softmax_unit.sv header)
#              + MUL2_CLR(1) + MUL2_COMP(22) + RESULT_CAP(1) = 24 cy
#              + STREAM(64) + DONE_ST(1) = 65 cy

N            = 8
NTILES       = 8
D            = N * NTILES          # 64
SM_LAT_CY    = 4 + 9 + N          # 21 for N=8
MUL_COMP_CY  = 22                  # compute_count 0..21

PRELOAD_CY   = N*N + N*N + N*N                                     # 192
TILE_NOLAST  = 1 + MUL_COMP_CY + 1 + N*N                          # 88
TILE_LAST    = 1 + MUL_COMP_CY + 1                                 # 24
KTPASS_CY    = (NTILES - 1) * TILE_NOLAST + TILE_LAST             # 640
SOFTMAX_CY   = N * (1 + SM_LAT_CY)                                # 176
POST_CY      = 1 + SOFTMAX_CY + (1 + MUL_COMP_CY + 1) + N*N + 1  # 266

TOTAL_CY     = PRELOAD_CY + KTPASS_CY + POST_CY                   # 1098

lat_us = TOTAL_CY / (freq_mhz * 1e6) * 1e6    # us
lat_ms = lat_us / 1e3

FLOPS_PER_TILE = 4 * N * N * D                 # 16384  (4*S^2*D formula, S=N=8)
rtl_gflops     = (FLOPS_PER_TILE / 1e9) / (lat_ms / 1e3)
rtl_gflops_w   = rtl_gflops / (total_mw / 1e3)
rtl_gflops_mm2 = rtl_gflops / (cell_area_um2 / 1e6)

print(f"\n=== RTL Performance Model (one 8x8 attention tile, D={D}) ===")
print(f"  Total cycles  : {TOTAL_CY}  "
      f"({PRELOAD_CY} preload + {KTPASS_CY} tile-passes + {POST_CY} post)")
print(f"  Latency       : {lat_us:.2f} us  ({lat_ms:.4f} ms)")
print(f"  FLOPs / tile  : {FLOPS_PER_TILE:,}")
print(f"  Throughput    : {rtl_gflops:.4f} GFLOPS")
print(f"  GFLOPS/W      : {rtl_gflops_w:.2f}  (at {total_mw:.2f} mW total power)")
print(f"  GFLOPS/mm2    : {rtl_gflops_mm2:.4f}  (cell area = {cell_area_um2/1e6:.4f} mm2)")

# ── 3.  Technology normalization: TSMC 130nm -> 12nm ─────────────────────────
# Empirical scaling across ~10x node reduction (130->65->28->16->12nm):
#   Dynamic power per gate: ~10x lower  (voltage + capacitance shrink)
#   Max frequency:          ~10x higher
#   Cell area:              ~(12/130)^2 ~= 0.0085x
# These are order-of-magnitude estimates.
TECH_POWER_SCALE = 0.10
TECH_FREQ_SCALE  = 10.0
TECH_AREA_SCALE  = (12 / 130) ** 2    # ~0.0085

rtl_12nm_power_mw   = total_mw    * TECH_POWER_SCALE
rtl_12nm_freq_mhz   = freq_mhz    * TECH_FREQ_SCALE
rtl_12nm_lat_ms     = lat_ms      / TECH_FREQ_SCALE
rtl_12nm_gflops     = rtl_gflops  * TECH_FREQ_SCALE
rtl_12nm_gflops_w   = rtl_12nm_gflops / (rtl_12nm_power_mw / 1e3)
rtl_12nm_area_mm2   = (cell_area_um2 / 1e6) * TECH_AREA_SCALE
rtl_12nm_gflops_mm2 = rtl_12nm_gflops / rtl_12nm_area_mm2

print(f"\n=== RTL @ 12nm (technology-normalized, order-of-magnitude) ===")
print(f"  Est. dynamic power: {rtl_12nm_power_mw:.4f} mW")
print(f"  Est. frequency    : {rtl_12nm_freq_mhz:.0f} MHz")
print(f"  Est. latency      : {rtl_12nm_lat_ms*1e3:.2f} us")
print(f"  Est. GFLOPS       : {rtl_12nm_gflops:.3f}")
print(f"  Est. GFLOPS/W     : {rtl_12nm_gflops_w:.1f}")
print(f"  Est. cell area    : {rtl_12nm_area_mm2*1e6:.1f} um2  ({rtl_12nm_area_mm2:.6f} mm2)")
print(f"  Est. GFLOPS/mm2   : {rtl_12nm_gflops_mm2:.1f}")

# ── 4.  GPU baseline data ────────────────────────────────────────────────────
gpu = pd.read_csv(GPU_CSV)
print(f"\n=== GPU Baseline (TITAN V, FP16 SDPA, D=64) ===")
print(gpu[["S", "latency_mean_ms", "gflops", "avg_power_w", "gflops_per_watt"]].to_string(index=False))

TITAN_TDP_W    = 250.0
TITAN_AREA_MM2 = 815.0    # Volta GV100, TSMC 12nm

# ── 5.  Build comparison table ───────────────────────────────────────────────
rows = []

rows.append({
    "Design"         : "RTL SA (TSMC 130nm, S=8 tile)",
    "Process"        : "TSMC 130nm",
    "Seq_len"        : 8,
    "Power_W"        : total_mw / 1e3,
    "Latency_ms"     : lat_ms,
    "GFLOPS"         : rtl_gflops,
    "GFLOPS_per_W"   : rtl_gflops_w,
    "Cell_area_mm2"  : cell_area_um2 / 1e6,
    "GFLOPS_per_mm2" : rtl_gflops_mm2,
    "Note"           : "INT8 inputs, fixed S=8 tile",
})

rows.append({
    "Design"         : "RTL SA (est. @ 12nm)",
    "Process"        : "TSMC 12nm (est.)",
    "Seq_len"        : 8,
    "Power_W"        : rtl_12nm_power_mw / 1e3,
    "Latency_ms"     : rtl_12nm_lat_ms,
    "GFLOPS"         : rtl_12nm_gflops,
    "GFLOPS_per_W"   : rtl_12nm_gflops_w,
    "Cell_area_mm2"  : rtl_12nm_area_mm2,
    "GFLOPS_per_mm2" : rtl_12nm_gflops_mm2,
    "Note"           : "10x power, 10x freq, 0.0085x area (order-of-magnitude)",
})

for _, r in gpu.iterrows():
    rows.append({
        "Design"         : f"TITAN V (S={int(r.S)})",
        "Process"        : "TSMC 12nm",
        "Seq_len"        : int(r.S),
        "Power_W"        : r.avg_power_w,
        "Latency_ms"     : r.latency_mean_ms,
        "GFLOPS"         : r.gflops,
        "GFLOPS_per_W"   : r.gflops_per_watt,
        "Cell_area_mm2"  : TITAN_AREA_MM2,
        "GFLOPS_per_mm2" : r.gflops / TITAN_AREA_MM2,
        "Note"           : "FP16 tensor cores, full GPU die",
    })

cmp = pd.DataFrame(rows)
cmp.to_csv(CMP_CSV, index=False, float_format="%.4f")
print(f"\nComparison table saved -> {CMP_CSV}")
print(cmp[["Design","Power_W","GFLOPS","GFLOPS_per_W","Cell_area_mm2","GFLOPS_per_mm2"]].to_string(index=False))

# ── 6.  Plots ────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("RTL Systolic Array vs GPU Baseline (TITAN V)", fontsize=13, fontweight="bold")

gpu_labels  = [f"S={int(r.S)}" for _, r in gpu.iterrows()]
gpu_gflopsw = list(gpu["gflops_per_watt"])
gpu_gflops  = list(gpu["gflops"])
gpu_power   = list(gpu["avg_power_w"])

colors_rtl = ["#e07b39", "#c0392b"]
colors_gpu = plt.cm.Blues(np.linspace(0.45, 0.85, len(gpu_labels)))

patch1 = mpatches.Patch(color="#e07b39", label="RTL systolic array (130nm)")
patch2 = mpatches.Patch(color="#c0392b", label="RTL @ 12nm (est.)")
patch3 = mpatches.Patch(color=plt.cm.Blues(0.65), label="TITAN V GPU")

x_rtl = [0, 1]
x_gpu = list(range(3, 3 + len(gpu_labels)))

# ── Plot 1: GFLOPS/W ─────────────────────────────────────────────────────────
ax = axes[0]
bars_rtl = [rtl_gflops_w, rtl_12nm_gflops_w]
bars_gpu = gpu_gflopsw
ymax = max(bars_rtl + bars_gpu) * 1.18

for xi, val, col in zip(x_rtl, bars_rtl, colors_rtl):
    ax.bar(xi, val, color=col, edgecolor="black", linewidth=0.7, width=0.7)
    ax.text(xi, val + ymax * 0.01, f"{val:.0f}", ha="center", va="bottom", fontsize=8)

for xi, val, col in zip(x_gpu, bars_gpu, colors_gpu):
    ax.bar(xi, val, color=col, edgecolor="black", linewidth=0.7, width=0.7)
    ax.text(xi, val + ymax * 0.01, f"{val:.0f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x_rtl + x_gpu)
ax.set_xticklabels(["RTL\n(130nm)", "RTL\n(12nm est.)"] + [f"GPU\n{l}" for l in gpu_labels], fontsize=8)
ax.set_ylabel("GFLOPS/W")
ax.set_title("Energy Efficiency (GFLOPS/W)")
ax.set_ylim(0, ymax)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(handles=[patch1, patch2, patch3], fontsize=7, loc="upper left")

# ── Plot 2: Power ─────────────────────────────────────────────────────────────
ax = axes[1]
p_rtl = [total_mw / 1e3, rtl_12nm_power_mw / 1e3]

for xi, val, col in zip(x_rtl, p_rtl, colors_rtl):
    ax.bar(xi, val, color=col, edgecolor="black", linewidth=0.7, width=0.7)
    ax.text(xi, val * 1.5, f"{val*1e3:.1f} mW", ha="center", va="bottom", fontsize=7.5)

for xi, val, col in zip(x_gpu, gpu_power, colors_gpu):
    ax.bar(xi, val, color=col, edgecolor="black", linewidth=0.7, width=0.7)
    ax.text(xi, val * 1.08, f"{val:.0f} W", ha="center", va="bottom", fontsize=7.5)

ax.set_yscale("log")
ax.set_xticks(x_rtl + x_gpu)
ax.set_xticklabels(["RTL\n(130nm)", "RTL\n(12nm est.)"] + [f"GPU\n{l}" for l in gpu_labels], fontsize=8)
ax.set_ylabel("Power (W, log scale)")
ax.set_title("Power Consumption")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(handles=[patch1, patch2, patch3], fontsize=7)

# ── Plot 3: GFLOPS/mm2 ───────────────────────────────────────────────────────
ax = axes[2]
ae_rtl = [rtl_gflops_mm2, rtl_12nm_gflops_mm2]
ae_gpu = list(gpu["gflops"] / TITAN_AREA_MM2)

for xi, val, col in zip(x_rtl, ae_rtl, colors_rtl):
    ax.bar(xi, val, color=col, edgecolor="black", linewidth=0.7, width=0.7)
    ax.text(xi, val * 1.15, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

for xi, val, col in zip(x_gpu, ae_gpu, colors_gpu):
    ax.bar(xi, val, color=col, edgecolor="black", linewidth=0.7, width=0.7)
    ax.text(xi, val * 1.15, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

ax.set_yscale("log")
ax.set_xticks(x_rtl + x_gpu)
ax.set_xticklabels(["RTL\n(130nm)", "RTL\n(12nm est.)"] + [f"GPU\n{l}" for l in gpu_labels], fontsize=8)
ax.set_ylabel("GFLOPS/mm2 (log scale)")
ax.set_title("Area Efficiency (GFLOPS/mm2)")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(handles=[patch1, patch2, patch3], fontsize=7)

plt.tight_layout()
out_png = OUT_DIR / "rtl_gpu_comparison.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Comparison plot saved -> {out_png}")

# ── 7.  Key takeaways ────────────────────────────────────────────────────────
gpu_best_gfw = gpu["gflops_per_watt"].max()
gpu_s_best   = int(gpu.loc[gpu["gflops_per_watt"].idxmax(), "S"])
gpu_area_eff = (gpu["gflops"] / TITAN_AREA_MM2).max()

print("\n=== Key Comparison Takeaways ===")
print(f"  RTL power vs GPU (min):   {total_mw/1e3:.4f} W  vs  {gpu['avg_power_w'].min():.1f} W  "
      f"->  {gpu['avg_power_w'].min()/(total_mw/1e3):.0f}x lower for RTL")
print(f"  RTL GFLOPS/W (130nm):     {rtl_gflops_w:.1f}  vs  GPU best {gpu_best_gfw:.1f} (S={gpu_s_best})")
print(f"  RTL GFLOPS/W (12nm est.): {rtl_12nm_gflops_w:.0f}  vs  GPU best {gpu_best_gfw:.1f}  "
      f"->  {rtl_12nm_gflops_w/gpu_best_gfw:.0f}x better for RTL")
print(f"  RTL cell area (130nm):    {cell_area_um2/1e6:.4f} mm2  vs  TITAN V {TITAN_AREA_MM2:.0f} mm2  "
      f"->  {TITAN_AREA_MM2/(cell_area_um2/1e6):.0f}x smaller")
print(f"  RTL cell area (12nm est): {rtl_12nm_area_mm2*1e6:.1f} um2  "
      f"({TITAN_AREA_MM2/rtl_12nm_area_mm2:.0f}x smaller than TITAN V)")
print(f"  RTL GFLOPS/mm2 (130nm):   {rtl_gflops_mm2:.4f}")
print(f"  RTL GFLOPS/mm2 (12nm est):{rtl_12nm_gflops_mm2:.1f}")
print(f"  GPU best GFLOPS/mm2:      {gpu_area_eff:.2f}  (S={int(gpu.loc[gpu['gflops'].idxmax(),'S'])})")
print(f"\nNote: RTL processes S=8 tiles only; GPU results are for S=1024-16384.")
print(f"      RTL workload: INT8, single-precision accum, fixed-function attention.")
print(f"      GPU workload: FP16 tensor cores, variable sequence length, general SDPA.")
print(f"      Technology normalization is order-of-magnitude (Dennard-like scaling).")
