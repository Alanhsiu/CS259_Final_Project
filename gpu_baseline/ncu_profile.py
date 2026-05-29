"""Profile SDPA with ncu for each S; write gpu_results_ncu.csv.

Run in tmux — ncu replays kernels and takes 10-30 min total.
Usage: python ncu_profile.py
"""
import csv
import io
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SDPA_SCRIPT = os.path.join(SCRIPT_DIR, "sdpa_one_pass.py")
OUT_CSV = os.path.join(SCRIPT_DIR, "data", "gpu_results_ncu.csv")

SEQ_LENS = [1024, 2048, 4096, 8192, 16384]
D = 64

METRICS = ",".join([
    # L1-level global load/store bytes: captures actual kernel memory traffic
    # regardless of whether data was evicted to DRAM during the kernel.
    # dram__bytes_write.sum is 0 for small S because the output stays in L2.
    "l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld.sum",   # L2→L1 global loads
    "l1tex__m_l1tex2xbar_write_bytes_mem_lg_op_st.sum",  # L1→L2 global stores
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
])

READ_M  = "l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld.sum"
WRITE_M = "l1tex__m_l1tex2xbar_write_bytes_mem_lg_op_st.sum"

# On TITAN V (sm_70) PyTorch uses xformers memory-efficient attention:
# kernel name is fmha_cutlassF_f16_aligned_*_sm70
SDPA_PAT = re.compile(r"fmha|flash_attn|AttentionKernel|sdp_attention", re.IGNORECASE)


def run_ncu(S: int) -> str:
    cmd = [
        "ncu", "--csv",
        "--metrics", METRICS,
        sys.executable, SDPA_SCRIPT, str(S),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and result.stderr:
        print(f"    ncu stderr (first 500 chars): {result.stderr[:500]}")
    return result.stdout


def parse_ncu_csv(raw: str) -> dict:
    """Return {kernel_name: {metric_name: float}} from ncu --csv stdout."""
    lines = [l for l in raw.splitlines()
             if l and not l.startswith("==") and not l.startswith("WARNING")]
    if not lines:
        return {}
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    kernels: dict = {}
    for row in reader:
        name = row.get("Kernel Name", "").strip()
        metric = row.get("Metric Name", "").strip()
        val_str = row.get("Metric Value", "0").strip().replace(",", "")
        try:
            val = float(val_str)
        except ValueError:
            val = 0.0
        kernels.setdefault(name, {})[metric] = val
    return kernels


def pick_sdpa_kernels(kernels: dict) -> list:
    """Prefer kernels matching the SDPA pattern; fall back to all kernels."""
    matched = [k for k in kernels if SDPA_PAT.search(k)]
    return matched if matched else list(kernels.keys())


def aggregate(kernels: dict, names: list) -> dict:
    """Sum memory bytes; weight utilization pcts by each kernel's read bytes."""
    dram_r = sum(kernels[n].get(READ_M, 0) for n in names)
    dram_w = sum(kernels[n].get(WRITE_M, 0) for n in names)
    weights = [kernels[n].get(READ_M, 1) or 1 for n in names]
    total_w = sum(weights)

    def wavg(metric: str) -> float:
        return sum(kernels[n].get(metric, 0) * w
                   for n, w in zip(names, weights)) / total_w

    return {
        "l1_read_bytes": dram_r,
        "l1_write_bytes": dram_w,
        "sm_throughput_pct": wavg("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "dram_throughput_pct": wavg("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
        "warps_active_pct": wavg("sm__warps_active.avg.pct_of_peak_sustained_active"),
    }


def main():
    print("NOTE: ncu replays every kernel — expect 10-30 min. Run this in tmux.")
    print(f"Profiling S in {SEQ_LENS}, D={D}\n")

    rows = []
    for S in SEQ_LENS:
        print(f"  S={S}: running ncu ...", flush=True)
        raw = run_ncu(S)
        kernels = parse_ncu_csv(raw)

        if not kernels:
            print(f"  S={S}: WARNING — no kernels parsed. Skipping.")
            continue

        names = pick_sdpa_kernels(kernels)
        print(f"    kernels used: {names}")
        m = aggregate(kernels, names)

        flops = 4 * S * S * D
        total_bytes = m["l1_read_bytes"] + m["l1_write_bytes"]
        ai = flops / total_bytes if total_bytes > 0 else 0.0

        rows.append({
            "S": S,
            "flops": flops,
            "l1_read_bytes": int(m["l1_read_bytes"]),
            "l1_write_bytes": int(m["l1_write_bytes"]),
            "total_bytes": int(total_bytes),
            "achieved_ai_flops_per_byte": round(ai, 4),
            "sm_throughput_pct": round(m["sm_throughput_pct"], 2),
            "dram_throughput_pct": round(m["dram_throughput_pct"], 2),
            "warps_active_pct": round(m["warps_active_pct"], 2),
        })
        print(f"    l1_r={m['l1_read_bytes']/1e6:.1f} MB  "
              f"l1_w={m['l1_write_bytes']/1e6:.1f} MB  "
              f"AI={ai:.2f} FLOP/B  "
              f"SM={m['sm_throughput_pct']:.1f}%  "
              f"DRAM={m['dram_throughput_pct']:.1f}%")

    if not rows:
        print("\nNo rows collected — nothing written.")
        return

    fieldnames = [
        "S", "flops",
        "l1_read_bytes", "l1_write_bytes", "total_bytes",
        "achieved_ai_flops_per_byte",
        "sm_throughput_pct", "dram_throughput_pct", "warps_active_pct",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {OUT_CSV}")


if __name__ == "__main__":
    main()
