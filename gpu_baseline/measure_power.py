"""Task 3: GPU power measurement during sustained attention runs.

Launches nvidia-smi power logging in a subprocess, runs attention for ~10s per
S config to get stable readings, then parses the log to compute average power,
energy, and GFLOPS/W. Writes power_results.csv and merges into gpu_results_power.csv.

Limitation: on a shared GPU the idle/background power from other users is
included; note this in the summary if utilization shows other activity.
"""
import csv
import os
import signal
import subprocess
import tempfile
import time

import torch
import torch.nn.functional as F

SEQ_LENS = [1024, 2048, 4096, 8192, 16384]
D = 64
WARMUP = 50
TARGET_SECONDS = 10.0   # run each config for ~10 s of active measurement
POWER_LOG = "power_log.csv"
OUT_CSV = "power_results.csv"
MERGED_CSV = "gpu_results_power.csv"


def run_attention_for_seconds(S: int, seconds: float) -> tuple[float, float]:
    """Run SDPA for `seconds` wall-clock seconds. Returns (iter_count, actual_seconds)."""
    Q = torch.randn(1, 1, S, D, device="cuda", dtype=torch.float16)
    K = torch.randn(1, 1, S, D, device="cuda", dtype=torch.float16)
    V = torch.randn(1, 1, S, D, device="cuda", dtype=torch.float16)

    # warmup
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    iters = 0
    t_start = time.time()
    t_end = t_start + seconds
    while time.time() < t_end:
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V)
        iters += 1
    torch.cuda.synchronize()
    actual = time.time() - t_start
    return iters, actual


def parse_power_log(log_path: str, t_start: float, t_end: float) -> list[float]:
    """Parse nvidia-smi CSV log and return power readings within [t_start, t_end]."""
    readings = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("timestamp"):
                continue
            # Format: "2024/01/01 12:00:00.000, 150.00 W, 95 %"
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                # Parse timestamp
                ts_str = parts[0]
                import datetime
                dt = datetime.datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S.%f")
                ts = dt.timestamp()
                # Parse power (strip " W" suffix)
                pwr_str = parts[1].replace(" W", "").strip()
                pwr = float(pwr_str)
                if t_start <= ts <= t_end:
                    readings.append(pwr)
            except (ValueError, IndexError):
                continue
    return readings


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Measuring power for {TARGET_SECONDS:.0f}s per config...\n")

    # Start nvidia-smi logging to a temp file that we persist as POWER_LOG
    smi_proc = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,power.draw,utilization.gpu",
            "--format=csv",
            "-lms", "100",
        ],
        stdout=open(POWER_LOG, "w"),
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)  # let it write a few idle samples

    results = []
    for S in SEQ_LENS:
        print(f"  S={S:6d}: running {TARGET_SECONDS:.0f}s of attention...")
        t0 = time.time()
        iters, actual = run_attention_for_seconds(S, TARGET_SECONDS)
        t1 = time.time()

        results.append({
            "S": S,
            "t_start": t0,
            "t_end": t1,
            "iters": iters,
            "actual_s": actual,
        })
        print(f"           done ({iters} iters in {actual:.2f}s)")
        time.sleep(0.5)  # brief idle between configs

    # Stop nvidia-smi
    smi_proc.terminate()
    smi_proc.wait()
    time.sleep(0.2)

    # Parse power log and compute stats per config
    power_rows = []
    for r in results:
        readings = parse_power_log(POWER_LOG, r["t_start"], r["t_end"])
        if readings:
            avg_w = sum(readings) / len(readings)
            min_w = min(readings)
            max_w = max(readings)
            n_samples = len(readings)
        else:
            avg_w = min_w = max_w = float("nan")
            n_samples = 0
            print(f"  WARNING: no power readings found for S={r['S']}")

        power_rows.append({
            "S": r["S"],
            "avg_power_w": round(avg_w, 2),
            "min_power_w": round(min_w, 2),
            "max_power_w": round(max_w, 2),
            "n_power_samples": n_samples,
        })
        print(f"  S={r['S']:6d}: avg={avg_w:.1f}W  min={min_w:.1f}W  max={max_w:.1f}W  n={n_samples}")

    # Write power_results.csv
    fields = ["S", "avg_power_w", "min_power_w", "max_power_w", "n_power_samples"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(power_rows)
    print(f"\nPower results written to {OUT_CSV}")

    # Merge with gpu_results.csv + gpu_results_ncu.csv -> gpu_results_power.csv
    base_rows = {}
    if os.path.exists("gpu_results.csv"):
        with open("gpu_results.csv") as f:
            for row in csv.DictReader(f):
                base_rows[int(row["S"])] = dict(row)

    ncu_rows = {}
    if os.path.exists("gpu_results_ncu.csv"):
        with open("gpu_results_ncu.csv") as f:
            for row in csv.DictReader(f):
                ncu_rows[int(row["S"])] = dict(row)

    merged = []
    for pr in power_rows:
        S = pr["S"]
        row = {}
        if S in base_rows:
            row.update(base_rows[S])
        if S in ncu_rows:
            # Merge ncu columns, skip duplicate S
            for k, v in ncu_rows[S].items():
                if k != "S":
                    row[k] = v
        row.update(pr)

        # Compute energy and GFLOPS/W
        lat_ms = float(row.get("latency_mean_ms", "nan"))
        gflops_throughput = float(row.get("gflops", "nan"))
        avg_w = pr["avg_power_w"]
        if lat_ms == lat_ms and avg_w == avg_w:  # not nan
            energy_mj = avg_w * lat_ms  # W * ms = mJ
            # GFLOPS/W = sustained throughput / power (industry-standard metric)
            gflops_per_watt = gflops_throughput / avg_w if avg_w > 0 else float("nan")
        else:
            energy_mj = float("nan")
            gflops_per_watt = float("nan")

        row["energy_mj"] = round(energy_mj, 4)
        row["gflops_per_watt"] = round(gflops_per_watt, 4)
        merged.append(row)

    if merged:
        all_fields = list(merged[0].keys())
        with open(MERGED_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(merged)
        print(f"Merged results written to {MERGED_CSV}")
        print("\n--- GFLOPS/W summary ---")
        for row in merged:
            print(f"  S={row['S']:6}  power={row['avg_power_w']}W  "
                  f"energy={row['energy_mj']}mJ  GFLOPS/W={row['gflops_per_watt']}")


if __name__ == "__main__":
    main()
