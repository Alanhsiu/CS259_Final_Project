"""Task 5: Extract real Q*K^T attention tiles from GPT-2 and quantize to INT8.

Uses GPT-2 small (117M) from HuggingFace. Runs a forward pass with a hook on
the first attention layer to capture the Q and K projections before softmax.
Extracts one 8x8 tile from position (0,0) in the Q*K^T matrix, quantizes both
Q-tile and K-tile to INT8 using per-tensor symmetric quantization, verifies
INT8 matmul matches the FP32 reference, and saves the hex files to
test_vectors/real_tile/.
"""
import os
import struct

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

TILE = 8
OUT_DIR = "../test_vectors/real_tile"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- GPT-2 hook to capture Q, K projections ----------
captured = {}

def make_qk_hook(module, input, output):
    """Hook on GPT2Attention to capture Q and K before scaling."""
    # GPT2Attention.c_attn splits into (Q, K, V) along last dim
    x = input[0]  # (batch, seq, hidden)
    qkv = module.c_attn(x)  # (batch, seq, 3*hidden)
    hidden = module.embed_dim
    q = qkv[:, :, :hidden]          # (1, seq, 768)
    k = qkv[:, :, hidden:2*hidden]  # (1, seq, 768)
    # Reshape to (num_heads, seq, head_dim) — GPT-2 small: 12 heads, 64 dims
    num_heads = module.num_heads
    head_dim = hidden // num_heads
    batch = q.shape[0]
    seq = q.shape[1]
    q = q.view(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3)  # (1,12,seq,64)
    k = k.view(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3)  # (1,12,seq,64)
    captured["q"] = q.detach().cpu().float()
    captured["k"] = k.detach().cpu().float()


def symmetric_quantize_int8(tensor: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric INT8 quantization. Returns (quantized, scale)."""
    amax = np.abs(tensor).max()
    scale = amax / 127.0
    if scale == 0:
        scale = 1e-8
    q = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
    return q, scale


def save_hex_int8(arr: np.ndarray, path: str):
    """Save INT8 matrix in row-major order, 2 hex chars per line (signed byte)."""
    with open(path, "w") as f:
        for val in arr.flatten():
            # convert to Python int first so bitwise op works on signed byte
            v = int(val) & 0xFF
            f.write(f"{v:02x}\n")


def save_hex_int32(arr: np.ndarray, path: str):
    """Save INT32 matrix in row-major order, 8 hex chars per line."""
    with open(path, "w") as f:
        for val in arr.flatten():
            v = int(val) & 0xFFFFFFFF
            f.write(f"{v:08x}\n")


def main():
    print("Loading GPT-2 small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    # Register hook on first attention layer
    first_attn = model.h[0].attn
    hook = first_attn.register_forward_hook(make_qk_hook)

    # Run a forward pass with a short sentence
    text = "The quick brown fox jumps over the lazy dog ."
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        _ = model(**inputs)
    hook.remove()

    q_all = captured["q"]  # (1, 12, seq_len, 64)
    k_all = captured["k"]  # (1, 12, seq_len, 64)
    seq_len = q_all.shape[2]
    print(f"Captured Q/K: shape {list(q_all.shape)}, seq_len={seq_len}")

    # Use head 0, tokens 0..7 for the 8x8 tile
    # Q tile: shape (8, 64), K tile: shape (8, 64)
    Q_fp32 = q_all[0, 0, :TILE, :].numpy()    # (8, 64)
    K_fp32 = k_all[0, 0, :TILE, :].numpy()    # (8, 64)

    # Compute FP32 reference: C = Q * K^T, shape (8, 8)
    C_fp32 = Q_fp32 @ K_fp32.T

    # Quantize to INT8
    Q_int8, q_scale = symmetric_quantize_int8(Q_fp32)
    K_int8, k_scale = symmetric_quantize_int8(K_fp32)

    # INT32 matmul on INT8 inputs
    C_int32 = Q_int8.astype(np.int32) @ K_int8.astype(np.int32).T

    # Dequantized result for comparison
    C_dequant = C_int32.astype(np.float32) * (q_scale * k_scale)

    # Error stats
    abs_err = np.abs(C_fp32 - C_dequant)
    rel_err = abs_err / (np.abs(C_fp32) + 1e-8)
    print(f"\nQuantization error stats (FP32 vs INT8-dequant):")
    print(f"  Max abs error: {abs_err.max():.4f}")
    print(f"  Mean abs error: {abs_err.mean():.4f}")
    print(f"  Max rel error: {rel_err.max():.4%}")
    print(f"  Q scale={q_scale:.6f}, K scale={k_scale:.6f}")

    # Save hex files
    q_path = os.path.join(OUT_DIR, "real_q_tile.hex")
    k_path = os.path.join(OUT_DIR, "real_k_tile.hex")
    c_path = os.path.join(OUT_DIR, "real_c_expected.hex")
    meta_path = os.path.join(OUT_DIR, "real_tile_meta.txt")

    save_hex_int8(Q_int8, q_path)
    save_hex_int8(K_int8, k_path)
    save_hex_int32(C_int32, c_path)

    with open(meta_path, "w") as f:
        f.write("Real GPT-2 attention tile (head 0, tokens 0-7)\n")
        f.write(f"Source: GPT-2 small (gpt2), first attention layer, head 0\n")
        f.write(f"Input text: \"{text}\"\n")
        f.write(f"Tile size: {TILE}x{TILE}\n")
        f.write(f"Q/K matrix dimensions: {TILE} rows x 64 cols\n")
        f.write(f"C = Q * K^T: {TILE}x{TILE}\n")
        f.write(f"Quantization: symmetric INT8, per-tensor\n")
        f.write(f"  Q scale: {q_scale:.8f}\n")
        f.write(f"  K scale: {k_scale:.8f}\n")
        f.write(f"  C dequant scale: {q_scale * k_scale:.8f}\n")
        f.write(f"Quantization error (max abs): {abs_err.max():.4f}\n")
        f.write(f"Quantization error (mean abs): {abs_err.mean():.4f}\n")
        f.write(f"\nFiles:\n")
        f.write(f"  real_q_tile.hex  - Q tile (8x64 INT8), 2 hex chars/line, row-major\n")
        f.write(f"  real_k_tile.hex  - K tile (8x64 INT8), 2 hex chars/line, row-major\n")
        f.write(f"  real_c_expected.hex - C = Q*K^T (8x8 INT32), 8 hex chars/line, row-major\n")
        f.write(f"\nNote: The RTL systolic array computes 8x8 x 8x8 tiles.\n")
        f.write(f"      For full Q*K^T, decompose into 64/8=8 partial sums per output tile.\n")
        f.write(f"\nVerilog usage:\n")
        f.write(f'  $readmemh("real_q_tile.hex", q_mem);\n')
        f.write(f'  $readmemh("real_k_tile.hex", k_mem);\n')
        f.write(f'  $readmemh("real_c_expected.hex", c_expected);\n')

    print(f"\nSaved to {OUT_DIR}/:")
    print(f"  real_q_tile.hex  ({TILE*64} lines, Q INT8)")
    print(f"  real_k_tile.hex  ({TILE*64} lines, K INT8)")
    print(f"  real_c_expected.hex  ({TILE*TILE} lines, C INT32)")
    print(f"  real_tile_meta.txt")

    # Verification: read back and check
    def load_hex_int8(path, rows, cols):
        vals = []
        with open(path) as f:
            for line in f:
                v = int(line.strip(), 16)
                if v > 127:
                    v -= 256
                vals.append(v)
        return np.array(vals, dtype=np.int8).reshape(rows, cols)

    def load_hex_int32(path, rows, cols):
        vals = []
        with open(path) as f:
            for line in f:
                v = int(line.strip(), 16)
                if v >= 0x80000000:
                    v -= 0x100000000
                vals.append(v)
        return np.array(vals, dtype=np.int32).reshape(rows, cols)

    Q_verify = load_hex_int8(q_path, TILE, 64)
    K_verify = load_hex_int8(k_path, TILE, 64)
    C_verify = load_hex_int32(c_path, TILE, TILE)
    C_recompute = Q_verify.astype(np.int32) @ K_verify.astype(np.int32).T
    if np.array_equal(C_recompute, C_verify):
        print("\nVerification: PASS — readback C matches recomputed Q*K^T ✓")
    else:
        print("\nVerification: FAIL — mismatch detected!")
        print("Max diff:", np.abs(C_recompute - C_verify).max())


if __name__ == "__main__":
    main()
