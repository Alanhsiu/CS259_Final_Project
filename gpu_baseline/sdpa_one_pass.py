"""Single SDPA forward pass for ncu to profile. Usage: python sdpa_one_pass.py <S>"""
import sys

import torch
import torch.nn.functional as F

S = int(sys.argv[1])
D = 64

Q = torch.randn(1, 1, S, D, device="cuda", dtype=torch.float16)
K = torch.randn(1, 1, S, D, device="cuda", dtype=torch.float16)
V = torch.randn(1, 1, S, D, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()

with torch.no_grad():
    out = F.scaled_dot_product_attention(Q, K, V)

torch.cuda.synchronize()
