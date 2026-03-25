"""Test different TILE_M/TILE_N/TILE_K for linear_kernel_tf32"""
import torch
import triton
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layers import linear_kernel_tf32, pad_to_multiple

device = torch.device("cuda")

def bench_linear(M, N, K, tile_m, tile_n, tile_k, n_runs=200):
    M_pad = pad_to_multiple(M, tile_m)
    K_pad = pad_to_multiple(K, tile_k)
    N_pad = pad_to_multiple(N, tile_n)
    
    a = torch.randn(M_pad, K_pad, device=device, dtype=torch.float32)
    b = torch.randn(K_pad, N_pad, device=device, dtype=torch.float32)
    c = torch.zeros(M_pad, N_pad, device=device, dtype=torch.float32)
    
    grid = (triton.cdiv(M_pad, tile_m), triton.cdiv(N_pad, tile_n))
    
    for _ in range(20):
        linear_kernel_tf32[grid](
            a, b, c, M_pad, N_pad, K_pad,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=tile_m, BLOCK_N=tile_n, BLOCK_K=tile_k,
        )
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(n_runs):
        linear_kernel_tf32[grid](
            a, b, c, M_pad, N_pad, K_pad,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=tile_m, BLOCK_N=tile_n, BLOCK_K=tile_k,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000

# Model's actual matrix dimensions during decode
# Text decoder: hidden=2048, intermediate=5632 (MLP), heads*head_dim for attention projections
test_cases = [
    ("Decoder MLP up (1x2048 -> 5632)", 1, 5632, 2048),
    ("Decoder MLP down (1x5632 -> 2048)", 1, 2048, 5632),
    ("Decoder attn proj (1x2048 -> 2048)", 1, 2048, 2048),
    ("Encoder linear (188x1280 -> 5120)", 188, 5120, 1280),
]

tile_configs = [
    (32, 32, 16),
    (32, 32, 32),
    (64, 64, 32),   # current default
    (64, 64, 64),
    (128, 64, 32),
    (64, 128, 32),
    (128, 128, 32),
]

print("Linear Kernel Tile Size Tuning")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 80)

for name, M, N, K in test_cases:
    print(f"\n{name} (M={M}, N={N}, K={K})")
    print(f"{'TILE_M':<10} {'TILE_N':<10} {'TILE_K':<10} {'Time (ms)':<12} {'vs default':<12}")
    print("-" * 54)
    
    default_time = None
    best_time = float('inf')
    best_config = None
    
    for tm, tn, tk in tile_configs:
        try:
            t = bench_linear(M, N, K, tm, tn, tk)
            if tm == 64 and tn == 64 and tk == 32:
                default_time = t
            speedup = f"{default_time/t:.2f}x" if default_time else "baseline"
            print(f"{tm:<10} {tn:<10} {tk:<10} {t:<12.4f} {speedup:<12}")
            if t < best_time:
                best_time = t
                best_config = (tm, tn, tk)
        except Exception as e:
            print(f"{tm:<10} {tn:<10} {tk:<10} {'FAILED':<12} {str(e)[:30]}")
    
    print(f"Best: TILE_M={best_config[0]}, TILE_N={best_config[1]}, TILE_K={best_config[2]} ({best_time:.4f}ms)")