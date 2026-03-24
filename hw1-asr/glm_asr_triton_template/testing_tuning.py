"""
Optimization 1: Tile/Block Size Tuning
Benchmarks attention kernels with different num_warps and num_stages configurations.

Save this file as:
  /workspace/edin-mls-26-spring/hw1-asr/glm_asr_triton_template/test_tuning.py

Run with:
  cd /workspace/edin-mls-26-spring/hw1-asr/glm_asr_triton_template
  python test_tuning.py
"""

import torch
import triton
import triton.language as tl
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attention import attention_scores_kernel, softmax_inplace_kernel, attention_output_kernel
from layers import softmax_kernel

device = torch.device("cuda")
torch.cuda.empty_cache()


def benchmark_kernel(fn, n_warmup=20, n_runs=200):
    """Benchmark a kernel function that takes no args (already wrapped in a lambda)."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000  # ms


def test_attention_scores_tuning():
    """Test attention_scores_kernel with different num_warps and num_stages."""
    print("=" * 70)
    print("ATTENTION SCORES KERNEL - TUNING")
    print("=" * 70)

    seq_q, seq_k, head_dim, num_heads, batch = 71, 71, 128, 16, 1
    BH = batch * num_heads

    q = torch.randn(BH, seq_q, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(BH, seq_k, head_dim, device=device, dtype=torch.float32)
    scores = torch.zeros(BH, seq_q, seq_k, device=device, dtype=torch.float32)
    scale = 1.0 / (head_dim ** 0.5)

    seq_k_p = triton.next_power_of_2(seq_k)
    head_dim_p = triton.next_power_of_2(head_dim)

    configs = [
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3),
        (4, 1), (4, 2), (4, 3),
        (8, 1), (8, 2), (8, 3),
    ]

    print(f"\nConfig: seq_q={seq_q}, seq_k={seq_k}, head_dim={head_dim}, heads={num_heads}")
    print(f"{'num_warps':<12} {'num_stages':<12} {'Time (ms)':<12} {'vs baseline':<12}")
    print("-" * 48)

    baseline_time = None
    best_time = float('inf')
    best_config = None

    for nw, ns in configs:
        try:
            def run():
                attention_scores_kernel[(BH, seq_q)](
                    q, k, scores, scale, seq_k, head_dim,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    scores.stride(0), scores.stride(1), scores.stride(2),
                    BLOCK_K=seq_k_p, BLOCK_D=head_dim_p,
                    num_warps=nw, num_stages=ns,
                )

            t = benchmark_kernel(run)

            if baseline_time is None:
                baseline_time = t

            speedup = f"{baseline_time / t:.2f}x"
            print(f"{nw:<12} {ns:<12} {t:<12.4f} {speedup:<12}")

            if t < best_time:
                best_time = t
                best_config = (nw, ns)
        except Exception as e:
            print(f"{nw:<12} {ns:<12} {'FAILED':<12} {str(e)[:30]}")

    print(f"\nBest config: num_warps={best_config[0]}, num_stages={best_config[1]} ({best_time:.4f} ms)")
    print(f"Speedup over default: {baseline_time / best_time:.2f}x")
    return best_config, baseline_time, best_time


def test_softmax_inplace_tuning():
    """Test softmax_inplace_kernel with different num_warps and num_stages."""
    print("\n" + "=" * 70)
    print("SOFTMAX INPLACE KERNEL - TUNING")
    print("=" * 70)

    seq_k = 71
    num_rows = 16 * 71
    scores = torch.randn(num_rows, seq_k, device=device, dtype=torch.float32)
    block = triton.next_power_of_2(seq_k)

    configs = [
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3),
        (4, 1), (4, 2), (4, 3),
        (8, 1), (8, 2), (8, 3),
    ]

    print(f"\nConfig: num_rows={num_rows}, seq_k={seq_k}")
    print(f"{'num_warps':<12} {'num_stages':<12} {'Time (ms)':<12} {'vs baseline':<12}")
    print("-" * 48)

    baseline_time = None
    best_time = float('inf')
    best_config = None

    for nw, ns in configs:
        try:
            def run():
                softmax_inplace_kernel[(num_rows,)](
                    scores, scores.stride(0), seq_k, BLOCK_SIZE=block,
                    num_warps=nw, num_stages=ns,
                )

            t = benchmark_kernel(run)

            if baseline_time is None:
                baseline_time = t

            speedup = f"{baseline_time / t:.2f}x"
            print(f"{nw:<12} {ns:<12} {t:<12.4f} {speedup:<12}")

            if t < best_time:
                best_time = t
                best_config = (nw, ns)
        except Exception as e:
            print(f"{nw:<12} {ns:<12} {'FAILED':<12} {str(e)[:30]}")

    print(f"\nBest config: num_warps={best_config[0]}, num_stages={best_config[1]} ({best_time:.4f} ms)")
    print(f"Speedup over default: {baseline_time / best_time:.2f}x")
    return best_config, baseline_time, best_time


def test_attention_output_tuning():
    """Test attention_output_kernel with different num_warps and num_stages."""
    print("\n" + "=" * 70)
    print("ATTENTION OUTPUT KERNEL - TUNING")
    print("=" * 70)

    seq_q, seq_k, head_dim, num_heads, batch = 71, 71, 128, 16, 1
    BH = batch * num_heads

    weights = torch.randn(BH, seq_q, seq_k, device=device, dtype=torch.float32)
    # Make weights look like softmax output
    weights = torch.softmax(weights, dim=-1)
    v = torch.randn(BH, seq_k, head_dim, device=device, dtype=torch.float32)
    output = torch.zeros(BH, seq_q, head_dim, device=device, dtype=torch.float32)

    seq_k_p = triton.next_power_of_2(seq_k)
    head_dim_p = triton.next_power_of_2(head_dim)

    configs = [
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3),
        (4, 1), (4, 2), (4, 3),
        (8, 1), (8, 2), (8, 3),
    ]

    print(f"\nConfig: seq_q={seq_q}, seq_k={seq_k}, head_dim={head_dim}, heads={num_heads}")
    print(f"{'num_warps':<12} {'num_stages':<12} {'Time (ms)':<12} {'vs baseline':<12}")
    print("-" * 48)

    baseline_time = None
    best_time = float('inf')
    best_config = None

    for nw, ns in configs:
        try:
            def run():
                attention_output_kernel[(BH, seq_q)](
                    weights, v, output, seq_k, head_dim,
                    weights.stride(0), weights.stride(1), weights.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    output.stride(0), output.stride(1), output.stride(2),
                    BLOCK_K=seq_k_p, BLOCK_D=head_dim_p,
                    num_warps=nw, num_stages=ns,
                )

            t = benchmark_kernel(run)

            if baseline_time is None:
                baseline_time = t

            speedup = f"{baseline_time / t:.2f}x"
            print(f"{nw:<12} {ns:<12} {t:<12.4f} {speedup:<12}")

            if t < best_time:
                best_time = t
                best_config = (nw, ns)
        except Exception as e:
            print(f"{nw:<12} {ns:<12} {'FAILED':<12} {str(e)[:30]}")

    print(f"\nBest config: num_warps={best_config[0]}, num_stages={best_config[1]} ({best_time:.4f} ms)")
    print(f"Speedup over default: {baseline_time / best_time:.2f}x")
    return best_config, baseline_time, best_time


def test_softmax_kernel_tuning():
    """Test standalone softmax_kernel with different num_warps and num_stages."""
    print("\n" + "=" * 70)
    print("SOFTMAX KERNEL (STANDALONE) - TUNING")
    print("=" * 70)

    n_rows = 16 * 71
    n_cols = 71
    x = torch.randn(n_rows, n_cols, device=device, dtype=torch.float32)
    y = torch.zeros_like(x)
    block = triton.next_power_of_2(n_cols)

    configs = [
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3),
        (4, 1), (4, 2), (4, 3),
        (8, 1), (8, 2), (8, 3),
    ]

    print(f"\nConfig: rows={n_rows}, cols={n_cols}")
    print(f"{'num_warps':<12} {'num_stages':<12} {'Time (ms)':<12} {'vs baseline':<12}")
    print("-" * 48)

    baseline_time = None
    best_time = float('inf')
    best_config = None

    for nw, ns in configs:
        try:
            def run():
                softmax_kernel[(n_rows,)](
                    x, y, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=block,
                    num_warps=nw, num_stages=ns,
                )

            t = benchmark_kernel(run)

            if baseline_time is None:
                baseline_time = t

            speedup = f"{baseline_time / t:.2f}x"
            print(f"{nw:<12} {ns:<12} {t:<12.4f} {speedup:<12}")

            if t < best_time:
                best_time = t
                best_config = (nw, ns)
        except Exception as e:
            print(f"{nw:<12} {ns:<12} {'FAILED':<12} {str(e)[:30]}")

    print(f"\nBest config: num_warps={best_config[0]}, num_stages={best_config[1]} ({best_time:.4f} ms)")
    print(f"Speedup over default: {baseline_time / best_time:.2f}x")
    return best_config, baseline_time, best_time


if __name__ == "__main__":
    print("=" * 70)
    print("OPTIMIZATION 1: TILE/BLOCK SIZE TUNING")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    results = {}

    r1 = test_attention_scores_tuning()
    results['attention_scores'] = r1

    r2 = test_softmax_inplace_tuning()
    results['softmax_inplace'] = r2

    r3 = test_attention_output_tuning()
    results['attention_output'] = r3

    r4 = test_softmax_kernel_tuning()
    results['softmax_kernel'] = r4

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Kernel':<25} {'Best Config':<20} {'Baseline (ms)':<15} {'Best (ms)':<15} {'Speedup':<10}")
    print("-" * 85)
    for name, (config, base, best) in results.items():
        print(f"{name:<25} w={config[0]},s={config[1]:<13} {base:<15.4f} {best:<15.4f} {base/best:<10.2f}x")