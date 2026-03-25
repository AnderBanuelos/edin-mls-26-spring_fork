import torch
import triton
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Temporarily modify FLASH_BLOCK_N and reimport
import attention as attn_module

device = torch.device("cuda")

def bench_flash(seq_k, block_n, n_runs=200):
    """Benchmark FlashAttention with specific BLOCK_N and seq_k."""
    batch, heads, head_dim = 1, 16, 128
    q = torch.randn(batch, heads, 1, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch, heads, seq_k, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch, heads, seq_k, head_dim, device=device, dtype=torch.float32)
    
    head_dim_padded = triton.next_power_of_2(head_dim)
    q_flat = q.reshape(batch * heads, 1, head_dim).to(torch.float32)
    k_flat = k.reshape(batch * heads, seq_k, head_dim).to(torch.float32)
    v_flat = v.reshape(batch * heads, seq_k, head_dim).to(torch.float32)
    output = torch.empty(batch * heads, 1, head_dim_padded, device=device, dtype=torch.float32)
    
    num_blocks = triton.cdiv(seq_k, block_n)
    grid = (batch * heads, 1)
    
    # Warmup
    for _ in range(20):
        attn_module.flash_attention_tiled_kernel[grid](
            q_flat, k_flat, v_flat, output,
            1.0 / (head_dim ** 0.5), seq_k, head_dim,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            0, IS_CAUSAL=True,
            NUM_BLOCK_N=num_blocks, BLOCK_N=block_n, BLOCK_D=head_dim_padded,
            num_warps=4, num_stages=2,
        )
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(n_runs):
        attn_module.flash_attention_tiled_kernel[grid](
            q_flat, k_flat, v_flat, output,
            1.0 / (head_dim ** 0.5), seq_k, head_dim,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            0, IS_CAUSAL=True,
            NUM_BLOCK_N=num_blocks, BLOCK_N=block_n, BLOCK_D=head_dim_padded,
            num_warps=4, num_stages=2,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000

print("FlashAttention BLOCK_N Tuning")
print(f"{'seq_k':<8} {'BLOCK_N':<10} {'Tiles':<8} {'Time (ms)':<12}")
print("-" * 40)

for seq_k in [71, 128, 256, 512]:
    for block_n in [32, 64, 128]:
        num_tiles = triton.cdiv(seq_k, block_n)
        try:
            t = bench_flash(seq_k, block_n)
            print(f"{seq_k:<8} {block_n:<10} {num_tiles:<8} {t:.4f}")
        except Exception as e:
            print(f"{seq_k:<8} {block_n:<10} {num_tiles:<8} FAILED: {str(e)[:30]}")