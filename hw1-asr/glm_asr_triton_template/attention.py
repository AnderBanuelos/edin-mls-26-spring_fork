"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Implements three optimization levels:
  - Opt 1: num_warps/num_stages tuning on kernel launches
  - Opt 2: Fused scores + causal mask + softmax kernel
  - Opt 3: True FlashAttention with tiled K/V and online softmax (Dao et al., 2022)
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Baseline Triton Kernels (kept for fallback path and comparison)
# ============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr, k_ptr, scores_ptr,
    scale, seq_k, head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_s0, stride_s1, stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute scaled attention scores. Grid: (batch_heads, seq_q)."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim, other=0.0,
    )
    k = tl.load(
        k_ptr + pid_bh * stride_k0 + offs_k[:, None] * stride_k1 + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0,
    )
    scores = tl.sum(k * q[None, :], axis=1) * scale
    tl.store(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        scores, mask=offs_k < seq_k,
    )


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """Softmax along last dimension. Grid: (batch_heads * seq_q,)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_k
    s = tl.load(scores_ptr + row * stride_s + offs, mask=mask, other=-float("inf"))
    s = s - tl.max(s, axis=0)
    exp_s = tl.exp(s)
    out = exp_s / tl.sum(exp_s, axis=0)
    tl.store(scores_ptr + row * stride_s + offs, out, mask=mask)


@triton.jit
def attention_output_kernel(
    attn_ptr, v_ptr, output_ptr,
    seq_k, head_dim,
    stride_w0, stride_w1, stride_w2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute attention output: weights @ V. Grid: (batch_heads, seq_q)."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    w = tl.load(
        attn_ptr + pid_bh * stride_w0 + pid_q * stride_w1 + offs_k * stride_w2,
        mask=offs_k < seq_k, other=0.0,
    )
    v = tl.load(
        v_ptr + pid_bh * stride_v0 + offs_k[:, None] * stride_v1 + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0,
    )
    out = tl.sum(v * w[:, None], axis=0)
    tl.store(
        output_ptr + pid_bh * stride_o0 + pid_q * stride_o1 + offs_d * stride_o2,
        out, mask=offs_d < head_dim,
    )


# ============================================================================
# Optimization 2: Fused scores + causal mask + softmax (kept for reference)
# ============================================================================

@triton.jit
def fused_scores_softmax_causal_kernel(
    q_ptr, k_ptr, scores_ptr,
    scale, seq_k, head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_s0, stride_s1, stride_s2,
    offset,
    IS_CAUSAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused: Q@K^T * scale + optional causal mask + softmax. Grid: (batch_heads, seq_q)."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim, other=0.0,
    )
    k = tl.load(
        k_ptr + pid_bh * stride_k0 + offs_k[:, None] * stride_k1 + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0,
    )
    scores = tl.sum(k * q[None, :], axis=1) * scale
    mask_k = offs_k < seq_k
    scores = tl.where(mask_k, scores, -float("inf"))
    if IS_CAUSAL:
        current_pos = pid_q + offset
        scores = tl.where(offs_k > current_pos, -float("inf"), scores)
    scores = scores - tl.max(scores, axis=0)
    exp_s = tl.exp(scores)
    softmax_out = exp_s / tl.sum(exp_s, axis=0)
    tl.store(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        softmax_out, mask=mask_k,
    )


# ============================================================================
# Optimization 3: True FlashAttention with Tiled K/V and Online Softmax
#
# Core idea from Dao et al., "FlashAttention: Fast and Memory-Efficient
# Exact Attention with IO-Awareness" (NeurIPS 2022):
#
#   Instead of loading all keys/values at once (which requires O(seq^2)
#   register space and limits max sequence length), we tile along the
#   K/V sequence dimension in blocks of BLOCK_N. Each iteration:
#     - Loads a block of K (BLOCK_N keys)
#     - Computes partial scores for that block
#     - Updates a running softmax using the "online softmax" algorithm
#     - Loads the corresponding block of V
#     - Accumulates the weighted output
#
#   Online softmax maintains three running variables:
#     m_i: running maximum score seen so far
#     l_i: running sum of exp(scores - m_i)
#     o_i: running weighted output accumulator
#
#   When a new block produces a new maximum m_new > m_i, previous
#   results are rescaled by exp(m_i - m_new) to maintain correctness.
#   This is mathematically equivalent to standard softmax but never
#   requires all scores to be in memory simultaneously.
#
# Memory traffic: Read Q (once) + stream K,V (block by block) + Write O (once)
# No intermediate tensors (scores, attention weights) in global memory.
# Memory complexity: O(seq) instead of O(seq^2) for the attention matrix.
# ============================================================================

@triton.jit
def flash_attention_tiled_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    scale,
    seq_k,          # actual number of key positions (for masking)
    head_dim,       # actual head dimension (for masking)
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    offset,         # position offset for autoregressive causal mask
    IS_CAUSAL: tl.constexpr,
    NUM_BLOCK_N: tl.constexpr,  # number of K/V blocks = ceil(seq_k_padded / BLOCK_N)
    BLOCK_N: tl.constexpr,      # tile size along K/V sequence dimension
    BLOCK_D: tl.constexpr,      # head dimension (padded to power of 2)
):
    """
    True FlashAttention: tiled K/V with online softmax.

    Grid: (batch_heads, seq_q)
    Each program instance processes one query position and streams
    through all K/V blocks, maintaining online softmax state.
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    # ── (1) Load query vector ─────────────────────────────────────────
    # Q is loaded once and stays in registers for the entire kernel.
    # This is the key insight: Q is reused across all K/V blocks.
    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=mask_d, other=0.0,
    )

    # ── (2) Initialize online softmax state ───────────────────────────
    # m_i: running max (initialized to large negative so first block dominates)
    # l_i: running sum of exp(scores - m_i)
    # o_i: running output accumulator (head_dim vector)
    m_i = tl.full([], -1e20, dtype=tl.float32)
    l_i = tl.full([], 0.0, dtype=tl.float32)
    o_i = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Precompute causal position
    if IS_CAUSAL:
        current_pos = pid_q + offset

    # ── (3) Stream through K/V blocks ─────────────────────────────────
    for block_idx in range(NUM_BLOCK_N):
        block_start = block_idx * BLOCK_N
        offs_n = block_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_k

        # ── Causal early-skip ─────────────────────────────────────
        # If the entire block is beyond the causal boundary,
        # all scores would be -inf and contribute nothing.
        # We still process it (Triton doesn't support break), but
        # the masking ensures zero contribution.
        # For non-causal attention, all blocks are processed.

        # ── (3a) Load K block: (BLOCK_N, BLOCK_D) ────────────────
        k_block = tl.load(
            k_ptr
            + pid_bh * stride_k0
            + offs_n[:, None] * stride_k1
            + offs_d[None, :] * stride_k2,
            mask=(mask_n[:, None]) & (mask_d[None, :]),
            other=0.0,
        )

        # ── (3b) Compute partial scores: q @ k_block^T ───────────
        # Result shape: (BLOCK_N,) — one score per key in this block
        s_j = tl.sum(k_block * q[None, :], axis=1) * scale

        # ── (3c) Mask out-of-bounds positions (padding) ───────────
        s_j = tl.where(mask_n, s_j, -float("inf"))

        # ── (3d) Apply causal mask for this block ─────────────────
        # Keys at positions beyond current_pos get -inf score
        if IS_CAUSAL:
            s_j = tl.where(offs_n > current_pos, -float("inf"), s_j)

        # ── (3e) Online softmax update ────────────────────────────
        # This is the core of FlashAttention. We maintain a running
        # max and sum, rescaling previous results when a new block
        # produces a larger maximum.
        #
        # Algorithm (Milakov & Gimelshein, 2018; Dao et al., 2022):
        #   m_new = max(m_i, max(s_j))
        #   alpha = exp(m_i - m_new)      ← rescaling factor
        #   l_i   = l_i * alpha + sum(exp(s_j - m_new))
        #   o_i   = o_i * alpha + exp(s_j - m_new) @ V_block
        #   m_i   = m_new

        # New maximum across running state and current block
        m_block = tl.max(s_j, axis=0)
        m_new = tl.where(m_block > m_i, m_block, m_i)

        # Rescale previous accumulator to account for new maximum
        # If m_i was -1e20 (first iteration), alpha ≈ 0, which correctly
        # zeroes out the initial state.
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha
        o_i = o_i * alpha

        # Compute attention weights for current block
        p_j = tl.exp(s_j - m_new)

        # Update running sum
        l_i = l_i + tl.sum(p_j, axis=0)

        # ── (3f) Load V block and accumulate output ───────────────
        v_block = tl.load(
            v_ptr
            + pid_bh * stride_v0
            + offs_n[:, None] * stride_v1
            + offs_d[None, :] * stride_v2,
            mask=(mask_n[:, None]) & (mask_d[None, :]),
            other=0.0,
        )

        # Weighted sum: p_j (BLOCK_N,) broadcast × v_block (BLOCK_N, BLOCK_D)
        # then reduce along BLOCK_N → (BLOCK_D,)
        o_i = o_i + tl.sum(v_block * p_j[:, None], axis=0)

        # Update running max
        m_i = m_new

    # ── (4) Final normalization ───────────────────────────────────────
    # Divide accumulated output by the total softmax denominator.
    # This completes the softmax: output = sum(exp(s-m)*v) / sum(exp(s-m))
    o_i = o_i / l_i

    # ── (5) Store output ──────────────────────────────────────────────
    # Single write to global memory — the only VRAM write in the kernel.
    tl.store(
        output_ptr + pid_bh * stride_o0 + pid_q * stride_o1 + offs_d * stride_o2,
        o_i, mask=mask_d,
    )


@triton.jit
def causal_mask_kernel(
    scores_ptr, seq_k, offset,
    stride_s0, stride_s1, stride_s2,
    BLOCK_K: tl.constexpr,
):
    """Apply causal mask to attention scores. Grid: (batch_heads, seq_q)."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        mask=mask, other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        scores, mask=mask,
    )


# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


# Increased from 256: tiled FlashAttention no longer needs the entire
# sequence to fit in one tile, so we can support longer sequences.
MAX_ATTENTION_DIM = 512

# Tile size for K/V sequence dimension in FlashAttention.
# Must be a power of 2. 64 balances register usage vs loop overhead.
FLASH_BLOCK_N = 128


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention using Triton kernels.

    Execution paths:
      1. FlashAttention (no explicit mask): single tiled kernel with online softmax.
         Q loaded once, K/V streamed in blocks of FLASH_BLOCK_N.
         No intermediate VRAM tensors. O(seq) memory instead of O(seq^2).
      2. Fallback (explicit attention_mask): separate kernels, scores tensor in VRAM.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    head_dim_padded = next_power_of_two(head_dim)

    use_triton = (
        q.is_cuda
        and head_dim_padded <= MAX_ATTENTION_DIM
    )

    if use_triton:
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32)
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)

        # Pad head_dim to power of 2 (required for tl.arange)
        if head_dim_padded != head_dim:
            q_padded = torch.zeros(
                (batch * num_heads, seq_q, head_dim_padded),
                dtype=torch.float32, device=q.device,
            )
            k_padded = torch.zeros(
                (batch * num_heads, seq_k, head_dim_padded),
                dtype=torch.float32, device=q.device,
            )
            v_padded = torch.zeros_like(k_padded)
            q_padded[:, :, :head_dim] = q_flat
            k_padded[:, :, :head_dim] = k_flat
            v_padded[:, :, :head_dim] = v_flat
            q_flat = q_padded
            k_flat = k_padded
            v_flat = v_padded

        output = torch.empty(
            (batch * num_heads, seq_q, head_dim_padded),
            dtype=torch.float32, device=q.device,
        )

        grid = (batch * num_heads, seq_q)
        has_explicit_mask = attention_mask is not None

        if not has_explicit_mask:
            # ============================================================
            # FLASH ATTENTION PATH (Optimization 3)
            #
            # True FlashAttention with tiled K/V and online softmax.
            # Q is loaded once per query position. K and V are streamed
            # in blocks of FLASH_BLOCK_N along the sequence dimension.
            # Online softmax maintains running max and sum across blocks,
            # rescaling the output accumulator when a new block changes
            # the maximum. No intermediate tensors in VRAM.
            #
            # NUM_BLOCK_N is a constexpr so Triton can compile the loop
            # efficiently. Different seq_k values trigger recompilation.
            # ============================================================
            num_blocks = triton.cdiv(seq_k, FLASH_BLOCK_N)

            flash_attention_tiled_kernel[grid](
                q_flat, k_flat, v_flat, output,
                float(scale),
                seq_k,
                head_dim,
                q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
                k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
                v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
                output.stride(0), output.stride(1), output.stride(2),
                0,  # offset for autoregressive decoding
                IS_CAUSAL=is_causal,
                NUM_BLOCK_N=num_blocks,
                BLOCK_N=FLASH_BLOCK_N,
                BLOCK_D=head_dim_padded,
                num_warps=4,
                num_stages=2,
            )
        else:
            # ============================================================
            # FALLBACK PATH: explicit attention_mask needs separate kernels
            # ============================================================
            seq_k_padded = next_power_of_two(seq_k)
            scores = torch.empty(
                (batch * num_heads, seq_q, seq_k_padded),
                dtype=torch.float32, device=q.device,
            )

            # Pad K for fallback path (needs power-of-2 seq_k)
            if seq_k_padded != seq_k:
                k_fb = torch.zeros(
                    (batch * num_heads, seq_k_padded, head_dim_padded),
                    dtype=torch.float32, device=q.device,
                )
                k_fb[:, :seq_k, :head_dim] = k_flat[:, :, :head_dim] if head_dim_padded != head_dim else k_flat
                v_fb = torch.zeros_like(k_fb)
                v_fb[:, :seq_k, :head_dim] = v_flat[:, :, :head_dim] if head_dim_padded != head_dim else v_flat
            else:
                k_fb = k_flat
                v_fb = v_flat

            attention_scores_kernel[grid](
                q_flat, k_fb, scores,
                float(scale), seq_k_padded, head_dim_padded,
                q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
                k_fb.stride(0), k_fb.stride(1), k_fb.stride(2),
                scores.stride(0), scores.stride(1), scores.stride(2),
                BLOCK_K=seq_k_padded,
                BLOCK_D=head_dim_padded,
                num_warps=8, num_stages=2,
            )

            if seq_k_padded != seq_k:
                scores[:, :, seq_k:] = -1e9

            if is_causal:
                mask = torch.triu(
                    torch.ones((seq_q, seq_k_padded), dtype=torch.float32, device=q.device),
                    diagonal=1,
                ) * -1e9
                scores = scores + mask[None, :, :]

            if attention_mask.ndim == 4:
                attention_mask = attention_mask.reshape(batch * num_heads, seq_q, seq_k)
            if seq_k_padded != seq_k:
                mask_padded = torch.zeros(
                    (batch * num_heads, seq_q, seq_k_padded),
                    dtype=torch.float32, device=q.device,
                )
                mask_padded[:, :, :seq_k] = attention_mask
                mask_padded[:, :, seq_k:] = -1e9
                attention_mask = mask_padded
            scores = scores + attention_mask

            scores_2d = scores.reshape(batch * num_heads * seq_q, seq_k_padded)
            softmax_inplace_kernel[(scores_2d.shape[0],)](
                scores_2d, scores_2d.stride(0), seq_k_padded, BLOCK_SIZE=seq_k_padded,
                num_warps=2, num_stages=2,
            )
            scores = scores_2d.reshape(batch * num_heads, seq_q, seq_k_padded)

            attention_output_kernel[grid](
                scores, v_fb, output,
                seq_k_padded, head_dim_padded,
                scores.stride(0), scores.stride(1), scores.stride(2),
                v_fb.stride(0), v_fb.stride(1), v_fb.stride(2),
                output.stride(0), output.stride(1), output.stride(2),
                BLOCK_K=seq_k_padded, BLOCK_D=head_dim_padded,
                num_warps=4, num_stages=3,
            )

        if head_dim_padded != head_dim:
            output = output[:, :, :head_dim]

        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    # CPU/unsupported fallback using PyTorch
    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale

    if is_causal:
        mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device),
            diagonal=1,
        ) * -1e9
        scores = scores + mask[None, None, :, :]

    if attention_mask is not None:
        scores = scores + attention_mask

    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)

    return output.to(q.dtype)


if __name__ == "__main__":
    print("Testing Triton Attention with FlashAttention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention (FlashAttention path):")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    # Verify against PyTorch reference
    scores_ref = torch.einsum("bnqd,bnkd->bnqk", q.float(), k.float()) / (head_dim ** 0.5)
    attn_ref = torch.softmax(scores_ref, dim=-1)
    output_ref = torch.einsum("bnqk,bnkd->bnqd", attn_ref, v.float())
    max_diff = (output.float() - output_ref).abs().max().item()
    print(f"  Max diff vs PyTorch reference: {max_diff:.6f}")
    assert max_diff < 1e-3, f"FlashAttention output differs from reference by {max_diff}"

    print("\nCausal attention (FlashAttention path):")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    # Verify causal against reference
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) * -1e9
    scores_ref_c = scores_ref + causal_mask[None, None, :, :]
    attn_ref_c = torch.softmax(scores_ref_c, dim=-1)
    output_ref_c = torch.einsum("bnqk,bnkd->bnqd", attn_ref_c, v.float())
    max_diff_c = (output_causal.float() - output_ref_c).abs().max().item()
    print(f"  Max diff vs PyTorch causal reference: {max_diff_c:.6f}")
    assert max_diff_c < 1e-3, f"Causal FlashAttention differs from reference by {max_diff_c}"

    print("\nWith attention mask (fallback path):")
    mask = torch.zeros(
        (batch_size, num_heads, seq_len, seq_len), dtype=torch.float32, device=device
    )
    mask[:, :, :, seq_len // 2 :] = -1e9
    output_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    print(f"  Output shape: {output_masked.shape}")

    print("\nGrouped Query Attention (GQA):")
    num_kv_heads = 2
    k_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    v_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    attn = MultiHeadAttention(
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    output_gqa = attn(q, k_gqa, v_gqa)
    print(f"  Output shape: {output_gqa.shape}")

    # Test with larger sequence to exercise multiple tiles
    print("\nLarger sequence (multiple K/V tiles):")
    seq_long = 256
    q_long = torch.randn(1, 4, seq_long, head_dim, device=device)
    k_long = torch.randn(1, 4, seq_long, head_dim, device=device)
    v_long = torch.randn(1, 4, seq_long, head_dim, device=device)
    output_long = scaled_dot_product_attention(q_long, k_long, v_long, is_causal=True)
    print(f"  Output shape: {output_long.shape}")

    # Verify long sequence
    scores_long = torch.einsum("bnqd,bnkd->bnqk", q_long.float(), k_long.float()) / (head_dim ** 0.5)
    causal_long = torch.triu(torch.ones(seq_long, seq_long, device=device), diagonal=1) * -1e9
    scores_long = scores_long + causal_long[None, None, :, :]
    attn_long = torch.softmax(scores_long, dim=-1)
    output_long_ref = torch.einsum("bnqk,bnkd->bnqd", attn_long, v_long.float())
    max_diff_long = (output_long.float() - output_long_ref).abs().max().item()
    print(f"  Max diff vs reference (seq={seq_long}): {max_diff_long:.6f}")
    assert max_diff_long < 1e-2, f"Long sequence FlashAttention differs by {max_diff_long}"

    print("\nAll FlashAttention tests passed!")