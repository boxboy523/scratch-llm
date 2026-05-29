"""
Grouped-Query Attention (GQA) with Rotary Position Embeddings (RoPE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precomputes the frequency complex exponentials for RoPE.

    Args:
        dim: The dimension of the embeddings (head_dim).
        end: The maximum sequence length (context_len).
        theta: The base value for the frequency scale.

    Returns:
        freqs_cis: Complex exponential tensor with shape (end, dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies RoPE to query and key tensors.

    Args:
        xq: Query tensor with shape (batch, seq_len, n_heads, head_dim).
        xk: Key tensor with shape (batch, seq_len, n_kv_heads, head_dim).
        freqs_cis: Precomputed frequencies with shape (seq_len, head_dim // 2).

    Returns:
        xq_out, xk_out: Tensors with RoPE applied, same shape as inputs.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs_cis for broadcasting: (1, seq_len, 1, head_dim // 2)
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats Key/Value heads to match Query heads in GQA.
    (batch, seq_len, n_kv_heads, head_dim) -> (batch, seq_len, n_heads, head_dim)
    """
    if n_rep == 1:
        return x
    batch, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
    )


class RoPEMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with GQA and RoPE support.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Multi-Head Attention (GQA).

        Args:
            x: Input tensor with shape (batch, seq_len, d_model).
            freqs_cis: RoPE frequency tensor with shape (seq_len, head_dim // 2).

        Returns:
            Output tensor with shape (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        xq = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Repeat KV heads for GQA
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Use memory-efficient scaled_dot_product_attention (Flash Attention)
        # It handles scaling, masking, and dropout internally with much lower VRAM
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.wo(output)
