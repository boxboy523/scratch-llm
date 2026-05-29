"""
Transformer Block with RMSNorm and SwiGLU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import RoPEMultiHeadAttention


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the RMSNorm normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.
        
        Args:
            x: Input tensor with shape (batch, seq_len, dim).
            
        Returns:
            Normalized tensor with shape (batch, seq_len, dim).
        """
        return self._norm(x.float()).type_as(x) * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function / MLP.
    """

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        # SwiGLU: W3 * (silu(W1 * x) * (W2 * x))
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_model, d_ffn, bias=False)
        self.w3 = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SwiGLU MLP.
        
        Args:
            x: Input tensor with shape (batch, seq_len, d_model).
            
        Returns:
            Output tensor with shape (batch, seq_len, d_model).
        """
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))


class TransformerBlock(nn.Module):
    """
    Transformer Block (Decoder Layer) with GQA support.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.attention = RoPEMultiHeadAttention(d_model, n_heads, n_kv_heads, dropout)
        self.feed_forward = SwiGLU(d_model, d_ffn, dropout)
        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer Block.
        
        Args:
            x: Input tensor with shape (batch, seq_len, d_model).
            freqs_cis: RoPE frequency tensor with shape (seq_len, head_dim // 2).
            
        Returns:
            Output tensor with shape (batch, seq_len, d_model).
        """
        # Pre-norm architecture
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
