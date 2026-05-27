"""
Full Decoder-only Language Model.
"""

import torch
import torch.nn as nn
from model.block import TransformerBlock, RMSNorm
from model.attention import precompute_freqs_cis


class KoLLM(nn.Module):
    """
    Korean 100M parameter LLM (Decoder-only).
    """

    def __init__(
        self, 
        vocab_size: int, 
        context_len: int, 
        d_model: int, 
        n_heads: int, 
        n_kv_heads: int,
        n_layers: int, 
        d_ffn: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.context_len = context_len
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight Tying: Shared weights between embedding and output
        self.output.weight = self.tok_embeddings.weight
        
        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(d_model // n_heads, context_len)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the LLM.

        Args:
            tokens: Input token IDs with shape (batch, seq_len).
            targets: Optional target token IDs with shape (batch, seq_len).

        Returns:
            logits: Predicted logits with shape (batch, seq_len, vocab_size).
            loss: Scalar loss tensor if targets provided, else None.
        """
        _, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        
        # Ensure freqs_cis is on the same device as h
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seq_len]
        
        # Causal mask for SDPA (boolean mask)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h.device), diagonal=1).bool()
        
        for layer in self.layers:
            # Note: attn_mask=True means positions to be masked out
            h = layer(h, freqs_cis, mask)
            
        h = self.norm(h)
        logits = self.output(h)
        
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.reshape(-1),
                ignore_index=-1
            )
            
        return logits, loss


def get_parameter_count(model: nn.Module) -> int:
    """Calculates the total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())
