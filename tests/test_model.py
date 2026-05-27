"""
Unit tests for the model architecture.
"""

import torch
import pytest
import config
from model.lm import KoLLM, get_parameter_count


def test_model_parameter_count():
    """Verifies that the model has approximately 100M parameters."""
    model = KoLLM(
        vocab_size=config.VOCAB_SIZE,
        context_len=config.CONTEXT_LEN,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_kv_heads=config.N_KV_HEADS,
        n_layers=config.N_LAYERS,
        d_ffn=config.D_FFN
    )
    n_params = get_parameter_count(model)
    print(f"Model parameters: {n_params:,}")
    
    # Target range: [90M, 110M]
    assert 90_000_000 <= n_params <= 110_000_000


def test_model_forward_pass():
    """Verifies the forward pass of the model on CPU."""
    model = KoLLM(
        vocab_size=1000,  # Small vocab for testing
        context_len=128,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ffn=256
    )
    model.eval()
    
    batch_size = 2
    seq_len = 32
    tokens = torch.randint(0, 1000, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, loss = model(tokens)
        
    assert logits.shape == (batch_size, seq_len, 1000)
    assert loss is None


def test_model_loss_calculation():
    """Verifies that the model calculates loss when targets are provided."""
    model = KoLLM(
        vocab_size=1000,
        context_len=128,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ffn=256
    )
    
    batch_size = 2
    seq_len = 32
    tokens = torch.randint(0, 1000, (batch_size, seq_len))
    targets = torch.randint(0, 1000, (batch_size, seq_len))
    
    logits, loss = model(tokens, targets=targets)
    
    assert logits.shape == (batch_size, seq_len, 1000)
    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
