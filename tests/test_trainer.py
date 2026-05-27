"""
Unit tests for the training loop and checkpointing.
"""

import torch
import os
import pytest
from model.lm import KoLLM
from train.trainer import train_step
from train.checkpoint import save_checkpoint, load_checkpoint


def test_train_step():
    """Verifies that a single training step runs without error on CPU."""
    model = KoLLM(
        vocab_size=100,
        context_len=16,
        d_model=32,
        n_heads=2,
        n_kv_heads=1,
        n_layers=1,
        d_ffn=64
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = torch.randint(0, 100, (2, 17))  # seq_len + 1 for input/target shift
    
    device = torch.device("cpu")
    loss = train_step(model, optimizer, batch, device, grad_clip=1.0)
    
    assert isinstance(loss, float)
    assert loss > 0


def test_checkpoint_save_load(tmp_path):
    """Verifies that checkpoints can be saved and loaded."""
    model = KoLLM(
        vocab_size=100,
        context_len=16,
        d_model=32,
        n_heads=2,
        n_kv_heads=1,
        n_layers=1,
        d_ffn=64
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    checkpoint_dir = tmp_path / "checkpoints"
    save_checkpoint(model, optimizer, step=10, path=str(checkpoint_dir))
    
    assert os.path.exists(checkpoint_dir / "checkpoint_step_10.pt")
    assert os.path.exists(checkpoint_dir / "latest.pt")
    
    # Load back into a new model
    model_new = KoLLM(
        vocab_size=100,
        context_len=16,
        d_model=32,
        n_heads=2,
        n_kv_heads=1,
        n_layers=1,
        d_ffn=64
    )
    optimizer_new = torch.optim.AdamW(model_new.parameters(), lr=1e-3)
    
    step = load_checkpoint(model_new, optimizer_new, str(checkpoint_dir / "latest.pt"))
    
    assert step == 10
    # Check if parameters match (basic check)
    for p1, p2 in zip(model.parameters(), model_new.parameters()):
        assert torch.equal(p1, p2)
