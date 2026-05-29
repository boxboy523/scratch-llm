"""
Checkpoint saving and loading utilities.
"""

import torch
import os
from typing import Optional


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, path: str):
    """
    Saves a training checkpoint.
    Inputs: model, optimizer, step (int), path (str)
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }
    
    filename = f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, os.path.join(path, filename))
    torch.save(checkpoint, os.path.join(path, "latest.pt"))


def load_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], path: str) -> int:
    """
    Loads a training checkpoint. Optimizer is optional.
    Inputs: model, optimizer (optional), path (str)
    Outputs: step (int)
    """
    if not os.path.exists(path):
        return 0
        
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint.get("step", 0)
