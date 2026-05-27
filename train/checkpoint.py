"""
Checkpoint saving and loading utilities.
"""

import torch
import os


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, path: str):
    """
    Saves a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        step: The current training step.
        path: Directory to save the checkpoint.
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
    # Update 'latest' symbolic link or indicator if needed
    torch.save(checkpoint, os.path.join(path, "latest.pt"))


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str) -> int:
    """
    Loads a training checkpoint.

    Args:
        model: The model to load weights into.
        optimizer: The optimizer to load state into.
        path: Path to the checkpoint file (.pt).

    Returns:
        The step at which the checkpoint was saved.
    """
    if not os.path.exists(path):
        return 0
        
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint.get("step", 0)
