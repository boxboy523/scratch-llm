"""
Training loop for the Korean LLM.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import config
from train.checkpoint import save_checkpoint


def get_device() -> torch.device:
    """Returns the available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_step(model, optimizer, batch, device, grad_clip):
    """Executes a single training step."""
    optimizer.zero_grad()
    inputs = batch[:, :-1].to(device)
    targets = batch[:, 1:].to(device)
    
    # Mixed precision context
    use_amp = device.type == "cuda"
    with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
        logits, loss = model(inputs, targets=targets)
        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    
    return loss.item()


def run_training(model, dataset, start_step=0):
    """
    Main training loop.
    """
    device = get_device()
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR)
    
    # Initialize Cosine Scheduler with Warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=config.MAX_STEPS
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        pin_memory=True
    )
    data_iter = iter(dataloader)
    
    for step in range(start_step, config.MAX_STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        loss_val = train_step(model, optimizer, batch, device, config.GRAD_CLIP)
        scheduler.step()
        
        if step % config.LOG_EVERY == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step}: Loss = {loss_val:.4f}, LR = {current_lr:.2e}")
            
        if step > 0 and step % config.SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, step, config.CHECKPOINT_DIR)
            
    print("Training complete.")
