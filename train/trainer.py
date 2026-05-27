"""
Training loop for the Korean LLM with Gradient Accumulation.
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


def train_step(model, batch, device):
    """Executes a single forward and backward pass, returning normalized loss."""
    inputs = batch[:, :-1].to(device)
    targets = batch[:, 1:].to(device)
    
    use_amp = device.type == "cuda"
    with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
        logits, loss = model(inputs, targets=targets)
        
    # Normalize loss for accumulation
    loss = loss / config.GRAD_ACC_STEPS
    loss.backward()
    
    return loss.item() * config.GRAD_ACC_STEPS


def run_training(model, dataset, start_step=0):
    """
    Main training loop with Gradient Accumulation.
    """
    device = get_device()
    print(f"🚀 Training device: {str(device).upper()}")
    if device.type == "cuda":
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=config.MAX_STEPS
    )
    
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, pin_memory=True)
    data_iter = iter(dataloader)
    
    print(f"📊 Effective Batch Size: {config.BATCH_SIZE * config.GRAD_ACC_STEPS}")
    
    for step in range(start_step, config.MAX_STEPS):
        accum_loss = 0.0
        optimizer.zero_grad()
        
        # Accumulate gradients over multiple micro-batches
        for _ in range(config.GRAD_ACC_STEPS):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            accum_loss += train_step(model, batch, device)
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        
        avg_loss = accum_loss / config.GRAD_ACC_STEPS
        
        if step % config.LOG_EVERY == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step}: Loss = {avg_loss:.4f}, LR = {current_lr:.2e}")
            
        if step > 0 and step % config.SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, step, config.CHECKPOINT_DIR)
            
    print("Training complete.")
