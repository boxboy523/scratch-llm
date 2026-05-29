"""
Training loop with WSD (Warmup-Stable-Decay) schedule and Gradient Accumulation.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from train.checkpoint import save_checkpoint


def get_wsd_scheduler(optimizer, warmup_steps: int):
    """
    Creates a Warmup then Stable learning rate scheduler.
    Inputs: optimizer, warmup_steps (int)
    Outputs: LambdaLR scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_device() -> torch.device:
    """Returns the available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_step(model, batch, device):
    """
    Executes a single forward and backward pass.
    Inputs: model, batch, device
    Outputs: micro-batch loss (float)
    """
    inputs = batch[:, :-1].to(device)
    targets = batch[:, 1:].to(device)
    
    use_amp = device.type == "cuda"
    with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
        _, loss = model(inputs, targets=targets)
        
    # Normalize loss for accumulation
    (loss / config.GRAD_ACC_STEPS).backward()
    
    return loss.item()


def run_training(model, dataset, start_step=0):
    """
    Main training loop with Gradient Accumulation and WSD Phase 1.
    Inputs: model, dataset, start_step (int)
    """
    device = get_device()
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR)
    scheduler = get_wsd_scheduler(optimizer, config.WARMUP_STEPS)
    
    # Set to start_step in case of resumption
    for _ in range(start_step):
        scheduler.step()

    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, pin_memory=True)
    data_iter = iter(dataloader)
    
    for step in range(start_step, config.MAX_STEPS):
        optimizer.zero_grad()
        accum_loss = 0.0
        
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
        
        if step % config.LOG_EVERY == 0:
            avg_loss = accum_loss / config.GRAD_ACC_STEPS
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step}: Loss = {avg_loss:.4f}, LR = {current_lr:.2e}")
            
        if step > 0 and step % config.SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, step, config.CHECKPOINT_DIR)
