import torch
import numpy as np
from tqdm import tqdm

def train_fn(loader, model, optimizer, loss_fn, scaler=None):
    """
    A simple training loop that iterates over 'loader',
    computes loss, does backward, and steps the optimizer.
    """
    model.train()
    loop = tqdm(loader)
    for batch_idx, (imgs, masks, _) in enumerate(loop):
        imgs  = imgs.float().to(model.device)
        masks = masks.long().to(model.device)

        # Forward pass
        preds = model(imgs)
        loss  = loss_fn(preds, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # (Optional) update tqdm description
        loop.set_postfix(loss=loss.item())

def check_accuracy_batch(inputs, targets, model, batch_size, device="cuda"):
    """
    A quick function to compute pixel‐accuracy and Dice score on a single batch.
    Returns (accuracy, dice) as Python scalars.
    """
    model.eval()
    with torch.no_grad():
        inputs  = inputs.to(device)
        targets = targets.to(device)

        # Get network output
        logits  = model(inputs)                    # shape: (B, C, H, W)
        probs   = torch.softmax(logits, dim=1)     # shape: (B, C, H, W)
        preds   = torch.argmax(probs, dim=1)       # shape: (B, H, W)

        # Pixel accuracy
        correct = (preds == targets).float().sum()
        total   = torch.numel(targets)
        acc     = (correct / total).item()

        # Dice: sum over batch
        # This is a simple two‐class Dice calculation, adjust for multiple classes if needed
        intersection = (preds * targets).sum().item()
        union        = preds.sum().item() + targets.sum().item()
        dice         = (2.0 * intersection) / (union + 1e-8)

    return acc, dice
