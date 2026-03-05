
# =============================================================
# src/trainer.py
# ClinicalShift — Contrastive Training Loop
# =============================================================
#
# TRAINING STRATEGY:
#   - Optimizer : Adam (lr=3e-4) — standard for deep learning
#   - Scheduler : CosineAnnealingLR — smoothly reduces lr
#   - Epochs    : 50
#   - Batch size: 256
#
# COSINE ANNEALING SCHEDULE:
#   lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi*t/T))
#   Smoothly reduces learning rate — better than step decay
#
# TRAINING ONLY ON IN-DISTRIBUTION DATA (young patients)
#   The encoder learns what "normal" ECG patterns look like.
#   Out-of-distribution data is NEVER seen during training.
#   This is critical — shift detection only works if the
#   model has never seen the shifted data.
# =============================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import time


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Run one full pass through the training data.

    Args:
        model     : ECGEncoder
        loader    : DataLoader for in-distribution data
        optimizer : Adam optimizer
        loss_fn   : NTXentLoss
        device    : cuda or cpu

    Returns:
        avg_loss : average loss across all batches
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch_x, _ in loader:
        # batch_x shape: (batch, 1, 250)
        batch_x = batch_x.to(device)

        # ---- Create two augmented views ----
        # We augment on CPU then move to GPU
        x_np = batch_x.cpu().numpy()

        aug1_list = []
        aug2_list = []

        for i in range(len(x_np)):
            window = x_np[i, 0]   # shape (250,)
            from augmentations import augment
            a1 = augment(window.copy())
            a2 = augment(window.copy())
            aug1_list.append(a1)
            aug2_list.append(a2)

        # Stack and move to device
        aug1 = torch.tensor(
            np.array(aug1_list), dtype=torch.float32
        ).unsqueeze(1).to(device)   # (batch, 1, 250)

        aug2 = torch.tensor(
            np.array(aug2_list), dtype=torch.float32
        ).unsqueeze(1).to(device)

        # ---- Forward pass ----
        z_i = model(aug1)   # embeddings from view 1
        z_j = model(aug2)   # embeddings from view 2

        # ---- Compute NT-Xent loss ----
        loss = loss_fn(z_i, z_j)

        # ---- Backward pass ----
        optimizer.zero_grad()   # clear old gradients
        loss.backward()         # compute new gradients
        optimizer.step()        # update weights

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


def train(model, dataset, loss_fn, device,
          epochs=50, batch_size=256, lr=3e-4,
          save_path="/content/ClinicalShift/checkpoints"):
    """
    Full training loop for contrastive learning.

    Args:
        model      : ECGEncoder
        dataset    : ECGDataset (in-distribution only)
        loss_fn    : NTXentLoss
        device     : cuda or cpu
        epochs     : number of training epochs
        batch_size : samples per batch
        lr         : learning rate
        save_path  : where to save model checkpoints

    Returns:
        loss_history : list of average loss per epoch
    """
    os.makedirs(save_path, exist_ok=True)

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Optimizer — Adam with weight decay for regularization
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    # Learning rate scheduler — cosine annealing
    # Smoothly reduces lr from lr_max to lr_min over training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    loss_history = []
    best_loss    = float('inf')

    print("=" * 55)
    print("  STARTING CONTRASTIVE TRAINING")
    print("=" * 55)
    print(f"  Device     : {device}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {batch_size}")
    print(f"  LR         : {lr}")
    print(f"  Dataset    : {len(dataset):,} windows")
    print("=" * 55)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        avg_loss = train_one_epoch(
            model, loader, optimizer, loss_fn, device
        )

        scheduler.step()
        loss_history.append(avg_loss)

        elapsed = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]

        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {elapsed:.1f}s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(),
                os.path.join(save_path, "best_encoder.pth")
            )

    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join(save_path, "final_encoder.pth")
    )

    print("=" * 55)
    print(f"  Training complete!")
    print(f"  Best loss  : {best_loss:.4f}")
    print(f"  Saved to   : {save_path}/best_encoder.pth")
    print("=" * 55)

    return loss_history
