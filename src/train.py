"""
Training Pipeline for 15-Layer Skin Condition CNN.

Features:
  - Hardware Acceleration: Automatically uses Apple Silicon GPU (MPS), NVIDIA CUDA, or CPU.
  - Medical Metrics: Evaluates Top-1 accuracy, Top-3 accuracy, and Loss.
  - Class Imbalance Handling: Integrates inverse-frequency class weights in CrossEntropyLoss.
  - Regularization & Schedulers: AdamW optimizer with Cosine Annealing or ReduceLROnPlateau.
  - Model Checkpointing: Saves the model state with the best validation performance.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in sys.path so 'src.' imports work when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        """Lightweight fallback when tqdm is not installed."""
        def __init__(self, iterable, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            for item in self.iterable:
                yield item
        def __len__(self):
            return len(self.iterable) if hasattr(self.iterable, "__len__") else 0
        def set_postfix(self, *args, **kwargs):
            pass

from src.dataset import get_dataloaders
from src.model import SkinConditionCNN


def get_device() -> torch.device:
    """Select the best available compute device (MPS for Apple Silicon, CUDA for NVIDIA, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (Metal Performance Shaders - MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def compute_topk_accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 3)) -> Dict[int, float]:
    """
    Computes Top-K accuracy for specified values of K.
    Particularly useful in dermatology for measuring differential diagnosis quality (e.g. Top-3).
    """
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res[k] = (correct_k.item() / batch_size) * 100.0
    return res


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping to stabilize training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * batch_size
        topk = compute_topk_accuracy(outputs, targets, topk=(1, 3))
        total_top1 += (topk[1] / 100.0) * batch_size
        total_top3 += (topk[3] / 100.0) * batch_size
        total_samples += batch_size

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "top1": f"{total_top1 / total_samples * 100:.1f}%"
        })

    epoch_loss = running_loss / total_samples
    epoch_top1 = (total_top1 / total_samples) * 100.0
    epoch_top3 = (total_top3 / total_samples) * 100.0
    return epoch_loss, epoch_top1, epoch_top3


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """Runs evaluation over the validation or test loader."""
    model.eval()
    running_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            topk = compute_topk_accuracy(outputs, targets, topk=(1, 3))
            total_top1 += (topk[1] / 100.0) * batch_size
            total_top3 += (topk[3] / 100.0) * batch_size
            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_top1 = (total_top1 / total_samples) * 100.0
    epoch_top3 = (total_top3 / total_samples) * 100.0
    return epoch_loss, epoch_top1, epoch_top3


def train(args):
    """Main training orchestrator."""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = get_device()

    print("\nLoading dataset...")
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        manifest_dir=args.manifest_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    if train_loader is None:
        print(f"\n[Error] Training data not found.")
        print(f"Option 1 Workflow expects:")
        print(f"  1. Images placed in: {os.path.abspath(args.data_dir)}/<Class Name>/<image.jpg>")
        print(f"  2. CSV splits generated in: {os.path.abspath(args.manifest_dir)}/train.csv")
        print(f"\nIf you have 'balanced_dataset.csv', generate the splits by running:")
        print(f"    .venv/bin/python preprocess.py\n")
        return

    class_names = loaders["class_names"]
    num_classes = len(class_names)
    num_samples = len(train_loader.dataset)
    print(f"Loaded {num_samples} training samples across {num_classes} classes from '{args.manifest_dir}/train.csv'.")
    print(f"Image source directory: {os.path.abspath(args.data_dir)}")

    model = SkinConditionCNN(num_classes=num_classes, dropout_rate=args.dropout).to(device)

    # Class-weighted CrossEntropyLoss for imbalance
    class_weights = loaders.get("class_weights")
    if class_weights is not None and args.use_class_weights:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using inverse-frequency class weights in loss function.")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.scheduler_patience)

    best_val_loss = float("inf")
    best_top1_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_top1": [], "val_loss": [], "val_top1": [], "val_top3": []}

    print(f"\nStarting training for {args.epochs} epochs...\n" + "=" * 65)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_top1, tr_top3 = train_one_epoch(model, train_loader, criterion, optimizer, device)

        if val_loader is not None:
            val_loss, val_top1, val_top3 = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            history["train_loss"].append(tr_loss)
            history["train_top1"].append(tr_top1)
            history["val_loss"].append(val_loss)
            history["val_top1"].append(val_top1)
            history["val_top3"].append(val_top3)

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] (lr: {current_lr:.1e}) | "
                f"Train Loss: {tr_loss:.4f} - Top-1: {tr_top1:.1f}% | "
                f"Val Loss: {val_loss:.4f} - Top-1: {val_top1:.1f}% - Top-3: {val_top3:.1f}%"
            )

            # Checkpoint on best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_top1_acc = val_top1
                patience_counter = 0

                checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_top1": val_top1,
                    "val_top3": val_top3,
                    "class_names": class_names,
                }, checkpoint_path)
                print(f"  --> Saved new best checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.early_stopping_patience:
                    print(f"\nEarly stopping triggered: validation loss did not improve for {args.early_stopping_patience} epochs.")
                    break
        else:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {tr_loss:.4f} - Top-1: {tr_top1:.1f}% - Top-3: {tr_top3:.1f}%")

    total_time = (time.time() - start_time) / 60.0
    print("=" * 65)
    print(f"Training completed in {total_time:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.4f} | Best Top-1 Accuracy: {best_top1_acc:.2f}%")

    # Save training history
    history_path = os.path.join(args.checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history log to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 15-Layer CNN for Skin Condition Classification")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing images")
    parser.add_argument("--manifest_dir", type=str, default="preprocessed", help="Directory containing preprocessed CSV splits (optional)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 weight decay for AdamW")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate in classifier")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader worker processes (0 for macOS stability)")
    parser.add_argument("--use_class_weights", action="store_true", default=True, help="Use inverse-frequency class weights")
    parser.add_argument("--scheduler_patience", type=int, default=3, help="LR scheduler patience (epochs)")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience (epochs)")

    args = parser.parse_args()
    train(args)