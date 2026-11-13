"""
Evaluation and Diagnostic Reporting for Skin Condition CNN.

Features:
  - Computes Top-1 and Top-3 accuracy across all classes.
  - Generates detailed per-class classification report (precision, recall, F1).
  - Plots and saves a normalized confusion matrix heatmap.
"""

import argparse
import os
import sys
from pathlib import Path

# Silence Matplotlib font cache warning in sandbox/temp directories
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Ensure project root is in sys.path so 'src.' imports work when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
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
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    """Plots and saves a normalized confusion matrix heatmap."""
    # Normalize by row (true labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm_norm,
        annot=False,  # Too crowded for 35 classes; color intensity represents density
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Ratio"}
    )
    plt.title("Normalized Confusion Matrix (35 Skin Conditions)", fontsize=16)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def evaluate_checkpoint(args):
    device = get_device()
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        return

    print(f"Loading checkpoint from '{args.checkpoint}'...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint.get("class_names")

    loaders = get_dataloaders(
        data_dir=args.data_dir,
        manifest_dir=args.manifest_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    # Prefer test split; fallback to val split if test does not exist
    loader = loaders.get("test") or loaders.get("val")

    if loader is None:
        print(f"\n[Error] No test or validation data found.")
        print(f"Option 1 Workflow looks for: '{args.manifest_dir}/test.csv' (or 'val.csv').")
        print(f"Please ensure CSV splits have been generated via 'python preprocess.py'.\n")
        return

    if class_names is None:
        class_names = loaders["class_names"]

    num_classes = len(class_names)
    model = SkinConditionCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    all_top3_matches = []

    print(f"Evaluating {len(loader.dataset)} samples across {num_classes} classes...")
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)

            # Top-1 predictions
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

            # Top-3 predictions
            _, top3_pred = outputs.topk(3, dim=1, largest=True, sorted=True)
            for idx, target in enumerate(targets):
                all_top3_matches.append(target.item() in top3_pred[idx].cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Metrics computation
    top1_acc = np.mean(all_preds == all_targets) * 100.0
    top3_acc = np.mean(all_top3_matches) * 100.0

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc:.2f}% (Differential Diagnosis window)")
    print("-" * 65)

    print("\nPer-Class Classification Report:")
    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    os.makedirs(args.output_dir, exist_ok=True)
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 15-Layer Skin Condition CNN")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing images")
    parser.add_argument("--manifest_dir", type=str, default="preprocessed", help="Directory containing preprocessed CSV splits (optional)")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save evaluation plots")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers (0 for macOS stability)")

    args = parser.parse_args()
    evaluate_checkpoint(args)