"""
Visualization tool to plot Training and Validation Learning Curves.

Reads 'training_history.json' generated during training and produces
publication/report-ready loss and accuracy curves (Top-1 and Top-3).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Silence Matplotlib cache warning and ensure src imports work
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(history_file: str, save_path: str):
    if not os.path.exists(history_file):
        print(f"Error: History file '{history_file}' not found.")
        print("Run 'src/train.py' first to generate training history.")
        return

    with open(history_file, "r") as f:
        history = json.load(f)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_top1 = history.get("train_top1", [])
    val_top1 = history.get("val_top1", [])
    val_top3 = history.get("val_top3", [])

    epochs = list(range(1, len(train_loss) + 1))
    if not epochs:
        print("Warning: History file contains no epoch data.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Loss Curves ---
    ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2)
    if val_loss:
        ax1.plot(epochs, val_loss, label="Validation Loss", color="#ff7f0e", linewidth=2, linestyle="--")
        best_epoch_loss = np.argmin(val_loss) + 1
        min_loss = val_loss[best_epoch_loss - 1]
        ax1.scatter([best_epoch_loss], [min_loss], color="red", s=60, zorder=5)
        ax1.annotate(f"Min: {min_loss:.3f} (Ep {best_epoch_loss})",
                     xy=(best_epoch_loss, min_loss),
                     xytext=(best_epoch_loss, min_loss + (max(train_loss) - min_loss) * 0.1),
                     arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
                     fontsize=9)

    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # --- Plot 2: Accuracy Curves ---
    ax2.plot(epochs, train_top1, label="Train Top-1 Accuracy", color="#2ca02c", linewidth=2)
    if val_top1:
        ax2.plot(epochs, val_top1, label="Validation Top-1 Accuracy", color="#d62728", linewidth=2, linestyle="--")
        best_epoch_acc = np.argmax(val_top1) + 1
        max_acc = val_top1[best_epoch_acc - 1]
        ax2.scatter([best_epoch_acc], [max_acc], color="blue", s=60, zorder=5)
        ax2.annotate(f"Max Top-1: {max_acc:.1f}% (Ep {best_epoch_acc})",
                     xy=(best_epoch_acc, max_acc),
                     xytext=(best_epoch_acc, max_acc - 6),
                     arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
                     fontsize=9)

    if val_top3:
        ax2.plot(epochs, val_top3, label="Validation Top-3 Accuracy", color="#9467bd", linewidth=1.8, linestyle=":")

    ax2.set_title("Accuracy (%) vs. Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Learning curves successfully saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Training & Validation Learning Curves")
    parser.add_argument("--history", type=str, default="checkpoints/training_history.json", help="Path to training history JSON")
    parser.add_argument("--output", type=str, default="checkpoints/learning_curves.png", help="Path to save output plot")

    args = parser.parse_args()
    plot_learning_curves(args.history, args.output)

