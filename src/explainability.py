"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Skin Condition CNN.

Explains model predictions by highlighting the salient visual features
(lesion borders, pigmentation, vascular patterns) that guided the classification,
ensuring the CNN is not learning spurious artifacts (such as skin markers or hairs).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Silence Matplotlib cache warning and ensure src imports work
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.dataset import get_eval_transforms
from src.model import SkinConditionCNN


class GradCAM:
    """Computes Grad-CAM heatmaps for a specified convolutional target layer."""
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        score = output[0, class_idx]
        confidence = probabilities[0, class_idx].item() * 100.0

        # Backpropagation to target layer
        score.backward()

        # Global average pooling of gradients: weights alpha_k
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # Shape: (1, C, 1, 1)

        # Weighted sum of activation maps followed by ReLU
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx, confidence


def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays the heatmap onto the original RGB PIL image."""
    cam_resized = Image.fromarray(np.uint8(255 * cam)).resize(image.size, resample=Image.BILINEAR)
    cam_array = np.array(cam_resized) / 255.0

    colormap = plt.get_cmap("jet")
    heatmap = colormap(cam_array)[:, :, :3]  # RGB without alpha

    img_array = np.array(image.convert("RGB")) / 255.0
    overlay = (1.0 - alpha) * img_array + alpha * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def run_explainability(args):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        return
    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint.get("class_names", [f"Class_{i}" for i in range(35)])
    num_classes = len(class_names)

    model = SkinConditionCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Target the last convolutional layer in Block 4 (Layer 11)
    target_layer = model.block4[3]
    cam_generator = GradCAM(model, target_layer)

    # Load and preprocess image
    raw_img = Image.open(args.image).convert("RGB")
    transform = get_eval_transforms((224, 224))
    input_tensor = transform(raw_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    heatmap, predicted_idx, confidence = cam_generator.generate_heatmap(input_tensor, class_idx=args.target_class)
    predicted_class = class_names[predicted_idx]

    overlay = overlay_cam(raw_img, heatmap)

    # Plot figure with Original, Heatmap, and Overlay
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(raw_img)
    axes[0].set_title("Input Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"Grad-CAM Heatmap (Layer 11)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=300)
    plt.close()

    print(f"Prediction: {predicted_class} ({confidence:.2f}%)")
    print(f"Grad-CAM visualization saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization for Skin Condition CNN")
    parser.add_argument("--image", type=str, required=True, help="Path to input skin image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Trained model checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/gradcam_result.png", help="Path to save visualization")
    parser.add_argument("--target_class", type=int, default=None, help="Specific class index to explain (default: predicted class)")

    args = parser.parse_args()
    run_explainability(args)

