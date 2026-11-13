"""
Single-Image Prediction & Differential Diagnosis CLI.

Features:
  - Loads trained 15-Layer Skin Condition CNN.
  - Outputs Top-1 prediction and Top-3 differential diagnoses with confidence scores.
  - Provides binary assessment: Healthy vs. Pathological.
  - Supports optional Grad-CAM heatmap generation (--cam).
"""

import argparse
import os
import sys
from pathlib import Path

# Silence Matplotlib cache warning and ensure src imports work
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import torch
import torch.nn.functional as F

from src.dataset import get_eval_transforms
from src.model import SkinConditionCNN


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(image_path: str, checkpoint_path: str, topk: int = 3, generate_cam: bool = False, cam_output: str = "checkpoints/pred_gradcam.png"):
    device = get_device()

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' not found.")
        print("Please train the model first with 'python src/train.py'.")
        return None

    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return None

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", [f"Class_{i}" for i in range(35)])
    num_classes = len(class_names)

    # Initialize model and load weights
    model = SkinConditionCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load and transform image
    raw_img = Image.open(image_path).convert("RGB")
    transform = get_eval_transforms((224, 224))
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    # Top-K predictions
    topk_probs, topk_indices = torch.topk(probs, k=min(topk, num_classes))

    top1_class = class_names[topk_indices[0].item()]
    top1_prob = topk_probs[0].item() * 100.0

    print("\n" + "=" * 60)
    print("DERMATOLOGICAL INFERENCE RESULTS")
    print("=" * 60)
    print(f"Image: {image_path}")
    print("-" * 60)

    # Binary health status
    is_healthy = top1_class.lower() == "healthy"
    status_str = "HEALTHY SKIN" if is_healthy else f"PATHOLOGICAL CONDITION DETECTED ({top1_class})"
    print(f"Assessment: {status_str}")
    print("-" * 60)

    print(f"Top-{topk} Differential Diagnoses:")
    for rank, (idx, prob) in enumerate(zip(topk_indices, topk_probs), start=1):
        cname = class_names[idx.item()]
        cprob = prob.item() * 100.0
        print(f"  {rank}. {cname:<48} : {cprob:5.2f}%")
    print("=" * 60)

    # Optional Grad-CAM
    if generate_cam:
        from src.explainability import GradCAM, overlay_cam
        import matplotlib.pyplot as plt

        target_layer = model.block4[3]
        cam_gen = GradCAM(model, target_layer)
        img_tensor_grad = transform(raw_img).unsqueeze(0).to(device)
        img_tensor_grad.requires_grad_(True)
        heatmap, pred_idx, _ = cam_gen.generate_heatmap(img_tensor_grad, class_idx=topk_indices[0].item())
        overlay = overlay_cam(raw_img, heatmap)

        os.makedirs(os.path.dirname(cam_output) or ".", exist_ok=True)
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.title(f"{top1_class} ({top1_prob:.1f}%)", fontsize=11)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(cam_output, dpi=300)
        plt.close()
        print(f"Grad-CAM overlay saved to: {cam_output}\n")

    return {
        "top1_class": top1_class,
        "top1_prob": top1_prob,
        "topk": [(class_names[idx.item()], prob.item() * 100.0) for idx, prob in zip(topk_indices, topk_probs)],
        "is_healthy": is_healthy
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict skin condition on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to best_model.pth")
    parser.add_argument("--topk", type=int, default=3, help="Number of differential diagnosis candidates to display")
    parser.add_argument("--cam", action="store_true", help="Also generate and save Grad-CAM heatmap")
    parser.add_argument("--cam_output", type=str, default="checkpoints/pred_gradcam.png", help="Path to save Grad-CAM image")

    args = parser.parse_args()
    predict(args.image, args.checkpoint, topk=args.topk, generate_cam=args.cam, cam_output=args.cam_output)

