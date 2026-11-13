"""
Dataset and DataLoader utilities for the Skin Condition CNN.

Supports:
  1. Standard directory hierarchy (ImageFolder):
       data/train/<class_name>/<image.jpg>
       data/val/<class_name>/<image.jpg>
       data/test/<class_name>/<image.jpg>
  2. Optional CSV-based metadata dataset if images are stored in a flat folder.
  3. Dermatological data augmentation (rotation, flips, subtle color jitter).
  4. Automatic class weight calculation for handling 35-class imbalance.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Data augmentation pipeline tailored for skin lesion images.
    Skin lesions are rotationally invariant, but color jitter is kept subtle
    to preserve erythema and pigmentation nuances.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_eval_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Deterministic evaluation transforms for validation and test splits."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


class SkinConditionCSVDataset(Dataset):
    """
    Alternative PyTorch Dataset loading from a CSV file.
    
    Expected CSV columns:
      - 'rel_path' or 'image_path': Path to the image file relative to img_dir.
      - 'label' or 'class_name': Target label index or category name.
    """
    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        elif "class_name" in self.df.columns:
            unique_classes = sorted(self.df["class_name"].unique())
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
        else:
            unique_classes = sorted(self.df["label"].unique())
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}

        self.classes = sorted(self.class_to_idx.keys(), key=lambda k: self.class_to_idx[k])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        path_col = "rel_path" if "rel_path" in row else "image_path"
        img_path = os.path.join(self.img_dir, str(row[path_col]))
        image = Image.open(img_path).convert("RGB")

        if "label" in row and isinstance(row["label"], (int, np.integer)):
            label = int(row["label"])
        elif "class_name" in row and row["class_name"] in self.class_to_idx:
            label = self.class_to_idx[row["class_name"]]
        else:
            label_raw = row["label"]
            label = self.class_to_idx[label_raw] if isinstance(label_raw, str) else int(label_raw)

        if self.transform:
            image = self.transform(image)

        return image, label


def compute_class_weights(dataset: Union[datasets.ImageFolder, SkinConditionCSVDataset], num_classes: int = 35) -> torch.Tensor:
    """
    Computes balanced class weights inversely proportional to class frequencies:
      weight[c] = total_samples / (num_classes * count[c])
    Crucial for handling severe class imbalance across the 35 conditions.
    """
    if isinstance(dataset, datasets.ImageFolder):
        targets = [target for _, target in dataset.samples]
    elif isinstance(dataset, SkinConditionCSVDataset):
        targets = [dataset.class_to_idx[lbl] if isinstance(lbl, str) else int(lbl) for lbl in dataset.df["label"]]
    else:
        raise ValueError("Unsupported dataset type for class weight computation.")

    counts = np.bincount(targets, minlength=num_classes)
    total_samples = len(targets)
    weights = []

    for count in counts:
        if count > 0:
            w = total_samples / (num_classes * count)
        else:
            w = 1.0  # fallback for unobserved class in split
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    data_dir: str = "data",
    manifest_dir: Optional[str] = "preprocessed",
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: Tuple[int, int] = (224, 224),
    data: Optional[str] = None
) -> Dict[str, Union[DataLoader, List[str], torch.Tensor]]:
    """
    Creates DataLoaders for train, val, and test splits from an ImageFolder structure.
    
    Supports:
      1. Preprocessed CSV manifests (if manifest_dir/train.csv exists) generated by preprocess.py
      2. Standard directory hierarchy (data_dir/train, data_dir/val, data_dir/test)
    """
    if data is not None:
        data_dir = data
    train_tf = get_train_transforms(image_size)
    eval_tf = get_eval_transforms(image_size)

    loaders = {}
    class_names = []
    class_weights = None

    # Check for CSV manifest splits first (from preprocess.py)
    if manifest_dir and os.path.exists(os.path.join(manifest_dir, "train.csv")):
        train_csv = os.path.join(manifest_dir, "train.csv")
        val_csv = os.path.join(manifest_dir, "val.csv")
        test_csv = os.path.join(manifest_dir, "test.csv")
        cmap_file = os.path.join(manifest_dir, "class_map.json")
        weights_file = os.path.join(manifest_dir, "class_weights.json")

        class_to_idx = None
        if os.path.exists(cmap_file):
            with open(cmap_file) as f:
                class_to_idx = json.load(f)
            class_names = list(class_to_idx.keys())
        elif os.path.exists("class_map.json"):
            with open("class_map.json") as f:
                class_to_idx = json.load(f)
            class_names = list(class_to_idx.keys())

        train_ds = SkinConditionCSVDataset(train_csv, img_dir=data_dir, transform=train_tf, class_to_idx=class_to_idx)
        loaders["train"] = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        if not class_names:
            class_names = train_ds.classes

        if os.path.exists(weights_file):
            with open(weights_file) as f:
                w_data = json.load(f)
                mc_w = w_data.get("multiclass")
                if mc_w:
                    class_weights = torch.tensor(mc_w, dtype=torch.float32)
        if class_weights is None:
            class_weights = compute_class_weights(train_ds, num_classes=len(class_names))

        if os.path.exists(val_csv):
            val_ds = SkinConditionCSVDataset(val_csv, img_dir=data_dir, transform=eval_tf, class_to_idx=train_ds.class_to_idx)
            loaders["val"] = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            loaders["val"] = None

        if os.path.exists(test_csv):
            test_ds = SkinConditionCSVDataset(test_csv, img_dir=data_dir, transform=eval_tf, class_to_idx=train_ds.class_to_idx)
            loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            loaders["test"] = None

        loaders["class_names"] = class_names
        loaders["class_weights"] = class_weights
        return loaders

    # Fallback to ImageFolder directory hierarchy
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    if os.path.exists(train_dir) and any(os.scandir(train_dir)):
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        loaders["train"] = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        class_names = train_ds.classes
        class_weights = compute_class_weights(train_ds, num_classes=len(class_names))
    else:
        loaders["train"] = None

    if os.path.exists(val_dir) and any(os.scandir(val_dir)):
        val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
        loaders["val"] = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        loaders["val"] = None

    if os.path.exists(test_dir) and any(os.scandir(test_dir)):
        test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
        loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        loaders["test"] = None

    loaders["class_names"] = class_names
    loaders["class_weights"] = class_weights
    return loaders


if __name__ == "__main__":
    import shutil
    import tempfile

    print("Running self-test for dataset.py...")
    # Create a small temporary dataset with 2 classes to test ImageFolder loading
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_path = os.path.join(tmp_dir, "train")
        for cls_name in ["healthy", "melanoma"]:
            cls_folder = os.path.join(train_path, cls_name)
            os.makedirs(cls_folder, exist_ok=True)
            dummy_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
            dummy_img.save(os.path.join(cls_folder, "test_1.jpg"))

        loaders = get_dataloaders(data_dir=tmp_dir, manifest_dir=None, batch_size=2, num_workers=0)
        train_loader = loaders["train"]
        images, labels = next(iter(train_loader))

        print(f"Batch images shape: {images.shape} (Expected: [2, 3, 224, 224])")
        print(f"Batch labels shape: {labels.shape}")
        print(f"Detected classes:   {loaders['class_names']}")
        print(f"Computed weights:   {loaders['class_weights']}")
        assert images.shape == (2, 3, 224, 224), "Transform shape mismatch!"
        print("Dataset self-test completed successfully!")

