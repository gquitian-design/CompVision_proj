"""
Skin Lesion Classification -- preprocessing

"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ----------------------------------------------------------------------
# CONFIG - Option 1: CSV Manifest with data/<Class Name>/<image.jpg>
# ----------------------------------------------------------------------
RAW_CSV = Path("balanced_dataset.csv")
OUT_DIR = Path("preprocessed")

# Folder containing the 35 class subfolders (e.g. data/Healthy/...)
DATA_ROOT = Path("data")

# Non-diagnosis folders to drop (case-insensitive)
DROP_CATEGORIES = {"test", "train", "val", "valid", "Test", "Train", "Valid", "Val"}

# Raw folder name -> clean class name. 
RENAME = {
    "Heathy": "Healthy",
    "Ba  Cellulitis": "Ba Cellulitis",
}

HEALTHY_CLASSES = {"Healthy"}  # binary label 0; everything else 1

# Split proportions 
TEST_FRAC, VAL_FRAC = 0.15, 0.15
N_FOLDS = 20 
SEED = 42

# Pre-generated "aug_" images:
KEEP_AUG_IN_TRAIN = True
DROP_AUG_FROM_EVAL = True

AUG_PATTERN = re.compile(r"^aug_\d+_")

GENERIC_STEM = re.compile(r"^(img)?\d+$", re.IGNORECASE)


def log(report, msg=""):
    print(msg)
    report.append(msg)


def build_manifest(report):
    if not RAW_CSV.exists():
        log(report, f"[Error] '{RAW_CSV.resolve()}' not found.")
        log(report, f"Please place 'balanced_dataset.csv' in the project directory before running preprocess.py.")
        return None, None

    df = pd.read_csv(RAW_CSV)
    log(report, f"Raw rows: {len(df):,} | classes: {df['category'].nunique()}")

    # portable paths
    # Windows paths mix "/" and "\" here
    parts = df["image_path"].str.replace("\\", "/", regex=False).str.split("/")
    df["folder"] = parts.str[-2]
    df["filename"] = parts.str[-1]
    df["rel_path"] = df["folder"] + "/" + df["filename"]
    mismatch = (df["folder"] != df["category"]).sum()
    log(report, f"Rows where folder != category: {mismatch}")

    # exact duplicates 
    before = len(df)
    df = df.drop_duplicates(subset=["rel_path", "category"])
    log(report, f"Dropped exact duplicate rows: {before - len(df):,}")

    # non-diagnosis folders 
    before = len(df)
    df = df[~df["category"].isin(DROP_CATEGORIES)]
    log(report, f"Dropped split-name folders {sorted(DROP_CATEGORIES)}: {before - len(df):,} rows")

    # clean class names 
    df["class_name"] = (
        df["category"].replace(RENAME).str.replace(r"\s+", " ", regex=True).str.strip()
    )

    # augmentation lineage 
    df["is_aug"] = df["filename"].str.match(AUG_PATTERN)
    df["source_file"] = df["filename"].str.replace(AUG_PATTERN, "", regex=True)
    stem = df["source_file"].str.rsplit(".", n=1).str[0]
    generic = stem.str.match(GENERIC_STEM)
    df["group_id"] = np.where(
        generic, df["class_name"] + "/" + df["source_file"], df["source_file"]
    )

    # Every aug row points to original existing in same class.
    originals = set(zip(df.loc[~df.is_aug, "class_name"], df.loc[~df.is_aug, "source_file"]))
    orphan = sum((c, s) not in originals for c, s in
                 zip(df.loc[df.is_aug, "class_name"], df.loc[df.is_aug, "source_file"]))
    log(report, f"Augmented rows: {df.is_aug.sum():,} | orphans (no original found): {orphan}")

    n_cls = df[~df.is_aug].groupby("group_id")["class_name"].nunique()
    multi = set(n_cls[n_cls > 1].index)
    df["multi_class_image"] = df["group_id"].isin(multi)
    log(report, f"Original images filed under >1 class: {len(multi):,}")

    # labels
    classes = sorted(df["class_name"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label"] = df["class_name"].map(class_to_idx)
    df["binary_label"] = (~df["class_name"].isin(HEALTHY_CLASSES)).astype(int)
    log(report, f"Final classes: {len(classes)} (Healthy + {len(classes) - 1} conditions)")
    return df, class_to_idx


def assign_splits(df, report):
    orig = df[~df.is_aug].drop_duplicates("group_id")
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = pd.Series(-1, index=orig.index)
    for k, (_, idx) in enumerate(sgkf.split(orig, orig["label"], groups=orig["group_id"])):
        fold.iloc[idx] = k

    n_test, n_val = round(TEST_FRAC * N_FOLDS), round(VAL_FRAC * N_FOLDS)
    split_of_fold = lambda k: "test" if k < n_test else "val" if k < n_test + n_val else "train"
    group_split = dict(zip(orig["group_id"], fold.map(split_of_fold)))
    df["split"] = df["group_id"].map(group_split)

    before = len(df)
    if DROP_AUG_FROM_EVAL:
        df = df[~(df.is_aug & df.split.isin(["val", "test"]))]
    if not KEEP_AUG_IN_TRAIN:
        df = df[~(df.is_aug & (df.split == "train"))]
    log(report, f"Removed pre-made augmentations per config: {before - len(df):,}")

    # Leakage check: no group may appear in two splits.
    leaks = (df.groupby("group_id")["split"].nunique() > 1).sum()
    log(report, f"Groups appearing in more than one split (must be 0): {leaks}")
    return df


def class_weights(df, class_to_idx):
    tr = df[(df.split == "train") & ~df.is_aug]
    counts = tr["label"].value_counts().reindex(range(len(class_to_idx)), fill_value=0)
    w = len(tr) / (len(counts) * counts.clip(lower=1))
    w = w / w.mean()
    b = tr["binary_label"].value_counts().reindex([0, 1], fill_value=0)
    bw = len(tr) / (2 * b.clip(lower=1))
    return {"multiclass": w.round(4).tolist(),
            "binary": (bw / bw.mean()).round(4).tolist(),
            "binary_pos_weight": round(float(b[0] / max(b[1], 1)), 6)}


def summarise(df, report):
    log(report)
    log(report, "Rows per split (all / originals only / healthy originals):")
    for s in ["train", "val", "test"]:
        part = df[df.split == s]
        o = part[~part.is_aug]
        log(report, f"  {s:5s}: {len(part):7,} / {len(o):6,} / {int((o.binary_label == 0).sum()):4}")
    o = df[~df.is_aug]
    log(report)
    log(report, "Original images per class (true data size before augmentation):")
    for name, n in o.groupby("class_name").size().sort_values().items():
        log(report, f"  {n:5,}  {name}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    report = []
    df, class_to_idx = build_manifest(report)
    if df is None:
        return
    df = assign_splits(df, report)
    summarise(df, report)

    cols = ["rel_path", "class_name", "label", "binary_label", "split",
            "is_aug", "group_id", "multi_class_image"]
    df[cols].sort_values(["split", "class_name", "rel_path"]).to_csv(
        OUT_DIR / "manifest_clean.csv", index=False)
    for s in ["train", "val", "test"]:
        df.loc[df.split == s, cols].to_csv(OUT_DIR / f"{s}.csv", index=False)
    (OUT_DIR / "class_map.json").write_text(json.dumps(class_to_idx, indent=2))
    (OUT_DIR / "class_weights.json").write_text(json.dumps(class_weights(df, class_to_idx), indent=2))
    (OUT_DIR / "preprocessing_report.txt").write_text("\n".join(report))
    print(f"\nSaved to {OUT_DIR.resolve()}")


# Image-level preprocessing 
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224


def load_and_preprocess(rel_path, root=DATA_ROOT, size=IMG_SIZE):
    """Read one image -> float32 array of shape (3, size, size), normalised."""
    from PIL import Image
    img = Image.open(Path(root) / rel_path).convert("RGB")  # PNGs RGBA/greyscale
    img = img.resize((size, size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0           # HWC in [0, 1]
    x = (x - MEAN) / STD                                     # per-channel normalisation
    return x.transpose(2, 0, 1)                              # CHW for PyTorch


def compute_mean_std(rel_paths, root=DATA_ROOT, size=IMG_SIZE):
    """Per-channel mean/std over a list of images (use train originals only)."""
    from PIL import Image
    s = np.zeros(3); s2 = np.zeros(3); n = 0
    for p in rel_paths:
        x = np.asarray(Image.open(Path(root) / p).convert("RGB")
                       .resize((size, size)), dtype=np.float64) / 255.0
        s += x.sum((0, 1)); s2 += (x ** 2).sum((0, 1)); n += size * size
    mean = s / n
    return mean, np.sqrt(s2 / n - mean ** 2)


if __name__ == "__main__":
    main()
