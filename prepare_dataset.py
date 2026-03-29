"""
prepare_dataset.py - Dataset Download & Preparation Helper
==========================================================
This script helps you:
  1. Verify dataset structure
  2. Analyse class distribution
  3. Augment underrepresented classes offline
  4. Generate a quick quality report
  5. (Optional) Download from Kaggle API
"""

import os, shutil, json
from pathlib import Path
from collections import Counter
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = "data/fish_dataset"
CLASS_NAMES = ["Fresh", "Medium", "Spoiled"]
MIN_IMAGES  = 1000      # minimum per class
IMG_SIZE    = 300


# ─── Verify structure ─────────────────────────────────────────────────────────
def verify_structure(data_dir: str) -> dict:
    """Check that all class folders exist and count images."""
    data_dir = Path(data_dir)
    counts   = {}
    missing  = []
    corrupt  = []

    for cls in CLASS_NAMES:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            missing.append(cls)
            counts[cls] = 0
            continue

        images = list(cls_dir.glob("*"))
        images = [f for f in images if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

        # Check for corrupt images
        valid = 0
        for img_path in images:
            try:
                img = Image.open(img_path)
                img.verify()
                valid += 1
            except Exception:
                corrupt.append(str(img_path))

        counts[cls] = valid

    print("\n" + "=" * 55)
    print("  DATASET STRUCTURE REPORT")
    print("=" * 55)
    for cls, n in counts.items():
        status = "✅" if n >= MIN_IMAGES else "⚠ LOW"
        print(f"  {cls:10s} : {n:5d} images  {status}")

    if missing:
        print(f"\n  ❌ Missing class folders: {missing}")
        print(f"     Create:  data/fish_dataset/{{Fresh,Medium,Spoiled}}/")

    if corrupt:
        print(f"\n  ⚠  {len(corrupt)} corrupt images found:")
        for p in corrupt[:5]:
            print(f"     {p}")

    total = sum(counts.values())
    print(f"\n  Total: {total} images")
    print("=" * 55 + "\n")

    return counts


# ─── Offline augmentation to balance classes ──────────────────────────────────
def balance_classes(data_dir: str, target: int = MIN_IMAGES):
    """
    For each class with fewer than `target` images, synthesise
    augmented copies until the target is reached.
    """
    aug = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.RandomPerspective(distortion_scale=0.2, p=0.5),
        T.GaussianBlur(3),
    ])

    data_dir = Path(data_dir)
    for cls in CLASS_NAMES:
        cls_dir = data_dir / cls
        images  = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        current = len(images)

        if current >= target:
            print(f"  {cls}: {current} images — OK, no augmentation needed.")
            continue

        needed = target - current
        print(f"  {cls}: {current}/{target} — generating {needed} augmented images …")

        aug_dir = cls_dir / "augmented"
        aug_dir.mkdir(exist_ok=True)

        for i in range(needed):
            src = random.choice(images)
            img = Image.open(src).convert("RGB")
            img = aug(img)
            img.save(aug_dir / f"aug_{i:05d}.jpg", quality=90)

        print(f"  {cls}: done → {cls_dir / 'augmented'}")


# ─── Sample grid visualisation ───────────────────────────────────────────────
def visualise_samples(data_dir: str, n_per_class: int = 6,
                       save_path: str = "outputs/dataset_samples.png"):
    data_dir = Path(data_dir)
    fig, axes = plt.subplots(
        len(CLASS_NAMES), n_per_class,
        figsize=(n_per_class * 2.5, len(CLASS_NAMES) * 2.5),
    )
    fig.patch.set_facecolor("#0f1117")

    colors = {"Fresh": "#22c55e", "Medium": "#f59e0b", "Spoiled": "#ef4444"}

    for row, cls in enumerate(CLASS_NAMES):
        cls_dir = data_dir / cls
        images  = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        samples = random.sample(images, min(n_per_class, len(images)))

        for col in range(n_per_class):
            ax = axes[row][col]
            ax.set_facecolor("#0f1117")
            ax.axis("off")

            if col < len(samples):
                img = Image.open(samples[col]).convert("RGB")
                img.thumbnail((IMG_SIZE, IMG_SIZE))
                ax.imshow(img)
                for spine in ax.spines.values():
                    spine.set_edgecolor(colors[cls])

            if col == 0:
                ax.set_ylabel(cls, color=colors[cls],
                              fontsize=12, fontweight="bold")

    plt.suptitle("Dataset Sample Grid", color="white", fontsize=15, y=1.01)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Sample grid saved → {save_path}")


# ─── Kaggle download helper ───────────────────────────────────────────────────
KAGGLE_DATASETS = [
    {
        "name":   "crowww/a-large-scale-fish-dataset",
        "desc":   "Large Scale Fish Dataset (~9k images, various species)",
        "note":   "Relabel by freshness after download.",
    },
    {
        "name":   "thedagger/fish-freshness",
        "desc":   "Fish Freshness (pre-labelled Fresh/Spoiled)",
        "note":   "Add 'Medium' class manually or label with confidence threshold.",
    },
]

def download_kaggle(dataset_slug: str, dest: str = "data/raw"):
    """
    Download a Kaggle dataset. Requires kaggle.json in ~/.kaggle/
    Run: pip install kaggle
    """
    try:
        import kaggle
        Path(dest).mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(dataset_slug, path=dest, unzip=True)
        print(f"  Downloaded {dataset_slug} → {dest}")
    except ImportError:
        print("  Install kaggle: pip install kaggle")
        print("  Then set up ~/.kaggle/kaggle.json with your API token.")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--balance",  action="store_true",
                        help="Augment minority classes to MIN_IMAGES")
    parser.add_argument("--visualise", action="store_true",
                        help="Save sample grid image")
    parser.add_argument("--kaggle",   type=str, default=None,
                        help="Kaggle dataset slug to download, e.g. thedagger/fish-freshness")
    args = parser.parse_args()

    if args.kaggle:
        download_kaggle(args.kaggle)

    counts = verify_structure(args.data_dir)

    if args.balance:
        print("Balancing classes …")
        balance_classes(args.data_dir)

    if args.visualise:
        print("Generating sample grid …")
        visualise_samples(args.data_dir)

    print("Suggested Kaggle datasets:")
    for ds in KAGGLE_DATASETS:
        print(f"  • {ds['name']}")
        print(f"    {ds['desc']}")
        print(f"    Note: {ds['note']}")
