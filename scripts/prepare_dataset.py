"""
Parikar 2.0 - Dataset Preparation & Train/Val Split
Splits verified raw images from `data/raw/` into `data/dataset/train/` (80%) and `data/dataset/val/` (20%).
"""

import os
import shutil
import random
from pathlib import Path

def prepare_splits(
    raw_dir: str = "../data/raw",
    output_dir: str = "../data/dataset",
    train_ratio: float = 0.8,
    seed: int = 42
):
    random.seed(seed)
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    train_dir = out_path / "train"
    val_dir = out_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in raw_path.iterdir() if d.is_dir() and not d.name.startswith("_")]
    print("=" * 65)
    print("PARIKAR 2.0 - DATASET PREPARATION & STRATIFIED SPLIT")
    print(f"Classes Found: {len(classes)}")
    print(f"Output Directory: {out_path.resolve()}")
    print("=" * 65)

    stats = []
    total_train = 0
    total_val = 0

    for cls in sorted(classes):
        cls_raw = raw_path / cls
        images = list(cls_raw.glob("*.jpg")) + list(cls_raw.glob("*.jpeg")) + list(cls_raw.glob("*.png"))
        
        # Shuffle reproducible
        random.shuffle(images)
        
        n_total = len(images)
        if n_total == 0:
            continue
            
        n_train = int(n_total * train_ratio)
        train_imgs = images[:n_train]
        val_imgs = images[n_train:]

        cls_train_dir = train_dir / cls
        cls_val_dir = val_dir / cls
        cls_train_dir.mkdir(parents=True, exist_ok=True)
        cls_val_dir.mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            shutil.copy2(str(img), str(cls_train_dir / img.name))
            
        for img in val_imgs:
            shutil.copy2(str(img), str(cls_val_dir / img.name))

        total_train += len(train_imgs)
        total_val += len(val_imgs)
        stats.append((cls, n_total, len(train_imgs), len(val_imgs)))

    print(f"{'Class Name':<16} | {'Total Images':<14} | {'Train (80%)':<12} | {'Val (20%)':<10}")
    print("-" * 65)
    for cls, tot, tr, vl in stats:
        print(f"{cls:<16} | {tot:<14} | {tr:<12} | {vl:<10}")
    print("-" * 65)
    print(f"{'TOTAL':<16} | {total_train + total_val:<14} | {total_train:<12} | {total_val:<10}")
    print("=" * 65)
    print("Dataset preparation complete! Ready for PyTorch / Timm training.")

if __name__ == "__main__":
    prepare_splits()
