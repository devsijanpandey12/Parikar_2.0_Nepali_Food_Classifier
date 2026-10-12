"""
Parikar 2.0 - High-Performance Dataset Expansion Pipeline (1,000 images per class)
Generates exactly 1,000 images per class (800 train, 200 val) across 10 Nepali food classes.
Features:
- Zero data leakage (strict independent raw base splitting into train & val pools)
- Photorealistic computer vision augmentations (spatial, photometric, framing variations)
- Standardized 256x256 RGB normalization for ultra-fast GPU DataLoader throughput
- Total dataset size: exactly 10,000 images (8,000 train / 2,000 val)
"""

import os
import sys
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CLASSES = [
    "momo", "selroti", "dalbhat", "panipuri", "samosa",
    "chiya", "pakauda", "jeri", "yomari", "chatpate"
]

def apply_augmentation(img: Image.Image, seed_idx: int) -> Image.Image:
    """Applies realistic camera, lighting, framing, and color variations."""
    w, h = img.size
    
    # 1. Random horizontal flip (50% chance)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
    # 2. Random slight rotation (-18 to +18 degrees with edge reflection/crop)
    if random.random() < 0.7:
        angle = random.uniform(-18, 18)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
        
    # 3. Random zoom / crop (80% to 100% of frame)
    if random.random() < 0.6:
        crop_scale = random.uniform(0.82, 0.98)
        cw, ch = int(w * crop_scale), int(h * crop_scale)
        left = random.randint(0, max(0, w - cw))
        top = random.randint(0, max(0, h - ch))
        img = img.crop((left, top, left + cw, top + ch))
        img = img.resize((w, h), Image.Resampling.BILINEAR)

    # 4. Color & Lighting Jitters
    enhancer_bright = ImageEnhance.Brightness(img)
    img = enhancer_bright.enhance(random.uniform(0.85, 1.20))
    
    enhancer_contrast = ImageEnhance.Contrast(img)
    img = enhancer_contrast.enhance(random.uniform(0.85, 1.25))
    
    enhancer_color = ImageEnhance.Color(img)
    img = enhancer_color.enhance(random.uniform(0.82, 1.25))
    
    # 5. Occasional slight camera blur or sharpness adjustment
    r = random.random()
    if r < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.1)))
    elif r > 0.8:
        enhancer_sharp = ImageEnhance.Sharpness(img)
        img = enhancer_sharp.enhance(random.uniform(1.1, 1.5))
        
    return img

def build_1k_dataset(
    raw_dir: str = "data/raw",
    output_dir: str = "data/dataset",
    target_train_per_class: int = 800,
    target_val_per_class: int = 200,
    target_size: tuple = (256, 256),
    seed: int = 42
):
    random.seed(seed)
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    train_dir = out_path / "train"
    val_dir = out_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    target_total = target_train_per_class + target_val_per_class

    print("=" * 72)
    print("PARIKAR 2.0 - 10,000 IMAGE DATASET GENERATOR (1K PER CLASS)")
    print(f"Classes: {len(CLASSES)} | Train per class: {target_train_per_class} | Val per class: {target_val_per_class}")
    print(f"Total target: {target_total * len(CLASSES):,} images | Resolution: {target_size}")
    print("=" * 72)

    summary = []

    for cls in sorted(CLASSES):
        cls_raw = raw_path / cls
        if not cls_raw.exists():
            print(f"Error: Directory for '{cls}' does not exist at {cls_raw}")
            continue

        raw_files = [f for f in cls_raw.glob("*.*") if f.is_file() and not f.name.startswith("_")]
        if not raw_files:
            print(f"Error: No raw images found for '{cls}' in {cls_raw}")
            continue

        # Shuffle and strictly partition raw base files to prevent any train/val leakage
        random.shuffle(raw_files)
        raw_val_count = max(10, int(len(raw_files) * 0.2))
        raw_val_files = raw_files[:raw_val_count]
        raw_train_files = raw_files[raw_val_count:]

        print(f"Generating '{cls.upper()}': {len(raw_files)} base ({len(raw_train_files)} train base, {len(raw_val_files)} val base)...")

        cls_train_dir = train_dir / cls
        cls_val_dir = val_dir / cls
        cls_train_dir.mkdir(parents=True, exist_ok=True)
        cls_val_dir.mkdir(parents=True, exist_ok=True)

        # Clear prior outputs if any
        for f in cls_train_dir.glob("*.jpg"): f.unlink(missing_ok=True)
        for f in cls_val_dir.glob("*.jpg"): f.unlink(missing_ok=True)

        # 1. Generate exactly target_train_per_class images for Train
        train_created = 0
        while train_created < target_train_per_class:
            base_file = raw_train_files[train_created % len(raw_train_files)]
            try:
                with Image.open(base_file) as img:
                    rgb = img.convert("RGB")
                    if train_created < len(raw_train_files):
                        # Direct original resized
                        final_img = rgb.resize(target_size, Image.Resampling.BILINEAR)
                        out_name = f"{cls}_tr_orig_{train_created:04d}.jpg"
                    else:
                        # Augmented variant
                        aug = apply_augmentation(rgb, train_created)
                        final_img = aug.resize(target_size, Image.Resampling.BILINEAR)
                        out_name = f"{cls}_tr_aug_{train_created:04d}.jpg"

                    final_img.save(cls_train_dir / out_name, "JPEG", quality=92)
                    train_created += 1
            except Exception:
                pass

        # 2. Generate exactly target_val_per_class images for Validation (from val base pool only)
        val_created = 0
        while val_created < target_val_per_class:
            base_file = raw_val_files[val_created % len(raw_val_files)]
            try:
                with Image.open(base_file) as img:
                    rgb = img.convert("RGB")
                    if val_created < len(raw_val_files):
                        # Direct original val
                        final_img = rgb.resize(target_size, Image.Resampling.BILINEAR)
                        out_name = f"{cls}_val_orig_{val_created:04d}.jpg"
                    else:
                        # Val augmented variant
                        aug = apply_augmentation(rgb, val_created + 500)
                        final_img = aug.resize(target_size, Image.Resampling.BILINEAR)
                        out_name = f"{cls}_val_aug_{val_created:04d}.jpg"

                    final_img.save(cls_val_dir / out_name, "JPEG", quality=92)
                    val_created += 1
            except Exception:
                pass

        summary.append((cls, len(raw_files), train_created, val_created, train_created + val_created))

    print("\n" + "=" * 74)
    print(f"{'Class':<12} | {'Raw Base':<10} | {'Train (800)':<12} | {'Val (200)':<10} | {'Total (1,000)':<12}")
    print("-" * 74)
    for cls, raw_c, tr_c, vl_c, tot_c in summary:
        print(f"{cls:<12} | {raw_c:<10} | {tr_c:<12} | {vl_c:<10} | {tot_c:<12}")
    print("=" * 74)
    total_all = sum(s[4] for s in summary)
    print(f"SUCCESS: Generated {total_all:,} total images ({sum(s[2] for s in summary):,} train, {sum(s[3] for s in summary):,} val)")
    print("Dataset ready for model training pipeline!")

if __name__ == "__main__":
    build_1k_dataset()
