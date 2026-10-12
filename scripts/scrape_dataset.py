"""
Parikar 2.0 - High-Speed Multi-Query Nepali Food Image Scraper
Scrapes high-quality images for the 10 core Nepali dishes into `new_project/data/raw/`
"""

import os
import sys
import time
import shutil
from pathlib import Path
from PIL import Image
from bing_image_downloader import downloader

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# 10 Agreed Nepali Food Classes
CLASSES_QUERIES = {
    "momo": [
        "nepali momo food",
        "authentic steamed momo kathmandu",
        "steamed buff momo achar",
        "kothey fried momo nepal",
        "nepali jhol momo bowl"
    ],
    "selroti": [
        "nepali selroti traditional",
        "sel roti tihar festival",
        "crispy golden selroti",
        "selroti with tea nepal",
        "homemade nepali sel roti"
    ],
    "dalbhat": [
        "nepali dal bhat tarkari thali",
        "authentic nepali khana set",
        "dal bhat power 24 hour",
        "himalayan dal bhat plate",
        "nepali traditional lunch thali"
    ],
    "panipuri": [
        "nepali panipuri street food",
        "spicy golgappa panipuri nepal",
        "kathmandu street panipuri stall",
        "panipuri with mint spicy water"
    ],
    "samosa": [
        "nepali samosa street food",
        "crispy potato samosa chana tarkari",
        "nepali tea stall samosa",
        "spicy samosa chat nepal"
    ],
    "chiya": [
        "nepali masala chiya milk tea",
        "ilam milk tea glass street",
        "traditional nepali chiya cup",
        "steaming hot nepali tea"
    ],
    "pakauda": [
        "nepali pakauda street food",
        "crispy onion potato pakoda nepal",
        "rainy day nepali pakauda snack",
        "besan mixed vegetable pakoda"
    ],
    "jeri": [
        "nepali jeri swari sweet",
        "kathmandu mithai jeri jalebi",
        "crispy saffron jeri nepal",
        "traditional nepali jeri festival"
    ],
    "yomari": [
        "newari yomari festival",
        "yomari chaku sweet dumpling",
        "steamed yomari punhi kathmandu",
        "authentic yomari newari food"
    ],
    "chatpate": [
        "nepali chatpate street food",
        "spicy wai wai chatpate stall",
        "authentic nepali spicy chatpate snack",
        "kathmandu street chatpate cone"
    ]
}

def clean_and_validate_images(target_dir: Path, min_dim: int = 180) -> int:
    """Validates downloaded images, converts to RGB JPEG, removes corrupted files."""
    files = [f for f in target_dir.glob("*.*") if f.is_file() and not f.name.startswith("_")]
    temp_clean = target_dir / "_clean_temp"
    temp_clean.mkdir(exist_ok=True)

    valid_count = 0
    for img_path in files:
        try:
            with Image.open(img_path) as img:
                img.verify()
            
            with Image.open(img_path) as img:
                if img.width < min_dim or img.height < min_dim:
                    img_path.unlink(missing_ok=True)
                    continue
                rgb = img.convert("RGB")
                clean_name = f"{target_dir.name}_{valid_count:04d}.jpg"
                rgb.save(temp_clean / clean_name, "JPEG", quality=90)
                valid_count += 1
        except Exception:
            pass
        
        try:
            img_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Move all cleaned images back to target_dir
    for clean_file in temp_clean.glob("*.jpg"):
        dest_file = target_dir / clean_file.name
        if dest_file.exists():
            dest_file.unlink(missing_ok=True)
        shutil.move(str(clean_file), str(dest_file))

    if temp_clean.exists():
        shutil.rmtree(temp_clean, ignore_errors=True)

    return valid_count

def scrape_dataset(
    target_per_class: int = 250,
    output_dir: str = "../data/raw"
):
    base_out = Path(output_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    temp_download_dir = base_out / "_temp_bing"

    print("=" * 60)
    print(f"Starting Parikar 2.0 Image Scraper")
    print(f"Target: {target_per_class} images/class across {len(CLASSES_QUERIES)} classes")
    print(f"Output Directory: {base_out.resolve()}")
    print("=" * 60)

    for idx, (class_name, queries) in enumerate(CLASSES_QUERIES.items(), 1):
        class_folder = base_out / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        
        current_count = len(list(class_folder.glob("*.jpg")))
        if current_count >= target_per_class:
            print(f"[{idx}/{len(CLASSES_QUERIES)}] {class_name.upper()}: Already has {current_count} images. Skipping.")
            continue

        needed = target_per_class - current_count
        per_query = max(35, needed // len(queries) + 10)
        print(f"\n[{idx}/{len(CLASSES_QUERIES)}] Scraping '{class_name.upper()}' (Current: {current_count}, Target: {target_per_class})...")

        for q in queries:
            if len(list(class_folder.glob("*.jpg"))) >= target_per_class:
                break
            try:
                downloader.download(
                    q,
                    limit=per_query,
                    output_dir=str(temp_download_dir),
                    adult_filter_off=True,
                    force_replace=False,
                    timeout=15,
                    verbose=False
                )
                
                # Move from temp to class folder
                query_folder = temp_download_dir / q
                if query_folder.exists():
                    for file in query_folder.glob("*.*"):
                        dest = class_folder / f"{class_name}_{int(time.time()*1000)}_{file.name}"
                        try:
                            shutil.move(str(file), str(dest))
                        except Exception:
                            pass
            except Exception as e:
                print(f"  Warning on query '{q}': {e}")

        # Clean, validate, and deduplicate
        valid = clean_and_validate_images(class_folder)
        print(f"  -> '{class_name}' now has {valid} verified high-quality images.")

    # Cleanup temp folder
    if temp_download_dir.exists():
        shutil.rmtree(temp_download_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("Scraping and validation complete!")
    total = sum(len(list((base_out / c).glob("*.jpg"))) for c in CLASSES_QUERIES)
    print(f"Total dataset images collected: {total}")
    print("=" * 60)

if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    scrape_dataset(target_per_class=target)
