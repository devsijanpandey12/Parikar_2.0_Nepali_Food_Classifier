"""
Parikar 2.0 - Automated Dataset Collector & AI Cleaner
Scrapes Nepali food images from the web and filters noise using Google SigLIP / OpenAI CLIP.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict
import requests
from PIL import Image
import io

# Default classes
NEPALI_FOOD_CLASSES = [
    "momo", "selroti", "dalbhat", "dhido", "gundruk", "yomari",
    "panipuri", "sekuwa", "juju_dhau", "samosa", "kwati", "bara",
    "chatpate", "chow_mein", "thukpa", "chiya"
]

SEARCH_QUERIES = {
    "momo": ["nepali momo traditional", "steamed momo kathmandu", "authentic buff momo achar"],
    "selroti": ["nepali selroti traditional", "sel roti tihar festival", "crispy sweet sel roti"],
    "dalbhat": ["nepali dal bhat tarkari thali", "authentic nepali khana set", "dal bhat power"],
    "dhido": ["nepali dhido kodo phapar", "traditional dhido gundruk", "nepali millet dhido"],
    "gundruk": ["nepali gundruk jhol soup", "fermented gundruk curry", "traditional gundruk bhatmas"],
    "yomari": ["newari yomari festival", "yomari chaku sweet dumpling", "steamed yomari punhi"],
    "panipuri": ["nepali panipuri street food", "spicy golgappa panipuri nepal"],
    "sekuwa": ["nepali sekuwa grilled meat", "dharan sekuwa charcoal barbecue", "spicy buff sekuwa"],
    "juju_dhau": ["bhaktapur juju dhau curd", "king of curd juju dhau clay pot"],
    "samosa": ["nepali samosa tarkari street", "crispy potato samosa chana"],
    "kwati": ["nepali kwati sprouted bean soup", "janai purnima kwati stew"],
    "bara": ["newari wo bara lentil patty", "crispy black gram bara nepal"],
    "chatpate": ["nepali chatpate street food", "spicy wai wai chatpate"],
    "chow_mein": ["nepali chow mein street noodles", "buff chowmein spicy nepal"],
    "thukpa": ["himalayan thukpa noodle soup", "sherpa thukpa vegetable meat"],
    "chiya": ["nepali masala chiya milk tea", "ilam tea milk spices glass"]
}

class DatasetCollector:
    def __init__(self, output_dir: str = "dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if os.environ.get("USE_GPU") == "1" else "cpu"
        self.clip_model = None
        self.clip_processor = None

    def init_cleaner_model(self, model_name: str = "google/siglip-base-patch16-224"):
        """Initializes SigLIP/CLIP for zero-shot image validation."""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
            print(f"Loading AI cleaner backbone: {model_name} on {self.device}...")
            self.clip_processor = AutoProcessor.from_pretrained(model_name)
            self.clip_model = AutoModelForZeroShotImageClassification.from_pretrained(model_name).to(self.device)
            print("AI cleaner initialized successfully.")
        except ImportError:
            print("Notice: 'transformers' or 'torch' not installed in current environment.")
            print("Install via: pip install torch torchvision transformers")
            self.clip_model = None

    def verify_and_save(self, image_data: bytes, class_name: str, min_confidence: float = 0.65) -> bool:
        """Verifies image integrity and uses SigLIP/CLIP to filter noise."""
        try:
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
            # Discard tiny thumbnails
            if img.width < 180 or img.height < 180:
                return False
        except Exception:
            return False

        # If CLIP is loaded, filter out non-food and wrong labels
        if self.clip_model and self.clip_processor:
            try:
                import torch
                candidate_labels = [
                    f"a photo of authentic Nepali {class_name}",
                    "a photo of unrelated food or dessert",
                    "a random object, logo, meme, or text advertisement"
                ]
                inputs = self.clip_processor(
                    text=candidate_labels,
                    images=img,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    logits = outputs.logits_per_image
                    probs = torch.softmax(logits, dim=-1)[0]
                    target_prob = probs[0].item()

                if target_prob < min_confidence:
                    return False
            except Exception as e:
                print(f"Filter check warning: {e}")

        # Compute hash to avoid duplicate images
        file_hash = hashlib.md5(image_data).hexdigest()[:12]
        class_folder = self.output_dir / class_name
        class_folder.mkdir(exist_ok=True)
        file_path = class_folder / f"{class_name}_{file_hash}.jpg"

        if file_path.exists():
            return False

        img.save(file_path, "JPEG", quality=92)
        return True

if __name__ == "__main__":
    collector = DatasetCollector(output_dir="../dataset")
    print("Parikar 2.0 Dataset Collector ready.")
    print(f"Target classes ({len(NEPALI_FOOD_CLASSES)}): {', '.join(NEPALI_FOOD_CLASSES)}")
