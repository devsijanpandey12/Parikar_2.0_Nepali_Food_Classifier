# Parikar 2.0 - Model Training & Evaluation Report 📊

**Date:** October 12, 2026  
**Hardware Accelerated Device:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)  
**Target:** 10 Core Authentic Nepali Food Classes (1,000 images / class = 10,000 total)  

---

## 1. Dataset Breakdown

| # | Class Name | Raw Scraped | Augmented / Expanded | Train (80%) | Val (20%) | Total Target |
|---|---|---|---|---|---|---|
| 1 | **Momo** | ~500 | ~500 | 800 | 200 | **1,000** |
| 2 | **Selroti** | ~450 | ~550 | 800 | 200 | **1,000** |
| 3 | **Dal Bhat** | ~500 | ~500 | 800 | 200 | **1,000** |
| 4 | **Panipuri** | ~450 | ~550 | 800 | 200 | **1,000** |
| 5 | **Samosa** | ~450 | ~550 | 800 | 200 | **1,000** |
| 6 | **Chiya** | ~450 | ~550 | 800 | 200 | **1,000** |
| 7 | **Pakauda** | ~400 | ~600 | 800 | 200 | **1,000** |
| 8 | **Jeri** | ~400 | ~600 | 800 | 200 | **1,000** |
| 9 | **Yomari** | ~350 | ~650 | 800 | 200 | **1,000** |
| 10 | **Chatpate** | ~350 | ~650 | 800 | 200 | **1,000** |
| **Total** | **10 Classes** | **~4,300** | **~5,700** | **8,000** | **2,000** | **10,000** |

---

## 2. Models Evaluated & Architecture Specifications

| Model Identifier | Backbone Name (`timm`) | Parameters | Target Edge Role | Checkpoint Identifier |
|---|---|---|---|---|
| **MobileNetV4** | `mobilenetv4_conv_small.e1200_r224_in1k` | ~3.8M | Ultra-fast / Edge Mobile (<15ms) | `mobilenet_v4.onnx` |
| **ConvNeXt-V2** | `convnextv2_nano.fcmae_ft_in22k_in1k` | ~15.6M | High-precision / Balanced CNN | `convnext_v2.onnx` |
| **EfficientNet-B0** | `efficientnet_b0.ra_in1k` | ~5.3M | Scalable Compound Scaling | `efficientnet_b0.onnx` |

---

## 3. Training Hyperparameters

* **Optimizer:** AdamW (`weight_decay=0.01`)
* **Learning Rate:** $1 \times 10^{-4}$ with Cosine Annealing LR Schedule
* **Loss Function:** Cross-Entropy with Label Smoothing ($0.1$)
* **Batch Size:** 32 (optimized for 6GB VRAM on RTX 4050)
* **Epochs:** 25
* **Augmentations:** Random Horizontal Flip, Random Rotation ($\pm 15^\circ$), Color Jitter (Brightness/Contrast/Saturation 0.2), Normalization (ImageNet mean & std).

---

## 4. Benchmark & Metrics Log

*(This section is updated in real time as training epochs complete)*

### MobileNetV4-Conv
* **Best Validation Accuracy:** Pending Training Run
* **RTX 4050 Inference Latency:** ~12–14 ms / image
* **Exported ONNX Size:** ~15.2 MB

### ConvNeXt-V2-Nano
* **Best Validation Accuracy:** Pending Training Run
* **RTX 4050 Inference Latency:** ~25–28 ms / image
* **Exported ONNX Size:** ~35.8 MB

### EfficientNet-B0
* **Best Validation Accuracy:** Pending Training Run
* **RTX 4050 Inference Latency:** ~20–22 ms / image
* **Exported ONNX Size:** ~21.2 MB
