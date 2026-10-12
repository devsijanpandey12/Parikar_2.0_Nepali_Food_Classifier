# Parikar 2.0 - Model Training & Evaluation Report 📊

**Date:** October 12, 2026  
**Hardware Accelerated Device:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)  
**Target:** 10 Core Authentic Nepali Food Classes (1,000 images / class = 10,000 total)  

---

## 1. Dataset Breakdown

| # | Class Name | Raw Base Pool | Generated Train (80%) | Generated Val (20%) | Total Verified |
|---|---|---|---|---|---|
| 1 | **Momo** | 198 | 800 | 200 | **1,000** |
| 2 | **Selroti** | 171 | 800 | 200 | **1,000** |
| 3 | **Dal Bhat** | 193 | 800 | 200 | **1,000** |
| 4 | **Panipuri** | 177 | 800 | 200 | **1,000** |
| 5 | **Samosa** | 151 | 800 | 200 | **1,000** |
| 6 | **Chiya** | 171 | 800 | 200 | **1,000** |
| 7 | **Pakauda** | 140 | 800 | 200 | **1,000** |
| 8 | **Jeri** | 176 | 800 | 200 | **1,000** |
| 9 | **Yomari** | 148 | 800 | 200 | **1,000** |
| 10 | **Chatpate** | 64 | 800 | 200 | **1,000** |
| **Total** | **10 Classes** | **1,588 Base** | **8,000 Train** | **2,000 Val** | **10,000 Images** |

---

## 2. Models Evaluated & Architecture Specifications

| Model Identifier | Backbone Name (`timm`) | Parameters | Target Role | Exported ONNX |
|---|---|---|---|---|
| **MobileNetV4** | `mobilenetv4_conv_small.e1200_r224_in1k` | ~2.5M | Real-time Edge (<15ms) | `mobilenet_v4.onnx` |
| **ConvNeXt-V2** | `convnextv2_nano.fcmae_ft_in22k_in1k` | ~15.0M | High-precision / Balanced CNN | `convnext_v2.onnx` |
| **EfficientNet-B0** | `efficientnet_b0.ra_in1k` | ~4.0M | Compound Scaled Lightweight | `efficientnet_b0.onnx` |

---

## 3. Training Hyperparameters

* **Hardware Device:** NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.4, PyTorch 2.6.0)
* **Precision:** Mixed Precision FP16 with PyTorch AMP & GradScaler
* **Optimizer:** AdamW (`weight_decay=0.01`)
* **Learning Rate Schedule:** CosineAnnealingLR ($3 \times 10^{-4} \to 1 \times 10^{-6}$)
* **Loss Function:** Cross-Entropy with Label Smoothing ($0.1$)
* **Batch Size:** 64
* **Epochs:** 15
* **Augmentations:** Random Horizontal Flip, Random Rotation ($\pm 18^\circ$), Random Crop/Zoom, Color Jitter, Normalization.

---

## 4. Benchmark & Metrics Log (Empirical Training Results)

| Model Name | Backbone | Best Val Accuracy | Final Train Accuracy | Inference Latency (GPU) | ONNX Size |
|---|---|---|---|---|---|
| **MobileNetV4-Conv** | `mobilenetv4_conv_small.e1200_r224_in1k` | **87.0%** | 99.98% | 5.34 ms | 9.53 MB |
| **ConvNeXt-V2-Nano** | `convnextv2_nano.fcmae_ft_in22k_in1k` | **90.0%** | 99.94% | 7.0 ms | 57.3 MB |
| **EfficientNet-B0** | `efficientnet_b0.ra_in1k` | **88.3%** | 99.98% | 10.73 ms | 15.31 MB |

---

## 5. Epoch-by-Epoch Convergence Details

### MobileNetV4-Conv

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Epoch Time |
|---|---|---|---|---|---|
| Epoch 01 | 1.3042 | 75.74% | 1.0299 | 81.60% | 93.1s |
| Epoch 02 | 0.6315 | 98.44% | 0.9029 | 85.15% | 35.9s |
| Epoch 03 | 0.5815 | 99.64% | 0.8855 | 86.00% | 35.2s |
| Epoch 04 | 0.5601 | 99.75% | 0.8879 | 86.20% | 37.1s |
| Epoch 05 | 0.5500 | 99.89% | 0.8553 | 86.50% | 37.1s |
| Epoch 06 | 0.5433 | 99.94% | 0.8670 | 86.40% | 36.8s |
| Epoch 07 | 0.5341 | 99.94% | 0.8635 | 86.10% | 36.6s |
| Epoch 08 | 0.5324 | 99.95% | 0.8480 | 87.00% | 36.9s |
| Epoch 09 | 0.5284 | 99.98% | 0.8612 | 86.60% | 36.4s |
| Epoch 10 | 0.5281 | 99.95% | 0.8597 | 86.20% | 41.7s |
| Epoch 11 | 0.5257 | 99.95% | 0.8494 | 86.70% | 39.2s |
| Epoch 12 | 0.5257 | 99.98% | 0.8546 | 86.40% | 36.8s |

### ConvNeXt-V2-Nano

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Epoch Time |
|---|---|---|---|---|---|
| Epoch 01 | 0.6773 | 93.85% | 0.8297 | 86.70% | 99.4s |
| Epoch 02 | 0.5186 | 99.61% | 0.7648 | 89.05% | 103.8s |
| Epoch 03 | 0.5125 | 99.61% | 0.8834 | 85.30% | 106.8s |
| Epoch 04 | 0.5563 | 98.01% | 0.7702 | 89.55% | 107.0s |
| Epoch 05 | 0.5076 | 99.74% | 0.8098 | 89.00% | 108.9s |
| Epoch 06 | 0.5029 | 99.90% | 0.7922 | 89.65% | 104.0s |
| Epoch 07 | 0.5021 | 99.94% | 0.7947 | 89.80% | 105.8s |
| Epoch 08 | 0.5020 | 99.91% | 0.7940 | 89.75% | 107.1s |
| Epoch 09 | 0.5018 | 99.95% | 0.7891 | 90.00% | 104.4s |
| Epoch 10 | 0.5018 | 99.94% | 0.7895 | 90.00% | 105.0s |

### EfficientNet-B0

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Epoch Time |
|---|---|---|---|---|---|
| Epoch 01 | 1.0089 | 87.28% | 1.0445 | 82.15% | 47.4s |
| Epoch 02 | 0.5921 | 99.70% | 0.9118 | 86.45% | 61.6s |
| Epoch 03 | 0.5592 | 99.83% | 0.8917 | 86.05% | 80.1s |
| Epoch 04 | 0.5436 | 99.90% | 0.8701 | 86.45% | 67.1s |
| Epoch 05 | 0.5365 | 99.92% | 0.8595 | 86.75% | 98.5s |
| Epoch 06 | 0.5307 | 99.92% | 0.8446 | 87.60% | 102.4s |
| Epoch 07 | 0.5271 | 99.92% | 0.8406 | 87.35% | 103.6s |
| Epoch 08 | 0.5236 | 99.96% | 0.8417 | 87.50% | 115.5s |
| Epoch 09 | 0.5220 | 99.95% | 0.8340 | 88.15% | 96.5s |
| Epoch 10 | 0.5205 | 99.95% | 0.8353 | 88.30% | 95.8s |
| Epoch 11 | 0.5198 | 99.96% | 0.8340 | 87.85% | 85.4s |
| Epoch 12 | 0.5191 | 99.98% | 0.8316 | 87.90% | 85.6s |

---

## 6. Summary & Deployment Recommendation

* **Primary Production Model:** **ConvNeXt-V2-Nano** delivers exceptional visual feature representations with modern 7x7 depthwise convolutions and inverted bottleneck design.
* **Low-Latency / Mobile Recommendation:** **MobileNetV4-Conv** delivers instantaneous sub-10ms inference with minimal memory footprint, ideal for continuous camera streams.
* **Compound Scaler:** **EfficientNet-B0** strikes an optimal middle ground with robust cross-domain generalizability.
