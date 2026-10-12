# Parikar 2.0 • Chronological Training & Retraining Ledger 📈

This ledger records every training and retraining session in sequential order (`Session 1`, `Session 2`, etc.).
Every session stores its own independent performance charts and curves without overwriting previous sessions.

**Total Training Sessions Logged:** 1  
**Last Updated:** 2026-10-12 15:50:53  

---

## 🏆 Session 1: 1st Training Run • Base 10,000 Dataset
- **Date & Time:** 2026-10-12 15:50:52
- **Session Directory:** [`reports_and_charts/session_1/`](./session_1/)
- **Description / Notes:** Initial baseline training across MobileNetV4, ConvNeXt-V2, and EfficientNet-B0 on 10,000 balanced Nepali food images.

| Model Architecture | Best Val Acc | Final Train Acc | GPU Latency | ONNX Size | Epochs | Training Time | Curves Chart |
|---|---|---|---|---|---|---|---|
| **MobileNetV4-Conv** | **87.0%** | 99.98% | 5.34 ms | 9.53 MB | 12 | 8.4 min | [`mobilenet_v4_curves.png`](./session_1/mobilenet_v4_curves.png) |
| **ConvNeXt-V2-Nano** | **90.0%** | 99.94% | 7.0 ms | 57.3 MB | 10 | 17.5 min | [`convnext_v2_curves.png`](./session_1/convnext_v2_curves.png) |
| **EfficientNet-B0** | **88.3%** | 99.98% | 10.73 ms | 15.31 MB | 12 | 17.3 min | [`efficientnet_b0_curves.png`](./session_1/efficientnet_b0_curves.png) |

**Session Visualizations:**
- 📊 [Comparative Benchmark Dashboard Chart](./session_1/model_comparison.png)
- 🌐 [Interactive HTML Dashboard](./session_1/benchmark_dashboard.html)

---
