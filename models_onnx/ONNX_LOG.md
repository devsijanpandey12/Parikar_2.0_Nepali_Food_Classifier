# Parikar 2.0 - ONNX Model Log

Every exported ONNX model is stored under `models_onnx/<model>/` with a run id.
Retraining adds a new row here - older models are never overwritten.

**Total exports logged:** 3  
**Last updated:** 2026-10-12 15:32:45

| # | Run ID | Model | Val Acc | Train Acc | Latency (GPU) | Size | Epochs | Batch | LR | Train/Val | Train Time | File |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `20261012_153245` | **EfficientNet-B0** | 88.3% | 99.98% | 10.73 ms | 15.31 MB | 12 | 64 | 0.0003 | 8000/2000 | 1040.0s | `efficientnet_b0/efficientnet_b0_20261012_153245_acc88.30.onnx` |
| 2 | `20261012_145458` | **ConvNeXt-V2-Nano** | 90.0% | 99.94% | 7.0 ms | 57.3 MB | 10 | 32 | 0.0003 | 8000/2000 | 1052.6s | `convnext_v2/convnext_v2_20261012_145458_acc90.00.onnx` |
| 3 | `20261012_144454` | **MobileNetV4-Conv** | 87.0% | 99.98% | 5.34 ms | 9.53 MB | 12 | 64 | 0.0003 | 8000/2000 | 502.9s | `mobilenet_v4/mobilenet_v4_20261012_144454_acc87.00.onnx` |

## SHA-256 Checksums

- `efficientnet_b0/efficientnet_b0_20261012_153245_acc88.30.onnx` - `bde409354e23685df5885752bafce615fce4885f39e01d0f1f556d34955393c4`
- `convnext_v2/convnext_v2_20261012_145458_acc90.00.onnx` - `c0d7cfd59735ea3304ee8cb8bd287e878567e21ba65b11655e16d120ab547543`
- `mobilenet_v4/mobilenet_v4_20261012_144454_acc87.00.onnx` - `31817e3c8477d1d03f79efe1af27e2dffc8e02a8a8c1500f35e0dac39e8dbb29`
