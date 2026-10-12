"""
Parikar 2.0 - PyTorch / Timm Model Training & ONNX Export Pipeline
Trains MobileNetV4, ConvNeXt-V2, and EfficientNetV2 on Nepali food dataset.
Exportable directly to ONNX for web / cloud deployment.
"""

import os
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# Model architecture catalog
ARCHITECTURES = {
    "mobilenet_v4": "mobilenetv4_conv_small.e1200_r224_in1k",
    "convnext_v2": "convnextv2_nano.fcmae_ft_in22k_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k"
}

def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def train_and_export(
    model_key: str,
    data_dir: str,
    output_dir: str = "../backend/weights",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training {model_key} on {device} ---")
    
    train_tf, val_tf = get_transforms()
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)
    
    num_classes = len(train_dataset.classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Instantiate pretrained model from timm
    backbone_name = ARCHITECTURES[model_key]
    model = timm.create_model(backbone_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_weights_path = out_path / f"{model_key}.pth"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)

    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")
    
    # Export to ONNX
    print("Exporting best model to ONNX...")
    model.load_state_dict(torch.load(best_weights_path, map_location="cpu"))
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_file = out_path / f"{model_key}.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_file),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14
    )
    print(f"ONNX model successfully saved at: {onnx_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(ARCHITECTURES.keys()), default="convnext_v2")
    parser.add_argument("--data", default="../dataset")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    
    print("Parikar 2.0 Trainer Pipeline Ready.")
