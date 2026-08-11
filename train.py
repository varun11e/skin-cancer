import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

CLASS_LABELS = ["NORMAL_SKIN", "PSORIASIS", "Ringworm", "acne"]


def build_transforms():
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_LABELS)))).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(len(CLASS_LABELS))), target_names=CLASS_LABELS,
            zero_division=0, output_dict=True,
        ),
    }


def train_model(data_dir, epochs=10, batch_size=16, lr=1e-4, save_path="skin_disease_model.pth"):
    data_dir = Path(data_dir)
    train_path = data_dir / "train"
    val_path = data_dir / "val"

    if not train_path.is_dir() or not val_path.is_dir():
        raise ValueError("Dataset must contain separate train/ and val/ directories. Validation on training data is not allowed.")

    transforms_map = build_transforms()
    train_dataset = datasets.ImageFolder(train_path, transforms_map["train"])
    val_dataset = datasets.ImageFolder(val_path, transforms_map["val"])

    if train_dataset.classes != CLASS_LABELS:
        raise ValueError(f"Dataset class order must be {CLASS_LABELS}, found {train_dataset.classes}.")
    if val_dataset.classes != CLASS_LABELS:
        raise ValueError(f"Validation class order must be {CLASS_LABELS}, found {val_dataset.classes}.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Train images: {len(train_dataset)} | Validation images: {len(val_dataset)}")

    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, len(CLASS_LABELS)))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=2)

    best_f1 = -1.0
    best_metrics = None
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_loss = running_loss / len(train_dataset)
        train_accuracy = correct / len(train_dataset)
        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["macro_f1"])

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={metrics['loss']:.4f} val_acc={metrics['accuracy']:.4f} "
            f"val_macro_f1={metrics['macro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_metrics = metrics
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_labels": CLASS_LABELS,
                "validation_metrics": metrics,
            }, save_path)
            print(f"  Saved best model -> {save_path}")

    metrics_path = save_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    print(f"Best validation macro-F1: {best_f1:.4f}")
    print(f"Metrics written to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the DermaAI skin-disease classifier")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", default="skin_disease_model.pth")
    args = parser.parse_args()
    train_model(**vars(args))
