import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn

CLASS_LABELS = ["NORMAL_SKIN", "PSORIASIS", "Ringworm", "acne"]


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, len(CLASS_LABELS)))
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def evaluate(data_dir, model_path, output_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform)
    if dataset.classes != CLASS_LABELS:
        raise ValueError(f"Dataset class order must be {CLASS_LABELS}, found {dataset.classes}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            y_true.extend(labels.tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())

    results = {
        "samples": len(dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_LABELS)))).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(len(CLASS_LABELS))), target_names=CLASS_LABELS,
            zero_division=0, output_dict=True,
        ),
    }
    Path(output_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a held-out dataset")
    parser.add_argument("--data_dir", required=True, help="Test dataset with one folder per class")
    parser.add_argument("--model_path", default="skin_disease_model.pth")
    parser.add_argument("--output", default="evaluation.metrics.json")
    args = parser.parse_args()
    evaluate(args.data_dir, args.model_path, args.output)
