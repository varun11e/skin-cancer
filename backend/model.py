from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

CLASS_LABELS = ["NORMAL_SKIN", "PSORIASIS", "Ringworm", "acne"]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(model_path: str) -> nn.Module:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}. Train a model with train.py or set MODEL_PATH."
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

    classifier_weight = state_dict.get("classifier.1.weight")
    in_features = classifier_weight.shape[1] if classifier_weight is not None else 1536

    model = models.efficientnet_b3(weights=None)
    if model.classifier[1].in_features != in_features:
        raise ValueError(
            f"Unsupported model classifier size: {in_features}. Expected EfficientNet-B3 ({model.classifier[1].in_features})."
        )

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, len(CLASS_LABELS)),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model
