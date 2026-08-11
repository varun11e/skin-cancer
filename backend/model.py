import hashlib
from pathlib import Path

import requests
import torch
import torch.nn as nn
from torchvision import models, transforms

CLASS_LABELS = ["NORMAL_SKIN", "PSORIASIS", "Ringworm", "acne"]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _download_model(url: str, path: Path, expected_sha256: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    digest = hashlib.sha256()
    try:
        with requests.get(url, stream=True, timeout=(10, 300)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    digest.update(chunk)
                    output.write(chunk)
        actual = digest.hexdigest().lower()
        if expected_sha256 and actual != expected_sha256.lower():
            raise RuntimeError("Downloaded model checksum does not match MODEL_SHA256.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(model_path: str, model_url: str = "", model_sha256: str = "") -> nn.Module:
    path = Path(model_path)
    if not path.is_file():
        if not model_url:
            raise FileNotFoundError(
                f"Model file not found: {path}. Provide MODEL_URL or train a model with train.py."
            )
        _download_model(model_url, path, model_sha256)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

    model = models.efficientnet_b3(weights=None)
    classifier_weight = state_dict.get("classifier.1.weight")
    in_features = classifier_weight.shape[1] if classifier_weight is not None else 1536
    if in_features != model.classifier[1].in_features:
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
