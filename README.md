# DermaAI — Skin Disease Classification

DermaAI is a prototype web application that uses a PyTorch EfficientNet-B3 image-classification model to classify four skin-image categories:

- `NORMAL_SKIN`
- `PSORIASIS`
- `Ringworm`
- `acne`

It combines a Flask API, MongoDB persistence, JWT authentication, and a React/Vite frontend.

> **Medical disclaimer:** This project is for educational/research purposes. A model prediction is not a medical diagnosis and should not replace assessment by a qualified clinician.

## Architecture

```text
skin-cancer/
├── backend/
│   ├── app.py          # Flask API and routes
│   ├── config.py       # Environment-driven configuration
│   ├── database.py     # MongoDB connection wrapper
│   └── model.py        # Model loading and image preprocessing
├── train.py            # Training with validation metrics
├── evaluate.py         # Held-out test-set evaluation
├── requirements.txt    # Python dependencies
├── .env.example        # Safe configuration template
├── .gitignore          # Local/generated files excluded from Git
├── README.md
└── frontend files       # React/Vite UI
```

The older root-level Flask files are retained for compatibility with the existing prototype. New backend work should go under `backend/`.

## Security improvements

- JWT secret is loaded from the environment rather than committed source code.
- The application refuses to start without a sufficiently long JWT secret.
- CORS is restricted to configured origins instead of allowing every origin.
- Prediction, dashboard, and hospital endpoints require authentication.
- Uploaded files are validated as images before inference.
- API errors return a generic message instead of exposing internal exception details.
- Request limits and coordinate validation reduce accidental or abusive inputs.
- Patient records are associated with the authenticated user for prediction records.
- `.env` and model artifacts are excluded from Git.

For production use, add rate limiting, stronger authorization/role controls, secure HTTPS deployment, audit logging, encryption and an appropriate privacy/data-retention policy before storing real patient information.

## Setup

### 1. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and set a unique secret:

```text
JWT_SECRET_KEY=<at-least-32-random-characters>
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=skin_disease_classifier
MODEL_PATH=skin_disease_model.pth
CORS_ORIGINS=http://localhost:5173
```

Do not commit `.env`.

### 4. MongoDB

Run a local MongoDB instance or provide a reachable MongoDB URI through `MONGO_URI`.

### 5. Model

The trained `.pth` file is intentionally not stored in Git. Train a model locally or obtain the approved model artifact separately and set `MODEL_PATH` accordingly.

### 6. Start the API

From the repository root:

```bash
python backend/app.py
```

The API listens on port `5000` by default.

## Dataset structure

Training requires a **separate validation set**. The training script deliberately refuses to use the training set as validation data because that would make the validation result misleading.

```text
dataset/
├── train/
│   ├── NORMAL_SKIN/
│   ├── PSORIASIS/
│   ├── Ringworm/
│   └── acne/
├── val/
│   ├── NORMAL_SKIN/
│   ├── PSORIASIS/
│   ├── Ringworm/
│   └── acne/
└── test/
    ├── NORMAL_SKIN/
    ├── PSORIASIS/
    ├── Ringworm/
    └── acne/
```

The class order must match the four labels above.

## Training

```bash
python train.py --data_dir ./dataset --epochs 10 --batch_size 16 --lr 0.0001 --save_path ./models/skin_disease_model.pth
```

The training script selects the best checkpoint using **macro-F1**, rather than accuracy alone. It records:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Confusion matrix
- Per-class classification report

The metrics are written beside the model as a `.metrics.json` file.

## Held-out evaluation

Use data that was not used for training or model selection:

```bash
python evaluate.py --data_dir ./dataset/test --model_path ./models/skin_disease_model.pth --output evaluation.metrics.json
```

Do not report validation accuracy as final model performance. For a serious evaluation, report the held-out test metrics and inspect the per-class confusion matrix.

## API overview

| Method | Endpoint | Auth |
|---|---|---|
| POST | `/api/auth/register` | No |
| POST | `/api/auth/login` | No |
| GET | `/api/user/profile` | JWT |
| POST | `/api/predict` | JWT |
| GET | `/api/dashboard/stats` | JWT |
| GET | `/api/dashboard/recent` | JWT |
| GET | `/api/hospitals/nearby` | JWT |
| GET | `/api/hospitals/search-city` | JWT |

## Project quality rules

1. Never commit virtual environments or generated executables.
2. Never commit `.env` files or secrets.
3. Keep model weights outside Git unless there is a deliberate artifact-storage strategy.
4. Keep train, validation and test data separate.
5. Use macro metrics for multi-class evaluation so minority classes are not hidden by overall accuracy.
6. Treat low-confidence predictions as uncertain; do not automatically convert them to `NORMAL_SKIN`.
7. Do not use this model as a standalone medical diagnostic system.

## License

Add the project's intended license before public distribution.
