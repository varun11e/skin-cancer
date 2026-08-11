# DermaAI — Skin Disease Classification

DermaAI is an educational/research web application that uses a PyTorch EfficientNet-B3 image-classification model to classify four skin-image categories:

- `NORMAL_SKIN`
- `PSORIASIS`
- `Ringworm`
- `acne`

It combines a Flask API, MongoDB persistence, JWT authentication, bcrypt password hashing, rate limiting, and a React/Vite frontend.

> **Medical disclaimer:** This project is for educational/research purposes. A model prediction is not a medical diagnosis and must not replace assessment by a qualified clinician.

## Architecture

```text
Browser
  │
  ▼
React/Vite frontend ──────────────┐
                                  │ same-origin /api
                                  ▼
                         Flask + Gunicorn API
                         ├── JWT authentication
                         ├── rate limiting
                         ├── image validation
                         ├── EfficientNet-B3 inference
                         └── MongoDB access
                                  │
                                  ▼
                              MongoDB Atlas

Model artifact (.pth)
  └── downloaded at startup from MODEL_URL and verified with MODEL_SHA256
```

The production container builds the React frontend and serves it from Flask, so the application can run as a single web service.

## Repository structure

```text
skin-cancer/
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── database.py
│   └── model.py
├── App.jsx
├── index.css
├── main.jsx
├── index.html
├── package.json
├── vite.config.js
├── train.py
├── evaluate.py
├── requirements.txt
├── Dockerfile
├── railway.toml
├── .env.example
├── .dockerignore
└── .github/workflows/python-checks.yml
```

## Production improvements

- JWT secrets are environment-only and require at least 32 characters.
- Passwords are hashed with bcrypt.
- Dashboard and prediction records are scoped to the authenticated user.
- CORS is configurable instead of hardcoded to allow every origin.
- Request bodies are limited to a configurable image size.
- Uploaded files must parse as valid images before inference.
- Login, registration, prediction, and global request rates are limited.
- Security response headers are added by default.
- MongoDB indexes are created for common queries.
- MongoDB connection timeouts are configured.
- Model artifacts stay outside Git and can be downloaded from a configured artifact URL.
- Model downloads can be verified using SHA-256.
- Gunicorn is used instead of Flask's development server.
- The frontend and API are packaged into one production container.
- Railway health checks use `/health`.
- CI compiles Python, builds the frontend, and builds the production Docker image.

## Local development

### Backend

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Create `.env` from `.env.example` and set a local MongoDB URI and JWT secret.

You also need the trained model:

```text
MODEL_PATH=models/skin_disease_model.pth
```

or configure `MODEL_URL` to a trusted model artifact URL.

Start the API:

```bash
python -m backend.app
```

### Frontend

```bash
npm install
npm run dev
```

The Vite development server runs on its normal local port. Set `VITE_API_URL` only when the API is hosted separately; for the production container, leave it empty so requests use the same origin.

## Training

Training requires separate train and validation datasets:

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

Train:

```bash
python train.py --data_dir ./dataset --epochs 10 --batch_size 16 --lr 0.0001 --save_path ./models/skin_disease_model.pth
```

The best checkpoint is selected by macro-F1. Accuracy, macro precision, macro recall, macro-F1, a confusion matrix, and a per-class classification report are written to the metrics JSON file.

Evaluate only on a held-out test set:

```bash
python evaluate.py --data_dir ./dataset/test --model_path ./models/skin_disease_model.pth --output evaluation.metrics.json
```

Do not report validation accuracy as final model performance.

## API

| Method | Endpoint | Auth |
|---|---|---|
| GET | `/health` | No |
| GET | `/api` | No |
| POST | `/api/auth/register` | No |
| POST | `/api/auth/login` | No |
| GET | `/api/user/profile` | JWT |
| GET | `/api/dashboard/stats` | JWT |
| GET | `/api/dashboard/recent` | JWT |
| POST | `/api/predict` | JWT |
| GET | `/api/hospitals/nearby` | JWT |
| GET | `/api/hospitals/search-city` | JWT |

## Deployment — Railway + MongoDB Atlas

Railway is the recommended first deployment target for this repository because the application is containerized, needs more memory than a tiny serverless function, and Railway supports Dockerfile deployments, configurable health checks, and vertical scaling. The included `railway.toml` is ready for a GitHub-connected service.

### 1. Prepare the model artifact

Do **not** commit `skin_disease_model.pth`; `.gitignore` deliberately excludes model weights.

Upload the approved trained model to trusted object storage or an artifact host that provides a stable HTTPS URL. Set:

```text
MODEL_URL=https://your-trusted-host.example/skin_disease_model.pth
MODEL_SHA256=<64-character-sha256>
```

The application downloads the artifact only when it is missing and verifies the checksum before loading it.

### 2. Create MongoDB Atlas

Create a MongoDB deployment and database user, restrict network access appropriately, and obtain its connection string.

Set:

```text
MONGO_URI=mongodb+srv://...
MONGO_DB_NAME=skin_disease_classifier
```

### 3. Create the Railway service

Connect this GitHub repository to a new Railway service. Railway will detect `Dockerfile` and `railway.toml`.

Set these service variables:

```text
JWT_SECRET_KEY=<at-least-32-random-characters>
MONGO_URI=<your-atlas-connection-string>
MONGO_DB_NAME=skin_disease_classifier
MODEL_URL=<your-model-url>
MODEL_SHA256=<your-model-sha256>
MAX_UPLOAD_MB=8
DISEASE_CONFIDENCE_THRESHOLD=0.55
JWT_ACCESS_TOKEN_HOURS=24
RATELIMIT_STORAGE_URI=memory://
FLASK_DEBUG=false
```

Do not set a fixed `PORT`; Railway supplies it.

### 4. Resource recommendation

EfficientNet-B3 has about 12.2 million parameters and a roughly 47 MB reference weight file before accounting for the Python/PyTorch runtime and application memory. Use a paid Railway service with sufficient RAM rather than a 0.5 GB hobby allocation for reliable production inference.

Start with one replica and at least 2 GB RAM. Increase RAM/CPU after observing real inference latency and memory usage.

### 5. Health check

Railway should use:

```text
/health
```

The endpoint verifies MongoDB connectivity and confirms that the model is loaded.

### 6. Domain and HTTPS

Use the Railway-generated domain for initial testing. Add a custom domain after the application passes production checks. Railway terminates HTTPS at the platform edge.

## Production checklist

Before using real patient information:

- [ ] Use only synthetic/test patient data until a formal privacy and security review is complete.
- [ ] Store the model in trusted artifact storage and set `MODEL_SHA256`.
- [ ] Use a strong random `JWT_SECRET_KEY`.
- [ ] Restrict MongoDB Atlas network access.
- [ ] Configure a production rate-limit backend such as Redis when running multiple replicas.
- [ ] Set `CORS_ORIGINS` if the frontend is hosted separately.
- [ ] Configure database backups and test restoration.
- [ ] Add monitoring and alerting.
- [ ] Add automated API/integration tests before enabling continuous production deploys.
- [ ] Review retention, deletion, encryption, access-control, and audit requirements for patient information.
- [ ] Obtain appropriate clinical validation before making any medical claim.

## Security and privacy

This application is not designed to store real patient data without additional controls. A production healthcare deployment should add appropriate encryption, authorization/roles, audit logging, retention/deletion policies, threat modeling, monitoring, and regulatory/privacy review.
