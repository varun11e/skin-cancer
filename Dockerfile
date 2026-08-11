FROM node:22-bookworm-slim AS frontend-build
WORKDIR /frontend
COPY package.json vite.config.js index.html main.jsx App.jsx index.css ./
RUN npm install --no-audit --no-fund
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3" "torchvision>=0.17,<1" \
    && grep -Ev '^(torch|torchvision)' requirements.txt > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY backend ./backend
COPY train.py evaluate.py ./
COPY --from=frontend-build /frontend/dist ./dist
COPY .env.example ./

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 CMD curl -fsS http://127.0.0.1:${PORT:-5000}/health || exit 1

CMD ["sh", "-c", "gunicorn --workers 1 --threads 4 --timeout 180 --access-logfile - --error-logfile - --bind 0.0.0.0:${PORT:-5000} backend.app:app"]
