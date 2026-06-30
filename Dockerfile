# Production image for the Clinical Trial Outcome Predictor API.
# Build:  docker build -t trials-api .
# Run:    docker run -p 8000:8000 trials-api
FROM python:3.11-slim

# --- Environment: fixes the FAISS/PyTorch OpenMP segfault + forces offline models ---
ENV KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps occasionally needed by faiss/numpy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model INTO the image so runtime is fully offline.
# (Runs while network is available during build; cached under /root/.cache.)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the application code and artifacts
COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000

# Container-level health check hits the API's health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fs http://localhost:8000/ || exit 1

# Bind to 0.0.0.0 so the port is reachable from outside the container.
# Honor the platform's $PORT if provided (Render/Railway set it), else 8000.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
