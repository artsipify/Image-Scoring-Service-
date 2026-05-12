FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pip dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Set Hugging Face cache dirs so we can pre-download model during build
ENV HF_HOME=/models/huggingface-cache
ENV TRANSFORMERS_CACHE=/models/huggingface-cache/transformers
RUN mkdir -p /models/huggingface-cache

# Pre-download the sentence-transformers CLIP model into the image cache
RUN python - <<'PY'
from huggingface_hub import snapshot_download
print('Downloading sentence-transformers/clip-ViT-B-32')
snapshot_download(repo_id='sentence-transformers/clip-ViT-B-32', cache_dir='/models/huggingface-cache')
print('Download completed')
PY

# Copy app
COPY . /app

ENV PORT=8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
