FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Cache models inside the container so restarts don't re-download
ENV HF_HOME=/app/.cache/huggingface

# System deps for Pillow, torch runtime, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 git wget && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x start.sh

# Render will set the PORT env variable automatically
CMD ["./start.sh"]