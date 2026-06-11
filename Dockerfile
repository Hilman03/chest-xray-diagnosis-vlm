# Backend — FastAPI + AI models
FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV + pydicom
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/   ./backend/
COPY models/    ./models/
COPY pipeline.py .

# HuggingFace model cache (persisted via volume in docker-compose)
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
