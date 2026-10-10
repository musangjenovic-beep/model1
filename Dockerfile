FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cargo \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV FASTEMBED_CACHE_PATH=/app/model_cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY preload.py .

RUN python preload.py

EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
