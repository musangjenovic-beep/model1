FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système pour onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY main.py .

# --- TRÈS IMPORTANT : Pré-téléchargement du modèle ---
# Cela évite de le télécharger à chaque démarrage et surtout, cela réduit le temps de démarrage à froid.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='ibm-granite/granite-embedding-97m-multilingual-r2')"

# Render définit la variable d'environnement PORT
EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
