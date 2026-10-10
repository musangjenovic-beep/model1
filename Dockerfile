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

# Pré-téléchargement avec add_custom_model
RUN python -c "
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource
TextEmbedding.add_custom_model(
    model='intfloat/multilingual-e5-small',
    pooling=PoolingType.MEAN,
    normalization=True,
    sources=ModelSource(hf='intfloat/multilingual-e5-small'),
    dim=384,
    model_file='onnx/model.onnx',
)
TextEmbedding(model_name='intfloat/multilingual-e5-small')
print('Modele pre-telecharge avec succes')
"

EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
