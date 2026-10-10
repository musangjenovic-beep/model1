from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource
import numpy as np

TextEmbedding.add_custom_model(
    model="intfloat/multilingual-e5-small",
    pooling=PoolingType.MEAN,
    normalization=True,
    sources=ModelSource(hf="intfloat/multilingual-e5-small"),
    dim=384,
    model_file="onnx/model.onnx",
)

print("Chargement du modele d'embedding...")
model = TextEmbedding(model_name="intfloat/multilingual-e5-small")
print("Modele charge !")

app = FastAPI(title="API de Similarite")

class Textes(BaseModel):
    textes: list[str]

class Comparaison(BaseModel):
    texte1: str
    texte2: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API de similarite operationnelle"}

@app.post("/embed")
def embed(payload: Textes):
    vecteurs = list(model.embed(payload.textes))
    return {"embeddings": [v.tolist() for v in vecteurs]}

@app.post("/similarite")
def similarite(payload: Comparaison):
    embeddings = list(model.embed([payload.texte1, payload.texte2]))
    a, b = np.array(embeddings[0]), np.array(embeddings[1])
    score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return {"score": round(score, 4)}
