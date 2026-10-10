from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding
import numpy as np

# On charge le modèle une seule fois au démarrage
print("Chargement du modèle d'embedding...")
model = TextEmbedding(model_name="ibm-granite/granite-embedding-97m-multilingual-r2")
print("Modèle chargé !")

app = FastAPI(title="API de Similarité")

class Textes(BaseModel):
    textes: list[str]

class Comparaison(BaseModel):
    texte1: str
    texte2: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API de similarité opérationnelle"}

@app.post("/embed")
def embed(payload: Textes):
    """Génère les embeddings pour une liste de textes."""
    vecteurs = list(model.embed(payload.textes))
    # On convertit en listes Python pour le JSON
    return {"embeddings": [v.tolist() for v in vecteurs]}

@app.post("/similarite")
def similarite(payload: Comparaison):
    """Calcule la similarité cosinus entre deux textes."""
    embeddings = list(model.embed([payload.texte1, payload.texte2]))
    a, b = np.array(embeddings[0]), np.array(embeddings[1])
    score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return {"score": round(score, 4)}
