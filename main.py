from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
import os

MODEL_PATH = os.path.expanduser("~/model_cache/onnx/model_quantized.onnx")
TOKENIZER_PATH = os.path.expanduser("~/model_cache/tokenizer.json")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
tokenizer.enable_truncation(max_length=256)
tokenizer.enable_padding(length=256)

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

app = FastAPI(title="API de Similarite")


class Textes(BaseModel):
    textes: list[str]


class Comparaison(BaseModel):
    texte1: str
    texte2: str


def get_embedding(text: str):
    encoded = tokenizer.encode("query: " + text)
    ort_inputs = {
        "input_ids": np.array([encoded.ids], dtype=np.int64),
        "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
    }
    input_names = [i.name for i in session.get_inputs()]
    if "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = np.array([encoded.type_ids], dtype=np.int64)

    last_hidden_state = session.run(None, ort_inputs)[0]
    mask = np.expand_dims(ort_inputs["attention_mask"], -1).astype(np.float32)
    sum_emb = np.sum(last_hidden_state * mask, axis=1)
    sum_mask = np.clip(mask.sum(axis=1), 1e-9, None)
    mean_pooled = sum_emb / sum_mask
    norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
    return (mean_pooled / np.clip(norm, 1e-9, None))[0]


@app.get("/")
def read_root():
    return {"status": "ok", "message": "API de similarite operationnelle"}


@app.post("/embed")
def embed(payload: Textes):
    vecteurs = [get_embedding(t).tolist() for t in payload.textes]
    return {"embeddings": vecteurs}


@app.post("/similarite")
def similarite(payload: Comparaison):
    a = get_embedding(payload.texte1)
    b = get_embedding(payload.texte2)
    score = float(np.dot(a, b))
    return {"score": round(score, 4)}
