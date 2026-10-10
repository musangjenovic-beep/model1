from huggingface_hub import hf_hub_download
import os

os.makedirs("/app/model_cache", exist_ok=True)

# Modèle multilingual-e5-small en INT8 (~118 Mo)
hf_hub_download(
    repo_id="Xenova/multilingual-e5-small",
    filename="onnx/model_quantized.onnx",
    local_dir="/app/model_cache",
)

hf_hub_download(
    repo_id="Xenova/multilingual-e5-small",
    filename="tokenizer.json",
    local_dir="/app/model_cache",
)

print("Modele pre-telecharge avec succes")
