"""Point d'entrée pour lancer le serveur FastAPI."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Empêche PyTorch (reranker) de prendre le GPU — réservé à Ollama
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
