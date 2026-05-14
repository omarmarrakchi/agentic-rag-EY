"""
Charge le modèle BAAI/bge-m3 et génère les embeddings par batch.

BGE-M3 produit des vecteurs de 1 024 dimensions.

Stratégie GPU/CPU :
  - Variable d'environnement EMBEDDER_DEVICE=cuda  → force GPU (indexation)
  - Variable d'environnement EMBEDDER_DEVICE=cpu   → force CPU (API serving)
  - Par défaut (non défini)                        → CPU

Pendant le serving API, on utilise CPU pour l'encodage des requêtes
(1 seul texte, ~30ms) afin de laisser le GPU à Ollama/qwen2.5:14b.
Pendant run_indexing.py, on force EMBEDDER_DEVICE=cuda pour le batch.
"""

import os
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

_model: Optional[SentenceTransformer] = None
_device: Optional[str] = None


def get_device() -> str:
    """
    Retourne le device à utiliser.
    Priorité : variable EMBEDDER_DEVICE > détection automatique (CPU par défaut).
    """
    env = os.environ.get("EMBEDDER_DEVICE", "").lower()
    if env == "cuda" and torch.cuda.is_available():
        return "cuda"
    if env == "cpu":
        return "cpu"
    # Par défaut : CPU pour éviter le conflit VRAM avec Ollama
    return "cpu"


def _get_model() -> SentenceTransformer:
    global _model, _device
    if _model is None:
        _device = get_device()
        _model = SentenceTransformer(EMBEDDING_MODEL, device=_device)
    return _model


def encode(texts: list[str]) -> list[list[float]]:
    """
    Encode une liste de textes en vecteurs.
    Traite les textes par batch de EMBEDDING_BATCH_SIZE.
    Retourne une liste de vecteurs (chaque vecteur = liste de 1024 floats).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()
