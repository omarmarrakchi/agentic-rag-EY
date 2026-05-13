"""
Charge le modèle BAAI/bge-m3 et génère les embeddings par batch.

BGE-M3 produit des vecteurs de 1 024 dimensions.
Le modèle est chargé une seule fois (singleton) pour éviter de le
recharger à chaque appel.
"""

from typing import Optional

from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
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
        normalize_embeddings=True,  # normalisation L2 — améliore la similarité cosine
    )
    return embeddings.tolist()
