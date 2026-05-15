"""
Reranker cross-encoder pour affiner les résultats de la recherche vectorielle.

Fonctionnement :
  1. BGE-M3 (vector search) récupère AGENT_TOP_K=6 candidats rapidement
  2. Le cross-encoder lit (query, chunk) ensemble et donne un score de pertinence précis
  3. On retrie par ce score et on garde RERANKER_TOP_K=3 meilleurs

Modèle : BAAI/bge-reranker-base
  - Multilingue (français, anglais, arabe)
  - ~278MB, tourne sur CPU
  - Même famille que BGE-M3 — compatibilité optimale
"""

from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config.settings import RERANKER_MODEL

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSequenceClassification] = None


def _get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        # dtype=float32 évite l'erreur "meta tensor" avec PyTorch 2.6
        _model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL, dtype=torch.float32
        )
        _model.eval()
    return _tokenizer, _model


def rerank(query: str, results: list[dict]) -> list[dict]:
    """
    Re-trie une liste de résultats ChromaDB par pertinence réelle.

    Paramètres :
      query   : question de l'utilisateur
      results : liste de dicts avec clés 'text', 'score', 'metadata', 'chunk_id'

    Retourne la liste triée du plus pertinent au moins pertinent,
    avec le champ 'rerank_score' ajouté à chaque résultat.
    """
    if not results:
        return results

    tokenizer, model = _get_model()

    pairs = [[query, r["text"]] for r in results]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        scores = model(**inputs).logits.squeeze(-1).tolist()

    if isinstance(scores, float):
        scores = [scores]

    for i, r in enumerate(results):
        r["rerank_score"] = float(scores[i])

    return sorted(results, key=lambda r: r["rerank_score"], reverse=True)
