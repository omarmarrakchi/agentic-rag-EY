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

from sentence_transformers import CrossEncoder

from config.settings import RERANKER_MODEL

_model: Optional[CrossEncoder] = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        # CPU uniquement — GPU réservé à Ollama
        _model = CrossEncoder(RERANKER_MODEL, device="cpu")
    return _model


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

    model = _get_model()

    pairs = [[query, r["text"]] for r in results]
    scores = model.predict(pairs, show_progress_bar=False)

    for i, r in enumerate(results):
        r["rerank_score"] = float(scores[i])

    return sorted(results, key=lambda r: r["rerank_score"], reverse=True)
