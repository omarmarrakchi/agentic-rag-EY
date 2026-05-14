"""
Outils ReAct de l'agent RAG.

Pipeline de recherche :
  1. BGE-M3 (vector search) → AGENT_TOP_K=6 candidats
  2. Filtre score >= AGENT_SCORE_THRESHOLD
  3. Cross-encoder reranker → re-trie par pertinence réelle
  4. Retourne RERANKER_TOP_K=3 meilleurs résultats
  5. retrieve_parent_chunks → contexte complet pour le LLM
"""

from langchain_core.tools import tool

from config.settings import AGENT_TOP_K, AGENT_SCORE_THRESHOLD, RERANKER_TOP_K
from pipeline.step3_indexing.embedder import encode
from pipeline.step3_indexing.vector_store import search_children, get_parents_by_ids
from pipeline.step4_agent.reranker import rerank


@tool
def search_child_chunks(query: str) -> str:
    """
    Searches for relevant passages in the TdR database based on a query.
    Always use this tool first to find information about TdRs.
    Returns the most relevant text passages with their source document and parent ID.
    """
    query_embedding = encode([query])[0]
    results = search_children(query_embedding, k=AGENT_TOP_K)

    if not results:
        return "Aucun résultat trouvé pour cette requête."

    # Filtre les résultats trop faibles avant le reranking
    filtered = [r for r in results if r["score"] >= AGENT_SCORE_THRESHOLD]
    if not filtered:
        best = results[0]["score"]
        return (
            f"Résultats trouvés mais scores trop faibles (meilleur: {best:.2f}). "
            "Essaie avec des mots-clés différents."
        )

    # Reranking : cross-encoder re-trie par pertinence réelle
    reranked = rerank(query, filtered)
    top = reranked[:RERANKER_TOP_K]

    output = []
    for r in top:
        output.append(
            f"[Source: {r['metadata'].get('source', '')} | "
            f"Score: {r['rerank_score']:.2f} | "
            f"Parent ID: {r['metadata'].get('parent_id', '')}]\n"
            f"{r['text']}"
        )
    return "\n\n---\n\n".join(output)


@tool
def retrieve_parent_chunks(parent_ids: str) -> str:
    """
    Retrieves the full context of parent chunks by their IDs.
    Use this after search_child_chunks to get more complete context before answering.
    parent_ids must be a comma-separated list of parent chunk IDs
    (e.g. "TDR_ERP_p3, TDR_ERP_p5").
    """
    ids = [pid.strip() for pid in parent_ids.split(",") if pid.strip()]
    if not ids:
        return "Aucun ID fourni."

    parents = get_parents_by_ids(ids)
    if not parents:
        return "Parents non trouvés pour ces IDs."

    output = []
    for p in parents:
        output.append(
            f"[Source: {p['metadata'].get('source', '')} | "
            f"Titre: {p['metadata'].get('titre', '')}]\n"
            f"{p['text']}"
        )
    return "\n\n---\n\n".join(output)
