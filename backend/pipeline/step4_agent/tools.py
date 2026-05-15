"""
Outils ReAct de l'agent RAG.

Outils disponibles :
  1. search_child_chunks      — recherche hybride (BGE-M3 + BM25 + RRF) + reranking
  2. retrieve_parent_chunks   — récupère le contexte complet par IDs
  3. count_documents          — nombre total de TdRs dans la base
  4. list_all_documents       — liste tous les TdRs avec métadonnées
  5. filter_documents         — filtre par organisation, lieu, budget, durée
  6. get_document_details     — fiche complète d'un TdR spécifique
"""

from langchain_core.tools import tool

from config.settings import AGENT_TOP_K, AGENT_SCORE_THRESHOLD, RERANKER_TOP_K
from pipeline.step3_indexing.embedder import encode
from pipeline.step3_indexing.vector_store import (
    hybrid_search,
    get_parents_by_ids,
    count_unique_documents,
    list_all_documents      as _list_all_documents,
    filter_documents_by_metadata,
    get_document_details    as _get_document_details,
)
from pipeline.step4_agent.reranker import rerank


@tool
def search_child_chunks(query: str) -> str:
    """
    Searches for relevant passages in the TdR database based on a query.
    Always use this tool first to find information about TdRs.
    Returns the most relevant text passages with their source document and parent ID.
    """
    query_embedding = encode([query])[0]
    results = hybrid_search(query_embedding, query, k=AGENT_TOP_K)

    if not results:
        return "Aucun résultat trouvé pour cette requête."

    # Les scores RRF sont dans [0, ~0.03] — on garde tous les résultats
    filtered = results if results else []
    if not filtered:
        return "Aucun résultat pertinent trouvé. Essaie avec des mots-clés différents."

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


@tool
def count_documents(query: str = "") -> str:
    """
    Returns the total number of TdRs in the database.
    Use this when the user asks how many TdRs exist (e.g. 'how many TdRs?', 'combien de TdRs?').
    The query parameter is ignored — always returns the total count.
    """
    count = count_unique_documents()
    return f"La base contient {count} TdR(s) au total."


@tool
def list_all_documents(query: str = "") -> str:
    """
    Lists all TdRs in the database with their main metadata (title, organisation, location, duration, budget).
    Use this when the user asks to list all TdRs or wants an overview of available documents.
    """
    docs = _list_all_documents()
    if not docs:
        return "Aucun TdR trouvé dans la base."

    output = [f"{len(docs)} TdR(s) disponibles :\n"]
    for i, doc in enumerate(docs, 1):
        line = f"{i}. [{doc['source']}]"
        if doc["titre"]:
            line += f" — {doc['titre']}"
        if doc["organisation"]:
            line += f" | Org: {doc['organisation']}"
        if doc["lieu"]:
            line += f" | Lieu: {doc['lieu']}"
        if doc["budget"]:
            line += f" | Budget: {doc['budget']}"
        output.append(line)
    return "\n".join(output)


@tool
def filter_documents(filters: str) -> str:
    """
    Filters TdRs by metadata fields. Use this when the user asks about specific organisations,
    locations, budgets or durations (e.g. 'TdRs from UNICEF', 'TdRs in Morocco', 'TdRs with budget').
    filters must be a comma-separated list of key:value pairs.
    Available keys: organisation, lieu, budget, duree.
    Example: "organisation:UNICEF, lieu:Maroc"
    """
    org = lieu = budget = duree = ""
    for part in filters.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            k = k.strip().lower()
            v = v.strip()
            if k == "organisation":
                org = v
            elif k == "lieu":
                lieu = v
            elif k == "budget":
                budget = v
            elif k == "duree":
                duree = v

    docs = filter_documents_by_metadata(
        organisation=org, lieu=lieu, budget=budget, duree=duree
    )
    if not docs:
        return f"Aucun TdR trouvé pour les filtres : {filters}"

    output = [f"{len(docs)} TdR(s) trouvé(s) :\n"]
    for i, doc in enumerate(docs, 1):
        line = f"{i}. [{doc['source']}]"
        if doc["titre"]:
            line += f" — {doc['titre']}"
        if doc["organisation"]:
            line += f" | Org: {doc['organisation']}"
        if doc["lieu"]:
            line += f" | Lieu: {doc['lieu']}"
        if doc["budget"]:
            line += f" | Budget: {doc['budget']}"
        if doc["duree"]:
            line += f" | Durée: {doc['duree']}"
        output.append(line)
    return "\n".join(output)


@tool
def get_document_details(source: str) -> str:
    """
    Returns all metadata of a specific TdR by its filename or title.
    Use this when the user asks for complete details about a specific TdR.
    source must be the filename or part of the filename (e.g. 'TDR-ERP', 'rapport_consultant').
    """
    doc = _get_document_details(source)
    if not doc:
        return f"Aucun TdR trouvé pour : {source}"

    lines = [f"Fiche complète — {doc['source']}\n"]
    fields = [
        ("Titre",               "titre"),
        ("Organisation",        "organisation"),
        ("Lieu",                "lieu"),
        ("Durée",               "duree"),
        ("Budget",              "budget"),
        ("Date limite",         "date_limite"),
        ("Profil consultant",   "profil_consultant"),
        ("Objectifs",           "objectifs"),
        ("Livrables",           "livrables"),
    ]
    for label, key in fields:
        val = doc.get(key)
        if val:
            lines.append(f"• {label} : {val}")
    return "\n".join(lines)
