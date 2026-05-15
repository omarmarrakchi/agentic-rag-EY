"""
Interface avec ChromaDB + index BM25 pour la recherche hybride.

Deux collections ChromaDB :
  - tdr_children : children avec embeddings → utilisés pour la recherche vectorielle
  - tdr_parents  : parents sans embeddings → récupérés par ID pour le contexte LLM

Index BM25 (rank_bm25) :
  - Construit en RAM au premier appel depuis les children ChromaDB
  - Recherche lexicale (mots exacts) — complémentaire à la recherche vectorielle
  - Fusionné via RRF (Reciprocal Rank Fusion) avec les résultats vectoriels

HNSW ef_search=64 (défaut=10) — meilleur rappel au moment de la recherche.
"""

import re
from typing import Optional

import chromadb
from pathlib import Path

from config.settings import VECTOR_DB_DIR, COLLECTION_CHILDREN, COLLECTION_PARENTS

# ── Index BM25 (lazy-loaded) ─────────────────────────────────────────────────
_bm25_index   = None
_bm25_chunks  = []   # liste de dicts {chunk_id, text, metadata} dans le même ordre que l'index

_client: Optional[chromadb.PersistentClient] = None

# ef_search : nombre de candidats explorés pendant la recherche HNSW.
# Plus élevé = meilleur rappel, légèrement plus lent.
# 64 est un bon compromis (défaut ChromaDB = 10).
_HNSW_EF_SEARCH = 64

_CHROMA_BATCH_SIZE = 2000


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    return _client


def _sanitize_metadata(meta: dict) -> dict:
    """Convertit les listes en chaînes pour respecter les contraintes ChromaDB."""
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, list):
            clean[k] = ", ".join(str(i) for i in v)
        else:
            clean[k] = v
    return clean


def get_children_collection() -> chromadb.Collection:
    return _get_client().get_or_create_collection(
        name=COLLECTION_CHILDREN,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:M": 16,
            "hnsw:construction_ef": 100,
            "hnsw:search_ef": _HNSW_EF_SEARCH,
        },
    )


def get_parents_collection() -> chromadb.Collection:
    return _get_client().get_or_create_collection(name=COLLECTION_PARENTS)


def clear_collections() -> None:
    """
    Supprime et recrée les deux collections.
    À appeler avant une re-indexation complète pour éviter
    les chunks obsolètes (anciens IDs qui n'existent plus).
    """
    client = _get_client()
    for name in [COLLECTION_CHILDREN, COLLECTION_PARENTS]:
        try:
            client.delete_collection(name)
        except Exception:
            pass


def _upsert_in_batches(collection, ids, embeddings, documents, metadatas) -> None:
    """Insère des données dans ChromaDB par batch pour éviter la limite de taille."""
    for start in range(0, len(ids), _CHROMA_BATCH_SIZE):
        end = start + _CHROMA_BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


def insert_children(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """Insère les children avec leurs embeddings dans ChromaDB."""
    collection = get_children_collection()
    clean_metas = [_sanitize_metadata(m) for m in metadatas]
    _upsert_in_batches(collection, ids, embeddings, documents, clean_metas)


def insert_parents(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """Insère les parents SANS embedding — récupérés uniquement par ID."""
    collection = get_parents_collection()
    clean_metas = [_sanitize_metadata(m) for m in metadatas]
    dummy_embeddings = [[0.0] * 1024] * len(ids)
    _upsert_in_batches(collection, ids, dummy_embeddings, documents, clean_metas)


def search_children(query_embedding: list[float], k: int = 5) -> list[dict]:
    """
    Cherche les k children les plus proches du vecteur requête.
    Retourne une liste de dicts avec : chunk_id, text, score, metadata.
    """
    collection = get_children_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text":     results["documents"][0][i],
            "score":    1 - results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        })
    return hits


def _tokenize(text: str) -> list[str]:
    """Tokenisation simple : minuscules + split sur les caractères non-alphanumériques."""
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())


def _get_bm25_index():
    """Construit l'index BM25 depuis ChromaDB (lazy — une seule fois au démarrage)."""
    global _bm25_index, _bm25_chunks
    if _bm25_index is not None:
        return _bm25_index

    from rank_bm25 import BM25Okapi

    collection = get_children_collection()
    results = collection.get(include=["documents", "metadatas"])

    _bm25_chunks = []
    corpus = []
    for i in range(len(results["ids"])):
        _bm25_chunks.append({
            "chunk_id": results["ids"][i],
            "text":     results["documents"][i],
            "metadata": results["metadatas"][i],
            "score":    0.0,
        })
        corpus.append(_tokenize(results["documents"][i]))

    _bm25_index = BM25Okapi(corpus)
    return _bm25_index


def search_children_bm25(query: str, k: int = 10) -> list[dict]:
    """
    Recherche BM25 (lexicale) dans les children.
    Retourne les k meilleurs résultats avec score normalisé [0, 1].
    """
    index = _get_bm25_index()
    tokens = _tokenize(query)
    scores = index.get_scores(tokens)

    # Associe chaque score à son chunk
    scored = [(score, chunk) for score, chunk in zip(scores, _bm25_chunks)]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]

    max_score = top[0][0] if top and top[0][0] > 0 else 1.0
    results = []
    for score, chunk in top:
        if score <= 0:
            break
        c = dict(chunk)
        c["score"] = score / max_score   # normalise [0, 1]
        results.append(c)
    return results


def hybrid_search(query_embedding: list[float], query: str, k: int = 10) -> list[dict]:
    """
    Recherche hybride : vectorielle (BGE-M3) + lexicale (BM25) fusionnées via RRF.

    RRF (Reciprocal Rank Fusion) :
      score_rrf(doc) = Σ 1 / (k_rrf + rang(doc))
      → favorise les documents bien classés dans LES DEUX recherches
    """
    K_RRF = 60   # constante RRF standard

    vector_results = search_children(query_embedding, k=k)
    bm25_results   = search_children_bm25(query, k=k)

    # Calcul des scores RRF
    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict]  = {}

    for rank, result in enumerate(vector_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (K_RRF + rank + 1)
        chunk_map[cid] = result

    for rank, result in enumerate(bm25_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (K_RRF + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = result

    # Trie par score RRF décroissant
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    fused = []
    for cid in sorted_ids[:k]:
        chunk = dict(chunk_map[cid])
        chunk["score"] = rrf_scores[cid]
        fused.append(chunk)
    return fused


def get_parents_by_ids(parent_ids: list[str]) -> list[dict]:
    """
    Récupère les parents par leurs IDs.
    Retourne une liste de dicts avec : chunk_id, text, metadata.
    """
    collection = get_parents_collection()
    results = collection.get(
        ids=parent_ids,
        include=["documents", "metadatas"],
    )

    parents = []
    for i in range(len(results["ids"])):
        parents.append({
            "chunk_id": results["ids"][i],
            "text":     results["documents"][i],
            "metadata": results["metadatas"][i],
        })
    return parents


def get_all_parents(limit: int = 500) -> list[dict]:
    """Récupère tous les parents — utilisé pour la recherche par filtres sans query."""
    collection = get_parents_collection()
    results = collection.get(limit=limit, include=["documents", "metadatas"])
    parents = []
    for i in range(len(results["ids"])):
        parents.append({
            "chunk_id": results["ids"][i],
            "text":     results["documents"][i],
            "metadata": results["metadatas"][i],
        })
    return parents


def collection_stats() -> dict:
    """Retourne le nombre de documents dans chaque collection."""
    return {
        "children": get_children_collection().count(),
        "parents":  get_parents_collection().count(),
    }


def count_unique_documents() -> int:
    """Retourne le nombre de TdRs uniques dans la base (basé sur le champ source)."""
    collection = get_parents_collection()
    results = collection.get(include=["metadatas"])
    sources = {m.get("source", "") for m in results["metadatas"] if m.get("source")}
    return len(sources)


def list_all_documents() -> list[dict]:
    """
    Retourne la liste de tous les TdRs avec leurs métadonnées principales.
    Dédupliqué par source — un seul enregistrement par TdR.
    """
    collection = get_parents_collection()
    results = collection.get(include=["metadatas"])
    seen = set()
    docs = []
    for meta in results["metadatas"]:
        source = meta.get("source", "")
        if source and source not in seen:
            seen.add(source)
            docs.append({
                "source":       source,
                "titre":        meta.get("titre", ""),
                "organisation": meta.get("organisation", ""),
                "lieu":         meta.get("lieu", ""),
                "duree":        meta.get("duree", ""),
                "budget":       meta.get("budget", ""),
                "date_limite":  meta.get("date_limite", ""),
            })
    return sorted(docs, key=lambda x: x["source"])


def filter_documents_by_metadata(
    organisation: str = "",
    lieu: str = "",
    budget: str = "",
    duree: str = "",
) -> list[dict]:
    """
    Filtre les TdRs par champs de métadonnées (recherche insensible à la casse).
    Retourne les TdRs dont les métadonnées contiennent les valeurs demandées.
    """
    collection = get_parents_collection()
    results = collection.get(include=["metadatas"])

    filters = {
        "organisation": organisation.lower().strip(),
        "lieu":         lieu.lower().strip(),
        "budget":       budget.lower().strip(),
        "duree":        duree.lower().strip(),
    }
    active_filters = {k: v for k, v in filters.items() if v}

    seen = set()
    docs = []
    for meta in results["metadatas"]:
        source = meta.get("source", "")
        if not source or source in seen:
            continue
        match = all(
            filters[field] in (meta.get(field) or "").lower()
            for field in active_filters
        )
        if match:
            seen.add(source)
            docs.append({
                "source":              source,
                "titre":               meta.get("titre", ""),
                "organisation":        meta.get("organisation", ""),
                "lieu":                meta.get("lieu", ""),
                "duree":               meta.get("duree", ""),
                "budget":              meta.get("budget", ""),
                "date_limite":         meta.get("date_limite", ""),
                "profil_consultant":   meta.get("profil_consultant", ""),
                "objectifs":           meta.get("objectifs", ""),
                "livrables":           meta.get("livrables", ""),
            })
    return docs


def get_document_details(source: str) -> dict | None:
    """
    Retourne toutes les métadonnées d'un TdR spécifique par son nom de fichier.
    Recherche insensible à la casse et aux espaces.
    """
    collection = get_parents_collection()
    results = collection.get(include=["metadatas"])

    source_clean = source.lower().strip()
    for meta in results["metadatas"]:
        if source_clean in (meta.get("source") or "").lower():
            return {
                "source":            meta.get("source", ""),
                "titre":             meta.get("titre", ""),
                "organisation":      meta.get("organisation", ""),
                "lieu":              meta.get("lieu", ""),
                "duree":             meta.get("duree", ""),
                "budget":            meta.get("budget", ""),
                "date_limite":       meta.get("date_limite", ""),
                "profil_consultant": meta.get("profil_consultant", ""),
                "objectifs":         meta.get("objectifs", ""),
                "livrables":         meta.get("livrables", ""),
            }
    return None
