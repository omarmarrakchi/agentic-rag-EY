"""
Interface avec ChromaDB.

Deux collections :
  - tdr_children : children avec embeddings → utilisés pour la recherche vectorielle
  - tdr_parents  : parents sans embeddings → récupérés par ID pour le contexte LLM

Les métadonnées listes (objectifs, livrables) sont converties en chaînes
car ChromaDB n'accepte que str, int, float ou bool comme valeurs de métadonnées.

HNSW ef_search=64 (défaut=10) — meilleur rappel au moment de la recherche.
"""

from typing import Optional

import chromadb
from pathlib import Path

from config.settings import VECTOR_DB_DIR, COLLECTION_CHILDREN, COLLECTION_PARENTS

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


def collection_stats() -> dict:
    """Retourne le nombre de documents dans chaque collection."""
    return {
        "children": get_children_collection().count(),
        "parents":  get_parents_collection().count(),
    }
