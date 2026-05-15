"""
Endpoint de recherche directe dans ChromaDB.
Retourne des TdRs structurés avec scores, métadonnées et missions similaires.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from config.settings import AGENT_TOP_K, AGENT_SCORE_THRESHOLD
from pipeline.step3_indexing.embedder import encode
from pipeline.step3_indexing.vector_store import search_children, get_parents_by_ids, get_all_parents
from pipeline.step4_agent.reranker import rerank

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = ""
    organisation: str = ""
    lieu: str = ""
    k: int = 12


class TdRResult(BaseModel):
    source: str
    score: float
    titre: str
    organisation: str
    lieu: str
    budget: str
    duree: str
    date_limite: str
    profil_consultant: str
    objectifs: list[str]
    livrables: list[str]


class SearchResponse(BaseModel):
    results: list[TdRResult]
    total: int


def _str(val) -> str:
    return str(val).strip() if val else ""


def _to_list(val) -> list[str]:
    if not val:
        return []
    if isinstance(val, list):
        return [str(v).strip() for v in val if v]
    s = str(val).strip()
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


@router.post("/search", response_model=SearchResponse)
def search_tdrs(req: SearchRequest):
    has_query   = bool(req.query.strip())
    has_filters = bool(req.organisation.strip() or req.lieu.strip())

    if not has_query and not has_filters:
        return SearchResponse(results=[], total=0)

    # ── Recherche par filtres uniquement (sans query) ─────────────────────
    if not has_query:
        all_p = get_all_parents(limit=500)
        seen: dict[str, dict] = {}
        for p in all_p:
            src = p["metadata"].get("source", "")
            if src and src not in seen:
                seen[src] = p

        results: list[TdRResult] = []
        for source, p in seen.items():
            meta = p["metadata"]
            org  = _str(meta.get("organisation", ""))
            lieu = _str(meta.get("lieu", ""))

            if req.organisation and req.organisation.lower() not in org.lower():
                continue
            if req.lieu and req.lieu.lower() not in lieu.lower():
                continue

            results.append(TdRResult(
                source=source,
                score=1.0,
                titre=_str(meta.get("titre", "")) or source,
                organisation=org,
                lieu=lieu,
                budget=_str(meta.get("budget", "")),
                duree=_str(meta.get("duree", "")),
                date_limite=_str(meta.get("date_limite", "")),
                profil_consultant=_str(meta.get("profil_consultant", "")),
                objectifs=_to_list(meta.get("objectifs", [])),
                livrables=_to_list(meta.get("livrables", [])),
            ))

        results = results[:req.k]
        return SearchResponse(results=results, total=len(results))

    # 1. Recherche vectorielle — on récupère beaucoup de candidats pour la déduplication
    query_embedding = encode([req.query])[0]
    hits = search_children(query_embedding, k=AGENT_TOP_K * 4)

    # 2. Filtre par score minimum
    filtered = [h for h in hits if h["score"] >= AGENT_SCORE_THRESHOLD]
    if not filtered:
        filtered = hits  # fallback si aucun résultat ne passe le seuil

    # 3. Reranking cross-encoder
    reranked = rerank(req.query, filtered)

    # 4. Déduplication par source — meilleur score par document
    best_by_source: dict[str, dict] = {}
    for hit in reranked:
        src = hit["metadata"].get("source", "")
        if src and src not in best_by_source:
            best_by_source[src] = hit

    # 5. Récupération des métadonnées complètes depuis la collection parents
    parent_ids = [
        h["metadata"].get("parent_id", "")
        for h in best_by_source.values()
        if h["metadata"].get("parent_id", "")
    ]
    parents = get_parents_by_ids(parent_ids[:40])

    # Map source → métadonnées (premier parent trouvé par source)
    meta_by_source: dict = {}
    for p in parents:
        src = p["metadata"].get("source", "")
        if src and src not in meta_by_source:
            meta_by_source[src] = p["metadata"]

    # 6. Construction des résultats + application des filtres
    results: list[TdRResult] = []
    for source, hit in best_by_source.items():
        meta = meta_by_source.get(source, hit["metadata"])

        org = _str(meta.get("organisation", ""))
        lieu = _str(meta.get("lieu", ""))

        # Filtres textuels (case-insensitive)
        if req.organisation and req.organisation.lower() not in org.lower():
            continue
        if req.lieu and req.lieu.lower() not in lieu.lower():
            continue

        results.append(TdRResult(
            source=source,
            score=round(hit["score"], 3),
            titre=_str(meta.get("titre", "")) or source,
            organisation=org,
            lieu=lieu,
            budget=_str(meta.get("budget", "")),
            duree=_str(meta.get("duree", "")),
            date_limite=_str(meta.get("date_limite", "")),
            profil_consultant=_str(meta.get("profil_consultant", "")),
            objectifs=_to_list(meta.get("objectifs", [])),
            livrables=_to_list(meta.get("livrables", [])),
        ))

    results = results[:req.k]
    return SearchResponse(results=results, total=len(results))
