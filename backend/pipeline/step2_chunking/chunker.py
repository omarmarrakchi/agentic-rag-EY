"""
Découpe un texte en chunks hiérarchiques parent-child.

Parent chunk : grande fenêtre (~800 chars) — contexte riche pour le LLM.
Child chunk  : petite fenêtre (~200 chars) — précis pour la recherche vectorielle.

Chaque child référence son parent via parent_id.
Chaque parent liste ses children via le champ children[].
"""

import re

from config.settings import (
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
)


def _sanitize_name(filename_stem: str) -> str:
    """Transforme le nom de fichier en identifiant valide (sans espaces ni caractères spéciaux)."""
    sanitized = re.sub(r"[^\w]", "_", filename_stem)
    return sanitized[:40]  # limite la longueur des IDs


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Découpe un texte en morceaux de taille chunk_size avec un chevauchement overlap.
    Le chevauchement évite de couper une phrase entre deux chunks consécutifs.
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap

    return chunks


def create_chunks(text: str, doc_name: str) -> dict:
    """
    Crée les parents et children pour un document.

    Retourne :
      {
        "parents" : liste de dicts parent,
        "children": liste de dicts child,
      }

    Structure d'un parent :
      { chunk_id, type="parent", text, children: [child_id, ...] }

    Structure d'un child :
      { chunk_id, type="child", text, parent_id }
    """
    safe_name = _sanitize_name(doc_name)
    parent_texts = _split_text(text, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)

    parents = []
    children = []

    for p_idx, p_text in enumerate(parent_texts):
        p_id = f"{safe_name}_p{p_idx}"
        child_ids = []

        child_texts = _split_text(p_text, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)
        for c_idx, c_text in enumerate(child_texts):
            c_id = f"{safe_name}_c{p_idx}_{c_idx}"
            children.append({
                "chunk_id": c_id,
                "type": "child",
                "text": c_text,
                "parent_id": p_id,
            })
            child_ids.append(c_id)

        parents.append({
            "chunk_id": p_id,
            "type": "parent",
            "text": p_text,
            "children": child_ids,
        })

    return {"parents": parents, "children": children}
