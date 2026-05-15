"""
Découpe un Markdown Docling en chunks hiérarchiques parent-child.

Stratégie de chunking par sections :
  - Parent chunk = une section logique du document (délimitée par les titres ##/###)
  - Si une section dépasse PARENT_CHUNK_SIZE → subdivisée avec RecursiveCharacterTextSplitter
  - Child chunk  = subdivision d'un parent (~400 chars) pour la recherche vectorielle

Avantages vs chunking à taille fixe :
  - Les chunks respectent les frontières sémantiques du document
  - "Durée et lieu" ne se retrouve jamais coupé avec "Budget"
  - Les tableaux Markdown restent intacts dans leur section
"""

import re

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import (
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
)

# Séparateurs adaptés au Markdown Docling
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def _sanitize_name(filename_stem: str) -> str:
    sanitized = re.sub(r"[^\w]", "_", filename_stem)
    return sanitized[:40]


def _split_by_sections(markdown: str) -> list[str]:
    """
    Découpe le Markdown aux titres de niveau 1, 2 et 3 (# ## ###).
    Chaque section = un bloc logique du document.
    """
    parts = re.split(r"\n(?=#{1,3} )", markdown)
    return [p.strip() for p in parts if p.strip()]


def create_chunks(text: str, doc_name: str) -> dict:
    """
    Crée les parents et children pour un document Markdown Docling.

    Retourne :
      {
        "parents" : liste de dicts parent,
        "children": liste de dicts child,
      }
    """
    safe_name = _sanitize_name(doc_name)

    # 1. Découpage en sections logiques (par titres Markdown)
    sections = _split_by_sections(text)

    # 2. Si une section dépasse PARENT_CHUNK_SIZE → on la subdivise
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=_SEPARATORS,
        length_function=len,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=_SEPARATORS,
        length_function=len,
    )

    # 3. Construction des parents
    parent_texts: list[str] = []
    for section in sections:
        if len(section) <= PARENT_CHUNK_SIZE:
            parent_texts.append(section)
        else:
            # Section trop longue → subdivisée
            parent_texts.extend(parent_splitter.split_text(section))

    # 4. Construction parent-child
    parents = []
    children = []

    for p_idx, p_text in enumerate(parent_texts):
        p_id = f"{safe_name}_p{p_idx}"
        child_texts = child_splitter.split_text(p_text)
        child_ids = []

        for c_idx, c_text in enumerate(child_texts):
            c_id = f"{safe_name}_c{p_idx}_{c_idx}"
            children.append({
                "chunk_id": c_id,
                "type":     "child",
                "text":     c_text,
                "parent_id": p_id,
            })
            child_ids.append(c_id)

        parents.append({
            "chunk_id": p_id,
            "type":     "parent",
            "text":     p_text,
            "children": child_ids,
        })

    return {"parents": parents, "children": children}
