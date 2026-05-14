"""
Découpe un texte en chunks hiérarchiques parent-child.

Améliorations vs version précédente :
  - RecursiveCharacterTextSplitter : coupe aux paragraphes/phrases,
    jamais en milieu de mot.
  - Blocs [TABLEAU]...[/TABLEAU] protégés : jamais fragmentés.
  - Détection des titres de section : traités comme frontières naturelles.

Parent chunk : grande fenêtre (~1500 chars) — contexte riche pour le LLM.
Child chunk  : petite fenêtre (~400 chars) — précis pour la recherche vectorielle.
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

# Séparateurs par ordre de priorité : paragraphe > ligne > phrase > mot
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

# Regex de détection des titres de section courants dans les TdRs
_SECTION_RE = re.compile(
    r"(?m)^("
    r"\d+[\.\)]\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ\w]"        # "1. Objectifs" / "2) Profil"
    r"|[IVX]+\.\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]"           # "I. Introduction"
    r"|[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]{3}[A-ZÀÂÉÈÊËÎÏÔÙÛÜ\s]{2,}$"  # "PROFIL DU CONSULTANT"
    r"|(?:Article|Section|Chapitre|Annexe)\s+\d+"  # "Article 1"
    r")"
)


def _sanitize_name(filename_stem: str) -> str:
    sanitized = re.sub(r"[^\w]", "_", filename_stem)
    return sanitized[:40]


def _protect_tables(text: str) -> tuple[str, dict]:
    """Remplace les blocs [TABLEAU] par des clés pour éviter leur découpage."""
    placeholders = {}
    counter = [0]

    def replace(m):
        key = f"\n\n__TABLE_{counter[0]}__\n\n"
        placeholders[f"__TABLE_{counter[0]}__"] = m.group(0)
        counter[0] += 1
        return key

    processed = re.sub(r"\[TABLEAU\].*?\[/TABLEAU\]", replace, text, flags=re.DOTALL)
    return processed, placeholders


def _restore_tables(text: str, placeholders: dict) -> str:
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def _mark_sections(text: str) -> str:
    """Insère un double saut de ligne avant chaque titre de section."""
    return _SECTION_RE.sub(lambda m: f"\n\n{m.group()}", text)


def create_chunks(text: str, doc_name: str) -> dict:
    """
    Crée les parents et children pour un document.

    Retourne :
      {
        "parents" : liste de dicts parent,
        "children": liste de dicts child,
      }
    """
    safe_name = _sanitize_name(doc_name)

    # 1. Protège les tableaux
    processed, placeholders = _protect_tables(text)

    # 2. Marque les frontières de section
    processed = _mark_sections(processed)

    # 3. Découpe en parents
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

    parent_texts = parent_splitter.split_text(processed)

    parents = []
    children = []

    for p_idx, p_text in enumerate(parent_texts):
        # Restaure les tableaux dans le parent
        p_text = _restore_tables(p_text, placeholders)
        p_id = f"{safe_name}_p{p_idx}"
        child_ids = []

        child_texts = child_splitter.split_text(p_text)
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
