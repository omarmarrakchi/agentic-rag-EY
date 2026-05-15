"""
Nettoyage léger du Markdown produit par Docling.

Docling produit déjà un texte propre et structuré — ce module
se limite à normaliser les sauts de ligne excessifs.
"""

import re


def clean_text(text: str, is_scanned: bool = False) -> str:
    """
    Normalise le Markdown Docling :
      - Supprime les sauts de ligne excessifs (> 2 consécutifs)
      - Supprime les espaces en début/fin
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
