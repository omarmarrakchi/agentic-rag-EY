"""
Nettoie le texte extrait d'un PDF avant le chunking.
Préserve les blocs [TABLEAU]...[/TABLEAU] intacts pendant le nettoyage.
"""

import re


def _protect_tables(text: str) -> tuple[str, dict]:
    """Remplace les blocs tableau par des marqueurs temporaires."""
    placeholders = {}
    counter = [0]

    def replace(m):
        key = f"__TBLPROTECT_{counter[0]}__"
        placeholders[key] = m.group(0)
        counter[0] += 1
        return f"\n\n{key}\n\n"

    processed = re.sub(r"\[TABLEAU\].*?\[/TABLEAU\]", replace, text, flags=re.DOTALL)
    return processed, placeholders


def _restore_tables(text: str, placeholders: dict) -> str:
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def clean_text(text: str, is_scanned: bool = False) -> str:
    """
    Normalise le texte :
      - Supprime les caractères de contrôle
      - Normalise les espaces et tabulations
      - Limite les sauts de ligne consécutifs à 2
      - Si OCR : recolle les mots coupés en fin de ligne
      - Préserve les blocs [TABLEAU] intacts
    """
    # Protège les tableaux avant nettoyage
    text, placeholders = _protect_tables(text)

    text = text.replace("­", "")   # tiret mou
    text = text.replace("\x0c", "\n")   # form feed → saut de ligne

    if is_scanned:
        text = re.sub(r"-\n([a-zàâéèêëîïôùûüa-z])", r"\1", text)

    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))

    # Restaure les tableaux
    text = _restore_tables(text, placeholders)

    return text.strip()
