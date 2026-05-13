"""
Nettoie le texte extrait d'un PDF avant le chunking.
Applique un nettoyage standard pour les PDFs natifs,
et un nettoyage plus agressif pour les sorties OCR.
"""

import re


def clean_text(text: str, is_scanned: bool = False) -> str:
    """
    Normalise le texte :
      - Supprime les caractères de contrôle
      - Normalise les espaces et tabulations
      - Limite les sauts de ligne consécutifs à 2
      - Si OCR : recollecte les mots coupés en fin de ligne (ex: "consul-\ntant")
    """
    # Caractères de contrôle et tirets mous
    text = text.replace("­", "")   # tiret mou
    text = text.replace("\x0c", "\n")   # form feed → saut de ligne

    if is_scanned:
        # Recolle les mots coupés par un trait d'union en fin de ligne
        text = re.sub(r"-\n([a-zàâéèêëîïôùûüa-z])", r"\1", text)

    # Normalise les espaces dans chaque ligne
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]

    # Supprime les lignes vides consécutives (max 2 sauts de ligne)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))

    return text.strip()
