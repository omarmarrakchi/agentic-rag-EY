"""
Calcule un score TdR pour un texte donné en cherchant
des mots-clés forts, faibles et d'exclusion.
"""

import unicodedata

from config.settings import (
    KEYWORDS_STRONG,
    KEYWORDS_WEAK,
    KEYWORDS_EXCLUSION,
    WEIGHT_STRONG,
    WEIGHT_WEAK,
    WEIGHT_EXCLUSION,
    SCORE_TDR_CONFIRMED,
    SCORE_REJECTED,
)


def _normalize(s: str) -> str:
    """Passe en minuscules et supprime les accents.
    'Termes de Références' → 'termes de references'
    Permet de matcher les textes OCR imparfaits et les variantes orthographiques.
    """
    return unicodedata.normalize("NFD", s.lower()).encode("ascii", "ignore").decode("ascii")


# Pré-normalisation des listes pour ne pas recalculer à chaque appel
_STRONG_NORM = [(_normalize(kw), kw) for kw in KEYWORDS_STRONG]
_WEAK_NORM   = [(_normalize(kw), kw) for kw in KEYWORDS_WEAK]
_EXCL_NORM   = [(_normalize(kw), kw) for kw in KEYWORDS_EXCLUSION]


def compute_score(text: str) -> dict:
    """
    Analyse le texte et retourne :
      - score         : score total (int)
      - matched_strong: mots forts trouvés
      - matched_weak  : mots faibles trouvés
      - matched_excl  : mots d'exclusion trouvés
      - verdict       : "tdr" | "rejected" | "ambiguous"
    """
    text_norm = _normalize(text)

    matched_strong = [kw for kw_norm, kw in _STRONG_NORM if kw_norm in text_norm]
    matched_weak   = [kw for kw_norm, kw in _WEAK_NORM   if kw_norm in text_norm]
    matched_excl   = [kw for kw_norm, kw in _EXCL_NORM   if kw_norm in text_norm]

    score = (
        len(matched_strong) * WEIGHT_STRONG
        + len(matched_weak)  * WEIGHT_WEAK
        + len(matched_excl)  * WEIGHT_EXCLUSION
    )

    if score >= SCORE_TDR_CONFIRMED:
        verdict = "tdr"
    elif score <= SCORE_REJECTED:
        verdict = "rejected"
    else:
        verdict = "ambiguous"

    return {
        "score": score,
        "matched_strong": matched_strong,
        "matched_weak": matched_weak,
        "matched_excl": matched_excl,
        "verdict": verdict,
    }
