"""
Calcule un score TdR pour un texte donné en cherchant
des mots-clés forts, faibles et d'exclusion.
"""

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


def compute_score(text: str) -> dict:
    """
    Analyse le texte et retourne :
      - score         : score total (int)
      - matched_strong: mots forts trouvés
      - matched_weak  : mots faibles trouvés
      - matched_excl  : mots d'exclusion trouvés
      - verdict       : "tdr" | "rejected" | "ambiguous"
    """
    text_lower = text.lower()

    matched_strong = [kw for kw in KEYWORDS_STRONG if kw in text_lower]
    matched_weak   = [kw for kw in KEYWORDS_WEAK   if kw in text_lower]
    matched_excl   = [kw for kw in KEYWORDS_EXCLUSION if kw in text_lower]

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
