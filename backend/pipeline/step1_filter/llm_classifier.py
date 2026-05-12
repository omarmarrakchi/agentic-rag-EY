"""
Utilise Ollama pour trancher les cas ambigus que le scorer
de mots-clés n'a pas pu classifier avec certitude.
"""

import ollama

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL

_SYSTEM_PROMPT = """Tu es un assistant expert en marchés publics et appels d'offres.
Ta seule tâche est de déterminer si un extrait de document est un Terme de Référence (TdR).

Un TdR est un document qui décrit une mission à confier à un consultant ou une entreprise :
il précise les objectifs, les livrables attendus, le profil recherché et le contexte du projet.

Réponds UNIQUEMENT par un JSON valide, sans aucun texte avant ou après :
{"is_tdr": true, "reason": "explication courte"}
ou
{"is_tdr": false, "reason": "explication courte"}"""


def classify_with_llm(text: str) -> dict:
    """
    Envoie les 500 premiers caractères du texte à Ollama et retourne :
      - is_tdr  : True / False
      - reason  : explication courte du modèle
      - verdict : "tdr" | "rejected"
      - error   : message si l'appel échoue (None sinon)
    """
    result = {"is_tdr": False, "reason": "", "verdict": "rejected", "error": None}

    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": text[:500]},
            ],
            options={"temperature": 0},
        )

        raw = response.message.content.strip()

        # Extraire le JSON même si le modèle ajoute du texte autour
        import re, json
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            result["is_tdr"]  = bool(parsed.get("is_tdr", False))
            result["reason"]  = parsed.get("reason", "")
            result["verdict"] = "tdr" if result["is_tdr"] else "rejected"
        else:
            result["error"] = f"Réponse non parseable : {raw[:100]}"

    except Exception as exc:
        result["error"] = str(exc)

    return result
