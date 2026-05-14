"""
Extrait les attributs structurés d'un TdR via Ollama.

Envoie les 3 000 premiers caractères + les 1 500 derniers au LLM :
  - Début du document : titre, organisation, objectifs, profil
  - Fin du document   : date limite, budget (souvent en fin de TdR)
"""

import re
import json

import ollama

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL

_SYSTEM_PROMPT = (
    "Tu es un assistant expert en marchés publics et appels d'offres.\n"
    "Extrait les informations structurées de ce Terme de Référence.\n"
    "Réponds UNIQUEMENT par un JSON valide avec exactement ces champs :\n"
    '{\n'
    '  "titre": "titre de la mission ou null",\n'
    '  "organisation": "organisation commanditaire ou null",\n'
    '  "objectifs": ["objectif 1", "objectif 2"],\n'
    '  "livrables": ["livrable 1", "livrable 2"],\n'
    '  "profil_consultant": "description du profil requis ou null",\n'
    '  "duree": "durée de la mission ou null",\n'
    '  "lieu": "lieu d\'intervention ou null",\n'
    '  "budget": "budget ou honoraires mentionnés ou null",\n'
    '  "date_limite": "date limite de soumission ou null"\n'
    '}\n'
    "Si une information n'est pas mentionnée dans le texte, mets null.\n"
    "Pour objectifs et livrables, retourne une liste vide [] si absent.\n"
    "Sois précis : extrais les valeurs exactes (dates, montants, durées)."
)

_DEFAULT_METADATA = {
    "titre": None,
    "organisation": None,
    "objectifs": [],
    "livrables": [],
    "profil_consultant": None,
    "duree": None,
    "lieu": None,
    "budget": None,
    "date_limite": None,
    "error": None,
}

_HEAD_CHARS = 3000   # début du document (titre, org, objectifs, profil)
_TAIL_CHARS = 1500   # fin du document (date limite, budget, adresse)


def _build_excerpt(text: str) -> str:
    """Construit l'extrait envoyé au LLM : début + fin du document."""
    if len(text) <= _HEAD_CHARS + _TAIL_CHARS:
        return text

    head = text[:_HEAD_CHARS]
    tail = text[-_TAIL_CHARS:]

    # Évite la duplication si le doc est court
    if text[_HEAD_CHARS:].startswith(tail[:50]):
        return head

    return head + "\n\n[...]\n\n" + tail


def extract_metadata(text: str) -> dict:
    """
    Envoie un extrait du texte à Ollama.
    Retourne un dict avec les attributs du TdR.
    Le champ 'error' est None si tout s'est bien passé.
    """
    result = dict(_DEFAULT_METADATA)

    try:
        excerpt = _build_excerpt(text)

        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": excerpt},
            ],
            options={"temperature": 0},
        )

        raw = response.message.content.strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            for key in _DEFAULT_METADATA:
                if key != "error" and key in parsed:
                    result[key] = parsed[key]
        else:
            result["error"] = f"Réponse non parseable : {raw[:100]}"

    except Exception as exc:
        result["error"] = str(exc)

    return result
