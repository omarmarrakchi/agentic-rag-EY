"""
Extrait les attributs structurés d'un TdR via LLM.

Provider configurable via settings.AGENT_PROVIDER :
  - "ollama"  → qwen2.5:14b local
  - "openai"  → gpt-4o via API (plus rapide, meilleure qualité)

Stratégie d'extraction intelligente des sections :
  1. Garder les 2 000 premiers chars (titre, organisation, contexte)
  2. Inclure les sections dont le titre contient des mots-clés métadonnées
     (objectifs, livrables, profil, budget, durée, lieu, date)
  3. Cap total à 7 000 chars → ~65% de réduction vs full-doc
     tout en conservant ~95% de la qualité des métadonnées
"""

import re
import json

import config.settings as _settings

_SYSTEM_PROMPT = (
    "Tu es un assistant expert en marchés publics et appels d'offres.\n"
    "Extrait les informations structurées de ce Terme de Référence.\n"
    "Réponds UNIQUEMENT par un JSON valide avec exactement ces champs :\n"
    '{\n'
    '  "titre": "titre de la mission ou null",\n'
    '  "organisation": "organisation commanditaire ou null",\n'
    '  "objectifs": ["objectif 1", "objectif 2"],\n'
    '  "livrables": ["livrable 1", "livrable 2"],\n'
    '  "profil_consultant": "description complète du profil requis ou null",\n'
    '  "duree": "durée de la mission ou null",\n'
    '  "lieu": "lieu ou pays d\'intervention ou null",\n'
    '  "budget": "budget ou honoraires mentionnés ou null",\n'
    '  "date_limite": "date limite de soumission ou null"\n'
    '}\n'
    "Si une information n'est pas mentionnée dans le texte, mets null.\n"
    "Pour objectifs et livrables, retourne une liste vide [] si absent.\n"
    "Sois précis : extrais les valeurs exactes (dates, montants, durées, lieux).\n"
    "Le document est en Markdown — utilise les titres de sections pour localiser les informations."
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

# Mots-clés qui indiquent qu'une section contient des métadonnées utiles
_SECTION_KEYWORDS = [
    # Français
    "objectif", "livrable", "profil", "consultant", "expert", "budget",
    "honoraire", "durée", "duree", "lieu", "pays", "région", "region",
    "date", "délai", "delai", "soumission", "candidature", "qualification",
    "compétence", "competence", "expérience", "experience", "mission",
    "contexte", "background", "organisation", "commanditaire", "bailleur",
    "résultat", "resultat", "tâche", "tache", "activité", "activite",
    # Anglais
    "objective", "deliverable", "profile", "budget", "duration", "location",
    "deadline", "submission", "qualification", "competency", "experience",
    "task", "activity", "result", "output", "scope",
]

_HEAD_CHARS  = 2_000   # début du doc → titre, organisation
_MAX_SECTION = 800     # max chars par section retenue
_MAX_TOTAL   = 7_000   # cap total envoyé au LLM


def _extract_smart(text: str) -> str:
    """
    Retourne un extrait intelligent du Markdown :
    - Les 2 000 premiers chars (tête du document)
    - Les sections dont le titre contient un mot-clé métadonnée
    Cap total à 7 000 chars.
    """
    # Tête du document (titre, organisation, intro)
    head = text[:_HEAD_CHARS]
    budget = _MAX_TOTAL - len(head)

    if budget <= 0:
        return head

    # Découper le reste en sections (délimitées par les titres Markdown)
    rest = text[_HEAD_CHARS:]
    sections = re.split(r"\n(?=#{1,4} )", rest)

    selected_parts = []
    for section in sections:
        first_line = section.split("\n")[0].lower()
        if any(kw in first_line for kw in _SECTION_KEYWORDS):
            chunk = section[:_MAX_SECTION]
            selected_parts.append(chunk)
            budget -= len(chunk)
            if budget <= 0:
                break

    return head + "\n\n" + "\n\n".join(selected_parts)


def _call_ollama(content: str) -> str:
    import ollama
    client = ollama.Client(host=_settings.OLLAMA_BASE_URL)
    response = client.chat(
        model=_settings.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
        options={"temperature": 0},
    )
    return response.message.content.strip()


def _call_openai(content: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=_settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=_settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content.strip()


def extract_metadata(text: str) -> dict:
    """
    Extrait les métadonnées d'un TdR via le provider configuré (ollama ou openai).
    """
    result = dict(_DEFAULT_METADATA)

    try:
        content = _extract_smart(text)

        if _settings.AGENT_PROVIDER == "openai":
            raw = _call_openai(content)
        else:
            raw = _call_ollama(content)

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
