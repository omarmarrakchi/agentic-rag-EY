"""
Reformule la question de l'utilisateur pour optimiser la recherche vectorielle.

Une question comme "C'est quoi le profil pour ERP ?" est reformulée en
"profil consultant expert ERP qualifications expérience requise" — plus
proche du vocabulaire des TdRs et donc meilleure pour la recherche.
"""

import ollama

from config.settings import OLLAMA_BASE_URL, AGENT_MODEL

_SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé dans la reformulation de questions pour la recherche dans des Termes de Référence (TdRs) de marchés publics.\n\n"
    "Le vocabulaire typique des TdRs inclut : consultant, expert, profil recherché, qualifications requises, "
    "expérience requise, livrables attendus, objectifs de la mission, durée de la mission, "
    "compétences techniques, scope of work, terms of reference, deliverables, required experience, "
    "individual consultant, consulting firm, recrutement, appel d'offres, cahier des charges.\n\n"
    "Reformule la question en :\n"
    "- Utilisant le vocabulaire ci-dessus quand c'est pertinent\n"
    "- Privilégiant les termes exacts qu'on trouverait dans un TdR\n"
    "- Restant concis (une seule phrase)\n"
    "- Gardant la même langue que la question (français ou anglais)\n\n"
    "Réponds UNIQUEMENT avec la question reformulée, sans explication."
)


def rewrite_query(question: str) -> str:
    """
    Reformule la question pour améliorer la recherche vectorielle.
    En cas d'erreur, retourne la question originale.
    """
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            options={"temperature": 0},
        )
        return response.message.content.strip()
    except Exception:
        return question
