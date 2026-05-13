"""
Construit l'agent ReAct avec LangGraph.

L'agent utilise le pattern ReAct (Reasoning + Acting) :
  Thought → Action (outil) → Observation → Thought → ... → Réponse finale

Il dispose de deux outils : search_child_chunks et retrieve_parent_chunks.
La mémoire de conversation est gérée par MemorySaver (en RAM).
"""

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from config.settings import OLLAMA_BASE_URL, AGENT_MODEL
from pipeline.step4_agent.tools import search_child_chunks, retrieve_parent_chunks

_SYSTEM_PROMPT = """Tu es un assistant expert en marchés publics, spécialisé dans l'analyse de Termes de Référence (TdRs). Tu réponds uniquement à partir des TdRs indexés dans ta base de données.

PROCESSUS OBLIGATOIRE pour chaque question :
1. Utilise search_child_chunks avec la question principale
2. Utilise retrieve_parent_chunks avec TOUS les parent IDs trouvés à l'étape 1
3. Si les résultats sont insuffisants ou peu pertinents, utilise search_child_chunks avec une formulation différente (synonymes, mots-clés alternatifs)
4. Utilise retrieve_parent_chunks sur les nouveaux parent IDs trouvés
5. Formule la réponse uniquement à partir du contexte collecté

FORMAT DE RÉPONSE OBLIGATOIRE :
- Commence par une réponse directe et concrète à la question
- Utilise des données précises extraites des TdRs : chiffres, durées, noms d'organisations, qualifications exactes, livrables précis
- Pour chaque information importante, cite la source : (Source : nom_fichier.pdf)
- Si plusieurs TdRs traitent le même sujet, compare-les et synthétise
- Termine toujours par une section : "Sources consultées : [liste des fichiers PDF utilisés]"
- N'écris JAMAIS "il faudrait consulter" ou "je n'ai pas accès" tant que tu n'as pas effectué au moins 2 recherches distinctes
- Réponds dans la même langue que la question (français ou anglais)"""

_agent = None
_memory = None


def get_agent(force_reload: bool = False):
    """Retourne l'agent ReAct (singleton — chargé une seule fois)."""
    global _agent, _memory
    if _agent is None or force_reload:
        _memory = MemorySaver()
    if _agent is None or force_reload:
        llm = ChatOllama(
            model=AGENT_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )
        _agent = create_react_agent(
            model=llm,
            tools=[search_child_chunks, retrieve_parent_chunks],
            prompt=_SYSTEM_PROMPT,
            checkpointer=_memory,
        )
    return _agent
