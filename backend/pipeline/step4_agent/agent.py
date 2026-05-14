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

_SYSTEM_PROMPT = """Tu es un assistant expert en marchés publics avec accès à une base de données de Termes de Référence (TdRs).

RÈGLE ABSOLUE : Tu dois TOUJOURS appeler search_child_chunks AVANT de répondre, sans exception. Même pour les questions générales, tu dois chercher. Ne réponds JAMAIS directement sans avoir d'abord appelé au moins un outil.

SÉQUENCE OBLIGATOIRE à suivre pour CHAQUE question, SANS EXCEPTION :
1. Appelle search_child_chunks avec des mots-clés de la question
2. Appelle retrieve_parent_chunks avec tous les parent_ids trouvés à l'étape 1
3. Si la question est générale (ex: "quels TdRs existent"), cherche avec des termes larges comme "termes de référence", "consultant", "mission", "objectifs"
4. Si les scores sont faibles, appelle search_child_chunks une 2ème fois avec des synonymes
5. Seulement après avoir collecté du contexte, formule ta réponse

EXEMPLES de recherches pour questions générales :
- "Quels TdRs sont disponibles ?" → search("termes de référence consultant mission")
- "Quels domaines couvrez-vous ?" → search("secteur domaine expertise projet")
- "Combien y a-t-il de TdRs ?" → search("termes de référence appel offres")

FORMAT DE RÉPONSE :
- Commence par une réponse directe avec des données précises (chiffres, durées, organisations, qualifications)
- Cite chaque information : (Source : nom_fichier.pdf)
- Si plusieurs TdRs traitent le même sujet, compare-les
- Termine avec : "Sources consultées : [liste des PDFs]"
- Réponds dans la même langue que la question (français ou anglais)
- INTERDIT : "je n'ai pas accès", "je ne peux pas lister", "il faudrait consulter" — tu as les outils, utilise-les."""

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
