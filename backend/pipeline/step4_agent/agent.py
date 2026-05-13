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

_SYSTEM_PROMPT = """Tu es un assistant expert en marchés publics et appels d'offres, \
spécialisé dans l'analyse de Termes de Référence (TdRs).

Tu as accès à une base de données de TdRs indexés. Voici comment tu dois procéder :

RÈGLES DE RECHERCHE (obligatoires) :
1. Fais TOUJOURS au moins 3 recherches avec search_child_chunks en utilisant des formulations différentes avant de conclure qu'une information est absente
2. Exemple : pour "profil ERP", essaie "profil consultant ERP", puis "expert système ERP qualifications", puis "compétences requises ERP implémentation"
3. Après chaque search_child_chunks, utilise retrieve_parent_chunks avec les parent IDs trouvés pour obtenir le contexte complet
4. Ne conclus jamais "information non disponible" avant d'avoir fait au moins 3 recherches différentes

RÈGLES DE RÉPONSE :
5. Base tes réponses UNIQUEMENT sur les informations trouvées dans les TdRs
6. Cite toujours le nom du fichier PDF source pour chaque information
7. Si après 3 recherches l'information est vraiment absente, dis-le clairement
8. Réponds dans la même langue que la question posée (français ou anglais)
9. Structure ta réponse de façon claire avec des points ou sections si nécessaire"""

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
