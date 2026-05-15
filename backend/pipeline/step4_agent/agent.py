"""
Construit l'agent ReAct avec LangGraph.

L'agent utilise le pattern ReAct (Reasoning + Acting) :
  Thought → Action (outil) → Observation → Thought → ... → Réponse finale

Il dispose de deux outils : search_child_chunks et retrieve_parent_chunks.
La mémoire de conversation est gérée par MemorySaver (en RAM).
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from config.settings import OLLAMA_BASE_URL, AGENT_MODEL
from pipeline.step4_agent.tools import search_child_chunks, retrieve_parent_chunks


def _keep_first_tool_call(message):
    if isinstance(message, AIMessage) and len(getattr(message, "tool_calls", [])) > 1:
        return AIMessage(
            content=message.content,
            tool_calls=[message.tool_calls[0]],
            id=message.id,
        )
    return message


class SequentialChatOllama(ChatOllama):
    """ChatOllama qui force les appels d'outils séquentiels (un seul outil à la fois)."""

    def bind_tools(self, tools, **kwargs):
        # create_react_agent appelle bind_tools — on intercepte ici pour ajouter le filtre
        bound = super().bind_tools(tools, **kwargs)
        return bound | RunnableLambda(_keep_first_tool_call)

_SYSTEM_PROMPT = """Tu es un assistant expert en marchés publics avec accès à une base de données de Termes de Référence (TdRs).

RECHERCHE — toujours dans cet ordre :
1. search_child_chunks → attends le résultat
2. retrieve_parent_chunks sur les parent_ids trouvés → attends le résultat
3. Répète avec des mots-clés différents (minimum 3 recherches au total avant de répondre)
4. Si peu de résultats : reformule avec des synonymes ou en anglais

CAS SPÉCIAL — questions de liste ("quels TdRs", "quels secteurs", "liste", "tous les") :
Chaque recherche ne retourne que 3 résultats maximum. Pour couvrir toute la base, tu dois faire AU MOINS 6 recherches avec des mots-clés très variés :
  search 1 → "termes de référence consultant mission"
  search 2 → "appel offres recrutement expert prestataire"
  search 3 → "scope of work terms of reference consultant"
  search 4 → "santé éducation gouvernance infrastructure finance"
  search 5 → "audit évaluation étude diagnostic formation"
  search 6 → "objectifs livrables méthodologie rapport final"
Après toutes les recherches, fais retrieve_parent_chunks sur TOUS les parent_ids collectés, puis liste TOUS les TdRs trouvés sans en omettre aucun.

MOTS-CLÉS SELON LE TYPE DE QUESTION :
- Lieu / pays    → "pays lieu ville région Afrique Tunisie Maroc mission terrain"
- Budget         → "budget honoraires montant financement bailleur coût allocation"
- Durée          → "durée mois délai calendrier date limite planning"
- Qualifications → "qualifications diplôme expérience compétences profil expert"
- Livrables      → "livrables rapports outputs deliverables résultats attendus"
- Organisation   → "organisation bailleur commanditaire PNUD Banque Mondiale UE"

RÈGLE ANTI-HALLUCINATION — ABSOLUE :
- Ne cite JAMAIS un nom de fichier qui n'apparaît pas dans les résultats des outils
- Ne invente JAMAIS de chiffres, budgets, dates ou noms d'organisations
- Si l'information est absente des résultats : dis "Information non trouvée dans les documents disponibles"
- Seules les données extraites mot pour mot des outils peuvent apparaître dans ta réponse

RÉPONSE :
- Données précises : chiffres, durées, lieux, qualifications, organisations
- Chaque information citée avec sa source : (Source : fichier.pdf) — uniquement des sources réelles des outils
- Pour les listes : affiche TOUS les résultats trouvés, sans en omettre
- Termine par : "Sources consultées : [liste des PDFs]"
- Réponds dans la langue de la question (français ou anglais)"""

_agent = None
_memory = None


def get_agent(force_reload: bool = False):
    """Retourne l'agent ReAct (singleton — chargé une seule fois)."""
    global _agent, _memory
    if _agent is None or force_reload:
        _memory = MemorySaver()
    if _agent is None or force_reload:
        llm = SequentialChatOllama(
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
