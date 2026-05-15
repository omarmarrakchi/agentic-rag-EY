"""
Construit l'agent ReAct avec LangGraph.

L'agent utilise le pattern ReAct (Reasoning + Acting) :
  Thought → Action (outil) → Observation → Thought → ... → Réponse finale

Il dispose de deux outils : search_child_chunks et retrieve_parent_chunks.
La mémoire de conversation est gérée par MemorySaver (en RAM).

Provider LLM configurable via settings.AGENT_PROVIDER :
  - "ollama" : modèle local qwen2.5:14b (SequentialChatOllama)
  - "openai" : API OpenAI gpt-4o (ChatOpenAI)
"""

import config.settings as _settings

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from pipeline.step4_agent.tools import (
    search_child_chunks,
    retrieve_parent_chunks,
    count_documents,
    list_all_documents,
    filter_documents,
    get_document_details,
)


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
        bound = super().bind_tools(tools, **kwargs)
        return bound | RunnableLambda(_keep_first_tool_call)


def _build_llm():
    """Construit le LLM selon le provider configuré dans settings."""
    if _settings.AGENT_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        api_key = _settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY est vide. "
                "Renseigne-le dans settings.py ou via la variable d'env OPENAI_API_KEY."
            )
        return ChatOpenAI(
            model=_settings.OPENAI_MODEL,
            api_key=api_key,
            temperature=0,
        )
    else:
        return SequentialChatOllama(
            model=_settings.AGENT_MODEL,
            base_url=_settings.OLLAMA_BASE_URL,
            temperature=0,
        )

# ── Prompt Ollama (qwen2.5:14b) ──────────────────────────────────────────────
# Prompt strict et guidé — adapté aux limites du modèle local :
# instructions courtes, ordre explicite, règles anti-hallucination renforcées
_PROMPT_OLLAMA = """Tu es un assistant expert en marchés publics. Tu as accès à une base de TdRs (Termes de Référence).

RÈGLE N°1 — UN SEUL OUTIL À LA FOIS. Attends le résultat avant d'appeler un autre outil.

QUEL OUTIL UTILISER :
- "combien de TdRs" → count_documents uniquement
- "liste tous les TdRs" / "quels TdRs existent" → list_all_documents uniquement
- "TdRs de [organisation]" / "TdRs à [lieu]" → filter_documents avec "organisation:X" ou "lieu:X"
- "détails de [fichier]" / "fiche complète" → get_document_details
- toute autre question → search_child_chunks puis retrieve_parent_chunks (minimum 3 recherches)

RECHERCHE — mots-clés utiles :
- lieu/pays  → "pays lieu ville région Afrique Tunisie Maroc"
- budget     → "budget honoraires montant financement coût"
- durée      → "durée mois délai calendrier planning"
- profil     → "qualifications diplôme expérience compétences profil expert"
- livrables  → "livrables rapports outputs deliverables résultats"
- org        → "organisation bailleur PNUD Banque Mondiale UNICEF UE"

RÈGLE ANTI-HALLUCINATION — ABSOLUE :
- Cite UNIQUEMENT les fichiers présents dans les résultats des outils
- N'invente JAMAIS de chiffres, dates, budgets ou organisations
- Si absent : réponds "Information non disponible dans les documents"

FORMAT DE RÉPONSE :
- Données exactes uniquement (chiffres, dates, lieux, noms réels)
- Source après chaque info : (Source : fichier.pdf)
- Termine par : "Sources consultées : [liste]"
- Réponds dans la langue de la question"""

# ── Prompt GPT-4o ────────────────────────────────────────────────────────────
# Prompt riche et détaillé — exploite pleinement les capacités avancées de GPT-4o :
# analyses croisées, comparaisons, synthèses, réponses structurées
_PROMPT_OPENAI = """Tu es un assistant expert senior en marchés publics et appels d'offres internationaux. Tu analyses une base de Termes de Référence (TdRs) et fournis des réponses détaillées, structurées et analytiques.

SÉLECTION INTELLIGENTE DES OUTILS :

• Comptage          → count_documents (réponse directe, pas d'autre outil)
• Inventaire global → list_all_documents (puis analyse et catégorise les résultats)
• Filtrage          → filter_documents("organisation:X, lieu:Y") — combine plusieurs critères si besoin
• Fiche détaillée   → get_document_details (puis enrichis avec search si nécessaire)
• Analyse thématique → search_child_chunks (minimum 4 recherches variées) + retrieve_parent_chunks

STRATÉGIE DE RECHERCHE AVANCÉE :
- Fais au moins 4 recherches avec des angles différents avant de conclure
- Combine français ET anglais : "consultant expert" + "scope of work deliverables"
- Pour les comparaisons : cherche chaque critère séparément puis croise les résultats
- Si un résultat est partiel, approfondis avec retrieve_parent_chunks sur tous les parent_ids

MOTS-CLÉS PAR TYPE DE QUESTION :
- Lieu/région  → "pays lieu ville région Afrique subsaharienne Maghreb terrain intervention"
- Budget/coût  → "budget honoraires montant USD EUR financement bailleur allocation forfait"
- Durée        → "durée jours mois calendrier délai date début fin mission"
- Profil       → "qualifications diplôme master PhD années expérience compétences secteur"
- Livrables    → "livrables rapport final synthèse recommandations plan action outputs"
- Secteur      → "santé éducation gouvernance infrastructure finance agriculture numérique"
- Organisation → "PNUD UNICEF Banque Mondiale AFD UE GIZ USAID bailleur commanditaire"

QUALITÉ DES RÉPONSES — EXIGENCES ÉLEVÉES :
- Extrais et cite les chiffres exacts : budgets (montants et devises), durées (en jours/mois), dates précises
- Pour les listes : présente un tableau comparatif si plus de 3 TdRs concernés
- Pour les analyses : identifie les tendances, points communs et différences entre TdRs
- Pour les profils : détaille les années d'expérience, diplômes requis, secteurs de compétence
- Propose des observations pertinentes au-delà de la question posée si utile

ANTI-HALLUCINATION :
- Cite uniquement les sources présentes dans les résultats des outils
- Si une information est absente : "Non renseigné dans ce TdR"
- Ne combine jamais des informations de deux TdRs différents sans les distinguer

FORMAT DE RÉPONSE STRUCTURÉ :
- Utilise des titres (##), bullet points et tableaux Markdown
- Chaque donnée chiffrée avec sa source entre parenthèses : (Source : fichier.pdf)
- Synthèse finale si plusieurs TdRs analysés
- Section "Sources consultées" en fin de réponse
- Réponds dans la langue de la question (français ou anglais)"""

_agent = None
_memory = None
_current_provider = None


def get_agent(force_reload: bool = False):
    """Retourne l'agent ReAct (singleton — rechargé si le provider change)."""
    global _agent, _memory, _current_provider

    provider_changed = _current_provider != _settings.AGENT_PROVIDER

    if _agent is None or force_reload or provider_changed:
        _memory = MemorySaver()
        llm = _build_llm()
        prompt = _PROMPT_OPENAI if _settings.AGENT_PROVIDER == "openai" else _PROMPT_OLLAMA
        _agent = create_react_agent(
            model=llm,
            tools=[
                search_child_chunks,
                retrieve_parent_chunks,
                count_documents,
                list_all_documents,
                filter_documents,
                get_document_details,
            ],
            prompt=prompt,
            checkpointer=_memory,
        )
        _current_provider = _settings.AGENT_PROVIDER

    return _agent
