"""
Interface principale de l'agent RAG.

Expose une fonction ask() qui :
  1. Reformule la question pour optimiser la recherche
  2. Passe la question reformulée à l'agent ReAct
  3. Retourne la réponse avec les métadonnées (question originale, reformulation, session)
"""

from pipeline.step4_agent.query_rewriter import rewrite_query
from pipeline.step4_agent.agent import get_agent


def ask(question: str, session_id: str = "default") -> dict:
    """
    Pose une question à l'agent RAG.

    Args:
        question  : question de l'utilisateur (français ou anglais)
        session_id: identifiant de session pour la mémoire de conversation

    Retourne un dict avec :
      - question        : question originale
      - rewritten_query : question reformulée envoyée à l'agent
      - response        : réponse de l'agent
      - session_id      : identifiant de session
    """
    agent = get_agent()

    rewritten = rewrite_query(question)

    config = {"configurable": {"thread_id": session_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": rewritten}]},
        config=config,
    )

    response = result["messages"][-1].content

    return {
        "question":        question,
        "rewritten_query": rewritten,
        "response":        response,
        "session_id":      session_id,
    }
