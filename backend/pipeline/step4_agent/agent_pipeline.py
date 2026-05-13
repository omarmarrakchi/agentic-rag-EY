"""
Interface principale de l'agent RAG.

Expose une fonction ask() qui :
  1. Reformule la question pour optimiser la recherche
  2. Passe la question reformulée à l'agent ReAct
  3. Retourne la réponse avec les métadonnées (question originale, reformulation, session)
"""

from pipeline.step4_agent.agent import get_agent


def ask(question: str, session_id: str = "default") -> dict:
    """
    Pose une question à l'agent RAG (mode non-streaming).
    Utilisé par l'API FastAPI.

    Retourne un dict avec : question, response, session_id.
    """
    agent = get_agent()
    config = {"configurable": {"thread_id": session_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )

    return {
        "question":   question,
        "response":   result["messages"][-1].content,
        "session_id": session_id,
    }
