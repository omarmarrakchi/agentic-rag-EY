"""Point d'entrée CLI pour interagir avec l'agent RAG en mode conversationnel."""

import os
import sys
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")

from pipeline.step4_agent.agent import get_agent

print("=" * 60)
print("   Agent RAG — Termes de Référence EY")
print("=" * 60)
print("Tapez votre question en français ou en anglais.")
print("Tapez 'exit' pour quitter.\n")

SESSION_ID = "session_1"
agent = get_agent()

while True:
    try:
        question = input("Question : ").strip()
    except (KeyboardInterrupt, EOFError):
        break

    if not question:
        continue
    if question.lower() in ("exit", "quit", "q"):
        break

    print("\nRéponse :\n")

    config = {"configurable": {"thread_id": SESSION_ID}}
    in_tool = False

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="messages",
    ):
        message, metadata = chunk
        node = metadata.get("langgraph_node", "")

        # Indicateur quand l'agent appelle un outil
        if node == "tools" and not in_tool:
            print("[Recherche en cours...]", flush=True)
            in_tool = True
        elif node == "agent":
            in_tool = False

        # Stream des tokens de la réponse finale
        if (
            node == "agent"
            and hasattr(message, "content")
            and isinstance(message.content, str)
            and message.content
        ):
            print(message.content, end="", flush=True)

    print("\n" + "─" * 60 + "\n")
