"""Point d'entrée CLI pour interagir avec l'agent RAG en mode conversationnel."""

import os
import sys
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")

from pipeline.step4_agent.agent_pipeline import ask

print("=" * 60)
print("   Agent RAG — Termes de Référence EY")
print("=" * 60)
print("Tapez votre question en français ou en anglais.")
print("Tapez 'exit' pour quitter.\n")

SESSION_ID = "session_1"

while True:
    try:
        question = input("Question : ").strip()
    except (KeyboardInterrupt, EOFError):
        break

    if not question:
        continue
    if question.lower() in ("exit", "quit", "q"):
        break

    print("\n Recherche en cours...\n")

    result = ask(question, session_id=SESSION_ID)

    print(f"[Requête reformulée : {result['rewritten_query']}]\n")
    print("Réponse :")
    print(result["response"])
    print("\n" + "─" * 60 + "\n")
