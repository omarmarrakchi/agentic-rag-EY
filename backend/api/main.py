"""
Serveur FastAPI — point d'entrée de l'API.
Lance avec : python run_api.py
"""

import os
import sys
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.ask import router as ask_router
from api.routes.health import router as health_router
from api.routes.search import router as search_router
from api.routes.provider import router as provider_router

app = FastAPI(title="Agentic RAG — TdRs EY", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask_router, prefix="/api")
app.include_router(health_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(provider_router, prefix="/api")
