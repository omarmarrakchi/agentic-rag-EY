"""
Endpoint pour consulter et changer le provider LLM à chaud.

GET  /api/provider          → provider actuel + modèle actif
POST /api/provider          → changer de provider (ollama | openai)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config.settings as _settings

router = APIRouter()


class ProviderRequest(BaseModel):
    provider: str       # "ollama" | "openai"
    openai_api_key: str = ""   # optionnel : renseigner la clé à la volée


class ProviderResponse(BaseModel):
    provider: str
    model: str
    status: str


@router.get("/provider", response_model=ProviderResponse)
def get_provider():
    model = (
        _settings.OPENAI_MODEL
        if _settings.AGENT_PROVIDER == "openai"
        else _settings.AGENT_MODEL
    )
    return ProviderResponse(
        provider=_settings.AGENT_PROVIDER,
        model=model,
        status="active",
    )


@router.post("/provider", response_model=ProviderResponse)
def set_provider(body: ProviderRequest):
    if body.provider not in ("ollama", "openai"):
        raise HTTPException(status_code=400, detail="provider doit être 'ollama' ou 'openai'")

    if body.provider == "openai":
        api_key = body.openai_api_key or _settings.OPENAI_API_KEY
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="openai_api_key requis pour passer en mode OpenAI",
            )
        _settings.OPENAI_API_KEY = api_key

    _settings.AGENT_PROVIDER = body.provider

    # Force le rechargement de l'agent au prochain appel
    from pipeline.step4_agent.agent import get_agent
    get_agent(force_reload=True)

    model = (
        _settings.OPENAI_MODEL
        if _settings.AGENT_PROVIDER == "openai"
        else _settings.AGENT_MODEL
    )
    return ProviderResponse(
        provider=_settings.AGENT_PROVIDER,
        model=model,
        status="switched",
    )
