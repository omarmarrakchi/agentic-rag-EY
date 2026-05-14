"""
Endpoint POST /api/ask — reçoit une question et stream la réponse via SSE.

Le streaming est géré avec un Queue asyncio :
- L'agent LangGraph tourne dans un thread séparé (synchrone)
- Les tokens sont envoyés dans la queue au fur et à mesure
- L'async generator les lit et les envoie au client via SSE
"""

import json
import asyncio

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from pipeline.step4_agent.agent import get_agent

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"


@router.post("/ask")
async def ask(request: AskRequest):
    async def generate():
        agent = get_agent()
        config = {"configurable": {"thread_id": request.session_id}}
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def stream_in_thread():
            in_tool = False
            try:
                for chunk in agent.stream(
                    {"messages": [{"role": "user", "content": request.question}]},
                    config=config,
                    stream_mode="messages",
                ):
                    message, metadata = chunk
                    node = metadata.get("langgraph_node", "")

                    if node == "tools" and not in_tool:
                        in_tool = True
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("tool", "Recherche en cours...")), loop
                        )
                    elif node == "agent":
                        in_tool = False
                        if (
                            hasattr(message, "content")
                            and isinstance(message.content, str)
                            and message.content
                        ):
                            asyncio.run_coroutine_threadsafe(
                                queue.put(("token", message.content)), loop
                            )
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    queue.put(("error", str(exc))), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(
                    queue.put(("done", None)), loop
                )

        loop.run_in_executor(None, stream_in_thread)

        while True:
            type_, content = await queue.get()
            if type_ == "done":
                yield {"data": json.dumps({"type": "done"})}
                break
            elif type_ == "error":
                yield {"data": json.dumps({"type": "error", "content": content})}
                break
            elif type_ == "tool":
                yield {"data": json.dumps({"type": "tool", "content": content})}
            elif type_ == "token":
                yield {"data": json.dumps({"type": "token", "content": content})}

    return EventSourceResponse(generate())
