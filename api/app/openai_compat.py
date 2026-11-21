"""Endpoints compatibles OpenAI (/v1/models, /v1/chat/completions) pour brancher Open WebUI.

Chaque matière indexée est exposée comme un « modèle » distinct (ex. rag-cours/Statistiques),
ce qui permet de filtrer par matière directement depuis le sélecteur de modèle d'Open WebUI.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.rag import format_sources_markdown
from app.services import Services, get_services

router = APIRouter(prefix="/v1", tags=["OpenAI compatible"])
ServicesDep = Annotated[Services, Depends(get_services)]

# Open WebUI envoie aussi des « tâches » (titre de la conversation, tags, suggestions…)
# au même modèle. Elles commencent par ce marqueur : on les passe directement au LLM, sans RAG.
OPENWEBUI_TASK_MARKER = "### Task:"


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False


def message_text(message: ChatMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        return "\n".join(p.get("text", "") for p in message.content if p.get("type") == "text")
    return ""


def last_user_question(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message_text(message).strip()
    return ""


def parse_matiere(model: str, base_name: str) -> str | None:
    prefix = f"{base_name}/"
    return model[len(prefix) :] if model.startswith(prefix) else None


def _completion(model: str, content: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse(pieces: Iterator[str], model: str) -> Iterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(delta: dict, finish_reason: str | None = None) -> str:
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    yield chunk({"role": "assistant", "content": ""})
    for piece in pieces:
        yield chunk({"content": piece})
    yield chunk({}, "stop")
    yield "data: [DONE]\n\n"


@router.get("/models")
def list_models(services: ServicesDep) -> dict:
    base = services.settings.rag_model_name
    try:
        matieres = services.store.matieres()
    except Exception:
        matieres = []
    ids = [base] + [f"{base}/{m}" for m in matieres]
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": i, "object": "model", "created": created, "owned_by": "rag-assistant"}
            for i in ids
        ],
    }


@router.post("/chat/completions", response_model=None)
def chat_completions(
    request: ChatCompletionRequest, services: ServicesDep
) -> dict | StreamingResponse:
    question = last_user_question(request.messages)

    if question.startswith(OPENWEBUI_TASK_MARKER):
        messages = [{"role": m.role, "content": message_text(m)} for m in request.messages]
        if request.stream:
            pieces = services.llm.stream_chat(messages)
            return StreamingResponse(_sse(pieces, request.model), media_type="text/event-stream")
        return _completion(request.model, services.llm.chat(messages))

    matiere = parse_matiere(request.model, services.settings.rag_model_name)

    if not request.stream:
        result = services.pipeline.answer(question, matiere=matiere)
        content = result.answer + format_sources_markdown(result.sources)
        return _completion(request.model, content)

    sources, tokens = services.pipeline.stream_answer(question, matiere=matiere)

    def pieces() -> Iterator[str]:
        yield from tokens
        footer = format_sources_markdown(sources)
        if footer:
            yield footer

    return StreamingResponse(_sse(pieces(), request.model), media_type="text/event-stream")
