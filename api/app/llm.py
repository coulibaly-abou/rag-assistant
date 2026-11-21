"""Appels au LLM via l'API /api/chat d'Ollama (réponse complète ou en streaming)."""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx


class OllamaLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 300.0,
        temperature: float = 0.1,
        num_ctx: int = 4096,
    ):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def _payload(self, messages: list[dict], stream: bool, model: str | None) -> dict:
        return {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }

    def chat(self, messages: list[dict], model: str | None = None) -> str:
        response = self._client.post("/api/chat", json=self._payload(messages, False, model))
        response.raise_for_status()
        return response.json()["message"]["content"]

    def stream_chat(self, messages: list[dict], model: str | None = None) -> Iterator[str]:
        payload = self._payload(messages, True, model)
        with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break

    def list_models(self) -> list[str]:
        response = self._client.get("/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
