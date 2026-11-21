"""Calcul des embeddings via l'API HTTP d'Ollama."""

from __future__ import annotations

import httpx


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, timeout: float = 300.0, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def _prefix(self, kind: str) -> str:
        # nomic-embed-text a été entraîné avec ces préfixes : ils améliorent la recherche.
        if "nomic" in self.model:
            return "search_query: " if kind == "query" else "search_document: "
        return ""

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i : i + self.batch_size]
            response = self._client.post("/api/embed", json={"model": self.model, "input": batch})
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefix = self._prefix("document")
        return self._embed([prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self._prefix("query") + text])[0]
