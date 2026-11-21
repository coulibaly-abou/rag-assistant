"""Faux composants : les tests tournent sans Ollama ni Qdrant."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.rag import RagPipeline
from app.services import Services, get_services
from app.store import SearchHit


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self, hits: list[SearchHit]):
        self.hits = hits
        self.last_matiere: str | None = None

    def search(self, vector, top_k, matiere=None):
        self.last_matiere = matiere
        return [h for h in self.hits if matiere is None or h.matiere == matiere][:top_k]

    def matieres(self):
        return sorted({h.matiere for h in self.hits})

    def count(self):
        return len(self.hits)

    def all_chunks(self):
        return list(self.hits)

    def fingerprint(self):
        return (len(self.hits), tuple(h.source for h in self.hits))


class FakeLLM:
    def __init__(self):
        self.calls: list[list[dict]] = []

    def chat(self, messages, model=None) -> str:
        self.calls.append(messages)
        return "La médiane partage la série en deux [1]."

    def stream_chat(self, messages, model=None) -> Iterator[str]:
        self.calls.append(messages)
        yield from ["La médiane ", "partage la série [1]."]

    def list_models(self):
        return ["llama3.2:3b", "nomic-embed-text:latest"]


HITS = [
    SearchHit("La médiane partage la série ordonnée en deux.", "Stats/cours.pdf", 3, "Stats", 0.82),
    SearchHit("Une liste Python est modifiable.", "Python/cours.md", 1, "Python", 0.41),
]


@pytest.fixture
def fakes():
    settings = Settings(auto_ingest=False, _env_file=None)
    embedder, store, llm = FakeEmbedder(), FakeStore(list(HITS)), FakeLLM()
    pipeline = RagPipeline(embedder, store, llm, top_k=settings.top_k)
    return Services(settings, embedder, store, llm, pipeline)


@pytest.fixture
def client(fakes):
    app.dependency_overrides[get_services] = lambda: fakes
    yield TestClient(app)
    app.dependency_overrides.clear()
