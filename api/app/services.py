"""Construction des composants partagés (clients Ollama / Qdrant, pipeline RAG)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.config import Settings, get_settings
from app.embeddings import OllamaEmbedder
from app.llm import OllamaLLM
from app.rag import RagPipeline
from app.store import VectorStore


@dataclass
class Services:
    settings: Settings
    embedder: OllamaEmbedder
    store: VectorStore
    llm: OllamaLLM
    pipeline: RagPipeline


def build_services(settings: Settings) -> Services:
    embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model, settings.request_timeout)
    store = VectorStore(settings.qdrant_url, settings.collection_name)
    llm = OllamaLLM(
        settings.ollama_url,
        settings.llm_model,
        timeout=settings.request_timeout,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
    )
    pipeline = RagPipeline(embedder, store, llm, settings.top_k, settings.min_score)
    return Services(settings, embedder, store, llm, pipeline)


@lru_cache
def get_services() -> Services:
    """Dépendance FastAPI (remplacée par des faux objets dans les tests)."""
    return build_services(get_settings())
