"""Configuration de l'application, lue depuis les variables d'environnement / le fichier .env."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Services
    ollama_url: str = "http://localhost:11434"
    qdrant_url: str = "http://localhost:6333"

    # Modèles
    llm_model: str = "llama3.2:3b"
    embed_model: str = "nomic-embed-text"

    # Indexation
    data_dir: str = "data"
    collection_name: str = "cours"
    chunk_size: int = Field(default=300, gt=0, description="Taille d'un chunk, en mots")
    chunk_overlap: int = Field(default=50, ge=0, description="Chevauchement entre chunks, en mots")
    auto_ingest: bool = True

    # Recherche et génération
    top_k: int = Field(default=3, gt=0)
    min_score: float = Field(default=0.0, description="Score cosinus minimal d'un chunk retenu")
    search_mode: Literal["vector", "hybrid"] = "hybrid"
    hybrid_candidates: int = Field(default=20, gt=0, description="Candidats par méthode avant RRF")
    temperature: float = 0.1
    num_ctx: int = 4096
    request_timeout: float = 300.0

    # Nom du « modèle » exposé à Open WebUI
    rag_model_name: str = "rag-cours"

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap doit être strictement inférieur à chunk_size")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
