"""Routes FastAPI : /ask, /ingest, /health, /matieres (+ /v1/* pour Open WebUI)."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import __version__
from app.ingest import ingest, ingest_lock
from app.openai_compat import router as openai_router
from app.rag import Source
from app.services import Services, get_services

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s")
logger = logging.getLogger("rag-api")

ServicesDep = Annotated[Services, Depends(get_services)]


def _run_ingest(services: Services, prune: bool = True):
    s = services.settings
    return ingest(
        Path(s.data_dir), services.embedder, services.store, s.chunk_size, s.chunk_overlap, prune
    )


def _auto_ingest(services: Services, attempts: int = 30, delay: float = 10.0) -> None:
    """Indexe data/ au démarrage, en attendant que Ollama et Qdrant soient prêts."""
    for attempt in range(1, attempts + 1):
        try:
            with ingest_lock:
                report = _run_ingest(services)
            logger.info(
                "Indexation automatique : %d indexés, %d inchangés, %d supprimés, %d erreurs",
                len(report.indexed),
                len(report.skipped),
                len(report.removed),
                len(report.errors),
            )
            return
        except Exception as exc:
            logger.warning("Indexation automatique impossible (essai %d) : %s", attempt, exc)
            time.sleep(delay)


@asynccontextmanager
async def lifespan(app: FastAPI):
    services = get_services()
    if services.settings.auto_ingest:
        threading.Thread(target=_auto_ingest, args=(services,), daemon=True).start()
    yield


app = FastAPI(
    title="Assistant RAG local",
    description="Questions-réponses sur mes cours avec Ollama + Qdrant, 100 % local.",
    version=__version__,
    lifespan=lifespan,
)
app.include_router(openai_router)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, examples=["Qu'est-ce que l'écart-type ?"])
    matiere: str | None = Field(default=None, description="Filtre optionnel par matière")
    top_k: int | None = Field(default=None, gt=0, le=50)


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_s: float


class IngestRequest(BaseModel):
    prune: bool = Field(default=True, description="Supprimer de l'index les fichiers disparus")


@app.get("/health")
def health(services: ServicesDep) -> dict:
    s = services.settings
    status = {"api": __version__, "ollama": False, "qdrant": False}
    try:
        models = services.llm.list_models()
        status["ollama"] = True
        # « nomic-embed-text » est listé par Ollama sous le nom « nomic-embed-text:latest »
        status["models"] = {
            m: m in models or f"{m}:latest" in models for m in (s.llm_model, s.embed_model)
        }
    except Exception as exc:
        status["ollama_error"] = str(exc)
    try:
        status["chunks_indexes"] = services.store.count()
        status["qdrant"] = True
    except Exception as exc:
        status["qdrant_error"] = str(exc)
    status["status"] = "ok" if status["ollama"] and status["qdrant"] else "degraded"
    return status


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, services: ServicesDep) -> AskResponse:
    result = services.pipeline.answer(request.question, request.top_k, request.matiere)
    return AskResponse(answer=result.answer, sources=result.sources, latency_s=result.latency_s)


@app.post("/ingest")
def run_ingest(services: ServicesDep, request: IngestRequest | None = None) -> dict:
    if not ingest_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Une indexation est déjà en cours.")
    try:
        report = _run_ingest(services, prune=request.prune if request else True)
    finally:
        ingest_lock.release()
    return asdict(report)


@app.get("/matieres")
def matieres(services: ServicesDep) -> list[str]:
    return services.store.matieres()
