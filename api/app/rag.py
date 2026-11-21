"""Phase B — Question / réponse : question → embedding → recherche top-k → prompt → LLM."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

from pydantic import BaseModel

from app.embeddings import OllamaEmbedder
from app.llm import OllamaLLM
from app.store import SearchHit, VectorStore

NO_ANSWER = "Je ne sais pas : cette information n'est pas dans tes documents."

SYSTEM_PROMPT = f"""Tu es un assistant de révision qui répond aux questions d'un étudiant \
sur ses cours et ses rapports.

Règles strictes :
1. Réponds toujours en français, de façon claire et concise.
2. Utilise UNIQUEMENT les informations du CONTEXTE ci-dessous. N'utilise jamais tes \
connaissances personnelles et n'invente rien.
3. Si le contexte ne contient pas la réponse, réponds exactement : « {NO_ANSWER} »
4. Cite tes sources avec leur numéro entre crochets, par exemple [1] ou [2][3].
"""


class Source(BaseModel):
    numero: int
    source: str
    page: int
    matiere: str
    score: float
    extrait: str


@dataclass
class RagAnswer:
    answer: str
    sources: list[Source]
    latency_s: float


def build_context(hits: list[SearchHit]) -> str:
    blocks = [f"[{i}] ({hit.source}, page {hit.page})\n{hit.text}" for i, hit in enumerate(hits, 1)]
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[SearchHit]) -> list[dict]:
    user = f"CONTEXTE :\n{build_context(hits)}\n\nQUESTION : {question}\n\nRÉPONSE :"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def to_sources(hits: list[SearchHit], excerpt_len: int = 300) -> list[Source]:
    sources = []
    for i, hit in enumerate(hits, 1):
        extrait = hit.text if len(hit.text) <= excerpt_len else hit.text[:excerpt_len] + "…"
        sources.append(
            Source(
                numero=i,
                source=hit.source,
                page=hit.page,
                matiere=hit.matiere,
                score=round(hit.score, 4),
                extrait=extrait,
            )
        )
    return sources


def format_sources_markdown(sources: list[Source]) -> str:
    if not sources:
        return ""
    lines = ["", "", "---", "**Sources :**"]
    lines += [f"- [{s.numero}] {s.source}, page {s.page} (score {s.score:.2f})" for s in sources]
    return "\n".join(lines)


class RagPipeline:
    def __init__(
        self,
        embedder: OllamaEmbedder,
        store: VectorStore,
        llm: OllamaLLM,
        top_k: int = 5,
        min_score: float = 0.0,
    ):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(
        self, question: str, top_k: int | None = None, matiere: str | None = None
    ) -> list[SearchHit]:
        vector = self.embedder.embed_query(question)
        hits = self.store.search(vector, top_k or self.top_k, matiere=matiere)
        return [h for h in hits if h.score >= self.min_score]

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        matiere: str | None = None,
        llm_model: str | None = None,
    ) -> RagAnswer:
        start = time.perf_counter()
        hits = self.retrieve(question, top_k, matiere)
        if not hits:
            return RagAnswer(NO_ANSWER, [], time.perf_counter() - start)
        text = self.llm.chat(build_messages(question, hits), model=llm_model)
        return RagAnswer(text.strip(), to_sources(hits), time.perf_counter() - start)

    def stream_answer(
        self, question: str, top_k: int | None = None, matiere: str | None = None
    ) -> tuple[list[Source], Iterator[str]]:
        hits = self.retrieve(question, top_k, matiere)
        if not hits:
            return [], iter([NO_ANSWER])
        return to_sources(hits), self.llm.stream_chat(build_messages(question, hits))
