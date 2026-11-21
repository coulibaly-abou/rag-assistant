"""Accès à la base vectorielle Qdrant : collection, insertion, recherche."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from qdrant_client import QdrantClient, models

from app.chunking import Chunk
from app.loaders import Document

KEYWORD_FIELDS = ("source", "matiere", "file_hash")


@dataclass
class SearchHit:
    text: str
    source: str
    page: int
    matiere: str
    score: float


def _match(**fields: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in fields.items()
        ]
    )


class VectorStore:
    def __init__(self, url: str, collection: str, timeout: int = 60):
        self.collection = collection
        self.client = QdrantClient(url=url, timeout=timeout)

    # --- Collection -------------------------------------------------------
    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self, dim: int) -> None:
        if self.exists():
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        for field in KEYWORD_FIELDS:
            self.client.create_payload_index(
                self.collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
            )

    def count(self) -> int:
        if not self.exists():
            return 0
        return self.client.count(self.collection, exact=True).count

    # --- Écriture ---------------------------------------------------------
    def is_indexed(self, source: str, file_hash: str) -> bool:
        if not self.exists():
            return False
        result = self.client.count(
            self.collection, count_filter=_match(source=source, file_hash=file_hash), exact=True
        )
        return result.count > 0

    def delete_source(self, source: str) -> None:
        if not self.exists():
            return
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=_match(source=source)),
        )

    def upsert(self, document: Document, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        indexed_at = datetime.now(UTC).isoformat(timespec="seconds")
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document.source}#{chunk.index}")),
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source": document.source,
                    "page": chunk.page,
                    "matiere": document.matiere,
                    "file_hash": document.file_hash,
                    "chunk_index": chunk.index,
                    "indexed_at": indexed_at,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[i : i + 128], wait=True)

    # --- Lecture ----------------------------------------------------------
    def _facet(self, key: str) -> list[str]:
        if not self.exists():
            return []
        response = self.client.facet(self.collection, key=key, limit=10_000, exact=True)
        return sorted(str(hit.value) for hit in response.hits)

    def sources(self) -> list[str]:
        return self._facet("source")

    def matieres(self) -> list[str]:
        return self._facet("matiere")

    def search(
        self, vector: list[float], top_k: int, matiere: str | None = None
    ) -> list[SearchHit]:
        if not self.exists():
            return []
        response = self.client.query_points(
            self.collection,
            query=vector,
            limit=top_k,
            query_filter=_match(matiere=matiere) if matiere else None,
            with_payload=True,
        )
        return [
            SearchHit(
                text=p.payload["text"],
                source=p.payload["source"],
                page=int(p.payload["page"]),
                matiere=p.payload["matiere"],
                score=float(p.score),
            )
            for p in response.points
        ]
