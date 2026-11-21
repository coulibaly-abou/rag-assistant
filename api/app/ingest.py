"""Phase A — Indexation : data/ → texte → chunks → embeddings → Qdrant.

Utilisation en ligne de commande (dans le conteneur) : python -m app.ingest
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from app.chunking import chunk_document
from app.embeddings import OllamaEmbedder
from app.loaders import file_sha256, iter_files, load_document, relative_source
from app.store import VectorStore

logger = logging.getLogger(__name__)

# Empêche deux indexations simultanées (démarrage automatique + appel à /ingest).
ingest_lock = threading.Lock()


@dataclass
class IngestReport:
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    chunks: int = 0


def ingest(
    data_dir: Path,
    embedder: OllamaEmbedder,
    store: VectorStore,
    chunk_size: int,
    chunk_overlap: int,
    prune: bool = True,
) -> IngestReport:
    report = IngestReport()
    files = iter_files(data_dir)

    if files and not store.exists():
        dim = len(embedder.embed_query("dimension"))
        store.ensure_collection(dim)

    current_sources = set()
    for path in files:
        source = relative_source(path, data_dir)
        current_sources.add(source)
        try:
            file_hash = file_sha256(path)
            if store.is_indexed(source, file_hash):
                report.skipped.append(source)
                continue

            document = load_document(path, data_dir, file_hash=file_hash)
            chunks = chunk_document(document, chunk_size, chunk_overlap)
            if not chunks:
                report.errors[source] = "aucun texte extrait (PDF scanné ?)"
                continue

            vectors = embedder.embed_documents([c.text for c in chunks])
            store.delete_source(source)  # anciens chunks d'une version précédente du fichier
            store.upsert(document, chunks, vectors)
            report.indexed.append(source)
            report.chunks += len(chunks)
            logger.info("Indexé : %s (%d chunks)", source, len(chunks))
        except Exception as exc:  # un fichier défectueux ne bloque pas les autres
            logger.exception("Erreur sur %s", source)
            report.errors[source] = str(exc)

    if prune:
        for source in set(store.sources()) - current_sources:
            store.delete_source(source)
            report.removed.append(source)

    return report


def main() -> None:
    from app.config import get_settings

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model, settings.request_timeout)
    store = VectorStore(settings.qdrant_url, settings.collection_name)
    report = ingest(
        Path(settings.data_dir), embedder, store, settings.chunk_size, settings.chunk_overlap
    )
    print(
        f"Indexés : {len(report.indexed)} | inchangés : {len(report.skipped)} | "
        f"supprimés : {len(report.removed)} | erreurs : {len(report.errors)} | "
        f"chunks : {report.chunks}"
    )
    for source, error in report.errors.items():
        print(f"  ERREUR {source} : {error}")


if __name__ == "__main__":
    main()
