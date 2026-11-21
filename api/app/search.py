"""Recherche manuelle dans l'index (sans LLM) : affiche les passages trouvés et leurs métadonnées.

Utilisation (dans le conteneur) :
    docker compose exec api python -m app.search "Qu'est-ce que la loi binomiale ?"
    docker compose exec api python -m app.search "variance" --matiere Statistiques_demo -k 3
    docker compose exec api python -m app.search --stats
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections import Counter

from app.config import get_settings
from app.embeddings import OllamaEmbedder
from app.store import VectorStore


def print_stats(store: VectorStore) -> None:
    """Résumé de l'index : nombre de chunks par fichier, avec matière et pages."""
    points, offset = [], None
    while True:
        batch, offset = store.client.scroll(
            store.collection, limit=256, offset=offset, with_payload=True, with_vectors=False
        )
        points.extend(batch)
        if offset is None:
            break

    by_source = Counter(p.payload["source"] for p in points)
    print(f"Collection « {store.collection} » : {len(points)} chunks, {len(by_source)} fichiers\n")
    for source in sorted(by_source):
        payloads = [p.payload for p in points if p.payload["source"] == source]
        pages = sorted({p["page"] for p in payloads})
        first = payloads[0]
        print(f"- {source}")
        print(f"    matière : {first['matiere']} | chunks : {by_source[source]} | pages : {pages}")
        print(f"    hash : {first['file_hash'][:16]}… | indexé le : {first['indexed_at']}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Recherche manuelle dans Qdrant")
    parser.add_argument("question", nargs="?", help="Question ou mots-clés")
    parser.add_argument("-k", type=int, default=None, help="Nombre de passages (défaut : TOP_K)")
    parser.add_argument("--matiere", default=None, help="Filtrer sur une matière")
    parser.add_argument("--stats", action="store_true", help="Afficher le contenu de l'index")
    args = parser.parse_args()

    s = get_settings()
    store = VectorStore(s.qdrant_url, s.collection_name)
    if not store.exists():
        raise SystemExit("Index vide : lance d'abord python -m app.ingest")

    if args.stats or not args.question:
        print_stats(store)
        return

    embedder = OllamaEmbedder(s.ollama_url, s.embed_model, s.request_timeout)
    hits = store.search(embedder.embed_query(args.question), args.k or s.top_k, args.matiere)
    print(f"Question : {args.question}\n")
    for rank, hit in enumerate(hits, 1):
        print(f"#{rank}  score {hit.score:.3f} | {hit.matiere} | {hit.source} | page {hit.page}")
        excerpt = hit.text if len(hit.text) <= 400 else hit.text[:400] + "…"
        print(textwrap.indent(textwrap.fill(excerpt, 96), "    "), end="\n\n")


if __name__ == "__main__":
    main()
