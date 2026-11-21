"""Recherche hybride : mots-clés (BM25) + sémantique (vecteurs), fusionnés par RRF.

- La recherche vectorielle retrouve les passages de même *sens*, même avec d'autres mots.
- BM25 retrouve les passages qui contiennent les *mots exacts* de la question
  (termes techniques, noms de fonctions comme « iloc », sigles…), où les embeddings sont faibles.
- La fusion RRF (Reciprocal Rank Fusion) combine les deux classements :
  score(d) = somme sur les listes de 1 / (k + rang de d), avec k = 60.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace

from rank_bm25 import BM25Okapi

from app.store import SearchHit

RRF_K = 60

_STOPWORDS_TEXT = """a au aux avec ce ces cet cette d dans de des du elle en est et etre il ils je
l la le les leur lui m ma mais me meme mes moi mon n ne nos notre nous on ou par pas pour qu que qui
s sa se ses son sont sur t ta te tes toi ton tu un une vos votre vous y quel quelle quels quelles
comment pourquoi quand c ca peut fait faire"""
STOPWORDS = set(_STOPWORDS_TEXT.split())


def tokenize(text: str) -> list[str]:
    """Minuscules, sans accents, sans mots vides. Les nombres sont gardés (« 68 », « 95 »)."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9_]+", text) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[SearchHit]):
        self.chunks = chunks
        corpus = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None

    def search(self, query: str, top_k: int, matiere: str | None = None) -> list[SearchHit]:
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked:
            if scores[i] <= 0:
                break
            chunk = self.chunks[i]
            if matiere and chunk.matiere != matiere:
                continue
            results.append(replace(chunk, score=float(scores[i])))
            if len(results) == top_k:
                break
        return results


def rrf_fuse(result_lists: list[list[SearchHit]], top_k: int, k: int = RRF_K) -> list[SearchHit]:
    """Fusionne plusieurs classements ; le score renvoyé est le score RRF."""
    scores: dict[tuple, float] = {}
    hits: dict[tuple, SearchHit] = {}
    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            scores[hit.key] = scores.get(hit.key, 0.0) + 1.0 / (k + rank)
            hits.setdefault(hit.key, hit)
    best = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [replace(hits[key], score=round(scores[key], 5)) for key in best]
