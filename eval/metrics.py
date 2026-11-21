"""Métriques d'évaluation (fonctions pures, testées dans tests/test_eval_metrics.py)."""

from __future__ import annotations

import re
import statistics
import unicodedata

REFUSAL_PATTERNS = ("je ne sais pas", "n'est pas dans tes documents", "pas dans le contexte")


def is_relevant(source: str, page: int, expected_source: str, expected_page: int | None) -> bool:
    """Pertinent si le passage vient du bon fichier (et de la bonne page si elle est donnée)."""
    source_ok = source == expected_source or source.endswith("/" + expected_source)
    return source_ok and (expected_page is None or page == expected_page)


def first_relevant_rank(
    retrieved: list[tuple[str, int]], expected_source: str, expected_page: int | None
) -> int | None:
    """Rang (à partir de 1) du premier passage pertinent, ou None s'il est absent."""
    for rank, (source, page) in enumerate(retrieved, start=1):
        if is_relevant(source, page, expected_source, expected_page):
            return rank
    return None


def hit_rate(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mrr(ranks: list[int | None]) -> float:
    return sum(1 / r for r in ranks if r is not None) / len(ranks) if ranks else 0.0


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower()).replace("’", "'")
    return "".join(c for c in text if not unicodedata.combining(c))


def is_refusal(answer: str) -> bool:
    normalized = _normalize(answer)
    return any(_normalize(p) in normalized for p in REFUSAL_PATTERNS)


def parse_judge_scores(text: str) -> tuple[int | None, int | None]:
    """Extrait les notes de 0 à 2 de la réponse du LLM-juge (« FIDELITE: 2 / PERTINENCE: 1 »)."""
    normalized = _normalize(text)
    labelled = [
        re.search(rf"{label}\W*([0-2])\b", normalized) for label in ("fidelite", "pertinence")
    ]
    if all(labelled):
        return int(labelled[0].group(1)), int(labelled[1].group(1))
    found = re.findall(r"\b([0-2])\b", normalized)
    if len(found) < 2:
        return None, None
    return int(found[0]), int(found[1])


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]
