"""Plan d'expériences : une configuration de référence, puis un seul paramètre changé à la fois.

    docker compose exec api python -m eval.plan_experiences            # recherche seule (rapide)
    docker compose exec api python -m eval.plan_experiences --complet  # + génération et baselines

Ensuite : docker compose exec api python -m eval.plot_results
"""

from __future__ import annotations

import argparse

from app.config import get_settings
from eval.run_eval import EVAL_DIR, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complet", action="store_true", help="Inclure la génération (plus long)")
    parser.add_argument(
        "--judge", action="store_true", help="Ajouter le LLM-juge (encore plus long)"
    )
    cli = parser.parse_args()

    s = get_settings()
    reference = {
        "questions": EVAL_DIR / "questions.csv",
        "mode": "rag" if cli.complet else "retrieval",
        "chunk_size": 300,
        "chunk_overlap": 50,
        "top_k": 5,
        "search_mode": "vector",
        "embed_model": s.embed_model,
        "llm_model": s.llm_model,
        "judge": cli.judge,
        "judge_model": s.llm_model,
        "limit": None,
    }

    experiences = [
        ("reference", {}),
        ("chunk500", {"chunk_size": 500}),
        ("chunk800", {"chunk_size": 800, "chunk_overlap": 80}),
        ("top_k3", {"top_k": 3}),
        ("top_k8", {"top_k": 8}),
        ("hybride", {"search_mode": "hybrid"}),
        ("hybride_k3", {"search_mode": "hybrid", "top_k": 3}),
    ]
    if cli.complet:
        experiences.append(("sans_rag", {"mode": "no-rag"}))

    for tag, changes in experiences:
        print(f"\n===== Expérience : {tag} =====")
        run(argparse.Namespace(**{**reference, **changes, "tag": tag}))


if __name__ == "__main__":
    main()
