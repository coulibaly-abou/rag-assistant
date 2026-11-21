"""Évaluation du RAG sur un jeu de questions annoté.

À lancer dans le conteneur de l'API (qui a accès à Ollama, Qdrant et data/) :

    docker compose exec api python -m eval.run_eval --tag reference
    docker compose exec api python -m eval.run_eval --chunk-size 500 --tag chunk500
    docker compose exec api python -m eval.run_eval --mode no-rag --tag sans_rag
    docker compose exec api python -m eval.run_eval --mode retrieval --top-k 3 --tag k3

Chaque exécution :
- ajoute une ligne de synthèse dans eval/resultats/experiences.csv ;
- écrit le détail question par question dans eval/resultats/details/.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.embeddings import OllamaEmbedder
from app.ingest import ingest
from app.llm import OllamaLLM
from app.rag import NO_ANSWER, RagPipeline, build_messages
from app.store import VectorStore
from eval.metrics import (
    first_relevant_rank,
    hit_rate,
    is_refusal,
    mrr,
    parse_judge_scores,
    percentile,
)

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "resultats"
SUMMARY_FILE = RESULTS_DIR / "experiences.csv"

NO_RAG_PROMPT = "Tu es un assistant de révision. Réponds en français, de façon claire et concise."

JUDGE_PROMPT = """Tu es un correcteur strict. Évalue la RÉPONSE d'un assistant.

QUESTION : {question}
RÉPONSE ATTENDUE : {expected}
CONTEXTE FOURNI À L'ASSISTANT :
{context}

RÉPONSE DE L'ASSISTANT : {answer}

Donne deux notes entières de 0 à 2 :
- FIDELITE : 2 si toutes les affirmations de la réponse sont justifiées par le contexte, \
1 si certaines le sont, 0 si la réponse invente.
- PERTINENCE : 2 si la réponse répond correctement à la question (comparée à la réponse attendue), \
1 si partiellement, 0 sinon.

Réponds uniquement sous la forme :
FIDELITE: <note>
PERTINENCE: <note>"""


@dataclass
class Question:
    question: str
    expected_answer: str
    expected_source: str
    expected_page: int | None
    out_of_corpus: bool


def load_questions(path: Path) -> list[Question]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    questions = []
    for row in rows:
        if not row.get("question", "").strip():
            continue
        page = (row.get("page_attendue") or "").strip()
        questions.append(
            Question(
                question=row["question"].strip(),
                expected_answer=(row.get("reponse_attendue") or "").strip(),
                expected_source=(row.get("source_attendue") or "").strip(),
                expected_page=int(page) if page else None,
                out_of_corpus=(row.get("hors_corpus") or "0").strip() in {"1", "oui", "true"},
            )
        )
    return questions


def collection_name(embed_model: str, chunk_size: int, chunk_overlap: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", embed_model.lower()).strip("_")
    return f"eval_{slug}_{chunk_size}_{chunk_overlap}"


def parse_args() -> argparse.Namespace:
    s = get_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.csv")
    parser.add_argument("--tag", default="", help="Nom court de l'expérience (ex. chunk500)")
    parser.add_argument(
        "--mode",
        choices=["rag", "no-rag", "retrieval"],
        default="rag",
        help="rag : pipeline complet | no-rag : LLM seul (baseline) | retrieval : recherche seule",
    )
    parser.add_argument("--chunk-size", type=int, default=s.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=s.chunk_overlap)
    parser.add_argument("--top-k", type=int, default=s.top_k)
    parser.add_argument(
        "--search-mode",
        choices=["vector", "hybrid"],
        default=s.search_mode,
        help="vector : sémantique seule | hybrid : sémantique + BM25 (fusion RRF)",
    )
    parser.add_argument("--embed-model", default=s.embed_model)
    parser.add_argument("--llm-model", default=s.llm_model)
    parser.add_argument("--judge", action="store_true", help="Noter fidélité/pertinence (LLM-juge)")
    parser.add_argument("--judge-model", default=s.llm_model)
    parser.add_argument("--limit", type=int, default=None, help="N'évaluer que les N premières")
    return parser.parse_args()


def with_retry(fn, attempts: int = 3, delay: float = 20.0):
    """Ollama peut renvoyer une erreur 500 si la mémoire manque : on réessaie après une pause."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f"  essai {attempt} échoué ({exc}), nouvel essai dans {delay:.0f} s", flush=True)
            time.sleep(delay)


def evaluate_question(q: Question, row: dict, args, pipeline, llm, judge) -> None:
    hits = []
    if args.mode != "no-rag":
        start = time.perf_counter()
        hits = with_retry(lambda: pipeline.retrieve(q.question, top_k=args.top_k))
        row["latence_recherche_s"] = round(time.perf_counter() - start, 3)
        row["sources_trouvees"] = " | ".join(f"{h.source}#p{h.page}" for h in hits)
        if not q.out_of_corpus and q.expected_source:
            retrieved = [(h.source, h.page) for h in hits]
            row["rang"] = first_relevant_rank(retrieved, q.expected_source, q.expected_page)

    if args.mode == "retrieval":
        return

    start = time.perf_counter()
    if args.mode == "rag":
        messages = build_messages(q.question, hits) if hits else None
    else:
        messages = [
            {"role": "system", "content": NO_RAG_PROMPT},
            {"role": "user", "content": q.question},
        ]
    answer = with_retry(lambda: llm.chat(messages)).strip() if messages else NO_ANSWER
    # en mode rag, la latence inclut la recherche (comme pour l'utilisateur)
    row["latence_s"] = round(time.perf_counter() - start + row.get("latence_recherche_s", 0), 2)
    row["reponse"] = answer
    row["refus"] = int(is_refusal(answer))

    if args.judge and not q.out_of_corpus:
        context = "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, 1))
        prompt = JUDGE_PROMPT.format(
            question=q.question,
            expected=q.expected_answer,
            context=context or "(aucun contexte)",
            answer=answer,
        )
        verdict = with_retry(lambda: judge.chat([{"role": "user", "content": prompt}]))
        row["fidelite"], row["pertinence"] = parse_judge_scores(verdict)


def run(args: argparse.Namespace) -> dict:
    s = get_settings()
    questions = load_questions(args.questions)[: args.limit]
    if not questions:
        raise SystemExit(f"Aucune question dans {args.questions}")

    embedder = OllamaEmbedder(s.ollama_url, args.embed_model, s.request_timeout)
    store = VectorStore(
        s.qdrant_url, collection_name(args.embed_model, args.chunk_size, args.chunk_overlap)
    )
    llm = OllamaLLM(s.ollama_url, args.llm_model, s.request_timeout, s.temperature, s.num_ctx)
    judge = OllamaLLM(s.ollama_url, args.judge_model, s.request_timeout, 0.0, s.num_ctx)
    pipeline = RagPipeline(
        embedder,
        store,
        llm,
        top_k=args.top_k,
        min_score=s.min_score,
        search_mode=args.search_mode,
        hybrid_candidates=s.hybrid_candidates,
    )

    if args.mode != "no-rag":
        print(f"Collection d'évaluation : {store.collection}")
        report = ingest(Path(s.data_dir), embedder, store, args.chunk_size, args.chunk_overlap)
        print(f"  {len(report.indexed)} fichiers indexés, {len(report.skipped)} inchangés")

    details = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q.question[:70]}", flush=True)
        row = {
            "question": q.question,
            "hors_corpus": int(q.out_of_corpus),
            "source_attendue": q.expected_source,
            "page_attendue": q.expected_page or "",
        }
        try:
            evaluate_question(q, row, args, pipeline, llm, judge)
        except Exception as exc:  # une question en échec ne fait pas perdre les autres
            print(f"  ERREUR : {exc}", flush=True)
            row["erreur"] = str(exc)[:300]
        details.append(row)

    summary = summarize(details, args)
    write_results(details, summary)
    return summary


def _mean(values: list) -> float | str:
    values = [v for v in values if v is not None and v != ""]
    return round(statistics.mean(values), 3) if values else ""


def summarize(details: list[dict], args: argparse.Namespace) -> dict:
    in_corpus = [d for d in details if not d["hors_corpus"]]
    out_corpus = [d for d in details if d["hors_corpus"]]
    ranks = [d.get("rang") for d in in_corpus if "rang" in d]
    latencies = [d["latence_s"] for d in details if "latence_s" in d]

    summary = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tag": args.tag or args.mode,
        "mode": args.mode,
        "llm": args.llm_model if args.mode != "retrieval" else "",
        "embedding": args.embed_model if args.mode != "no-rag" else "",
        "chunk_size": args.chunk_size if args.mode != "no-rag" else "",
        "chunk_overlap": args.chunk_overlap if args.mode != "no-rag" else "",
        "top_k": args.top_k if args.mode != "no-rag" else "",
        "recherche": args.search_mode if args.mode != "no-rag" else "",
        "n_questions": len(details),
        "erreurs": sum(1 for d in details if "erreur" in d),
        "hit_rate": round(hit_rate(ranks, args.top_k), 3) if ranks else "",
        "mrr": round(mrr(ranks), 3) if ranks else "",
        "refus_correct": _mean([d["refus"] for d in out_corpus if "refus" in d]),
        "faux_refus": _mean([d["refus"] for d in in_corpus if "refus" in d]),
        "fidelite_0_2": _mean([d.get("fidelite") for d in in_corpus]),
        "pertinence_0_2": _mean([d.get("pertinence") for d in in_corpus]),
        "latence_mediane_s": round(statistics.median(latencies), 2) if latencies else "",
        "latence_p95_s": round(percentile(latencies, 95), 2) if latencies else "",
    }
    return summary


def write_results(details: list[dict], summary: dict) -> None:
    details_dir = RESULTS_DIR / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = details_dir / f"{stamp}_{summary['tag']}.csv"
    fields = list(dict.fromkeys(key for row in details for key in row))
    with detail_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(details)

    # Synthèse : on relit les lignes existantes pour accepter l'ajout de nouvelles colonnes
    rows = []
    if SUMMARY_FILE.exists():
        with SUMMARY_FILE.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f, delimiter=";"))
    rows.append(summary)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with SUMMARY_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";", restval="")
        writer.writeheader()
        writer.writerows(rows)

    print("\nRésultats :")
    for key, value in summary.items():
        print(f"  {key:<18} {value}")
    print(f"\nDétail : {detail_path}\nSynthèse : {SUMMARY_FILE}")


if __name__ == "__main__":
    run(parse_args())
