import pytest

from eval.metrics import (
    first_relevant_rank,
    hit_rate,
    is_refusal,
    mrr,
    parse_judge_scores,
    percentile,
)
from eval.run_eval import collection_name, load_questions


def test_first_relevant_rank_avec_et_sans_page():
    retrieved = [("a.pdf", 1), ("Stats/b.pdf", 4), ("Stats/b.pdf", 2)]
    assert first_relevant_rank(retrieved, "Stats/b.pdf", 2) == 3
    assert first_relevant_rank(retrieved, "Stats/b.pdf", None) == 2
    assert first_relevant_rank(retrieved, "b.pdf", None) == 2
    assert first_relevant_rank(retrieved, "c.pdf", None) is None


def test_hit_rate_et_mrr():
    ranks = [1, 3, None, 2]
    assert hit_rate(ranks, k=2) == 0.5
    assert hit_rate(ranks, k=5) == 0.75
    assert mrr(ranks) == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)
    assert hit_rate([], 5) == 0.0 and mrr([]) == 0.0


def test_is_refusal():
    assert is_refusal("Je ne sais pas : cette information n'est pas dans tes documents.")
    assert is_refusal("JE NE SAIS PAS")
    assert not is_refusal("La médiane partage la série en deux.")


def test_parse_judge_scores():
    assert parse_judge_scores("FIDELITE: 2\nPERTINENCE: 1") == (2, 1)
    assert parse_judge_scores("Fidélité : 0 ; pertinence : 2") == (0, 2)
    assert parse_judge_scores("pas de note") == (None, None)


def test_percentile():
    assert percentile([], 95) == 0.0
    assert percentile([4.0], 95) == 4.0
    assert percentile([float(i) for i in range(1, 101)], 50) == pytest.approx(50.5)


def test_collection_name():
    assert collection_name("nomic-embed-text", 300, 50) == "eval_nomic_embed_text_300_50"


def test_questions_csv_valide():
    from eval.run_eval import EVAL_DIR

    questions = load_questions(EVAL_DIR / "questions.csv")
    assert len(questions) >= 10
    assert any(q.out_of_corpus for q in questions)
    assert all(q.expected_source for q in questions if not q.out_of_corpus)
