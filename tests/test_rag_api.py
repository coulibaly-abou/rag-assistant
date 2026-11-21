import json

from app.rag import NO_ANSWER, SYSTEM_PROMPT, build_messages, format_sources_markdown, to_sources
from app.store import SearchHit
from tests.conftest import HITS


def test_build_messages_contient_contexte_et_question():
    messages = build_messages("Qu'est-ce que la médiane ?", HITS)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = messages[1]["content"]
    assert "[1] (Stats/cours.pdf, page 3)" in user
    assert "[2] (Python/cours.md, page 1)" in user
    assert user.rstrip().endswith("RÉPONSE :")
    assert "Qu'est-ce que la médiane ?" in user


def test_prompt_systeme_impose_francais_et_je_ne_sais_pas():
    assert "français" in SYSTEM_PROMPT
    assert NO_ANSWER in SYSTEM_PROMPT


def test_sources_et_markdown():
    long_hit = SearchHit("x" * 500, "a.pdf", 2, "m", 0.123456)
    sources = to_sources([long_hit])
    assert sources[0].extrait.endswith("…") and len(sources[0].extrait) == 301
    assert sources[0].score == 0.1235
    assert "[1] a.pdf, page 2" in format_sources_markdown(sources)
    assert format_sources_markdown([]) == ""


def test_pipeline_sans_resultat_ne_appelle_pas_le_llm(fakes):
    fakes.store.hits = []
    result = fakes.pipeline.answer("Question hors cours ?")
    assert result.answer == NO_ANSWER
    assert result.sources == []
    assert fakes.llm.calls == []


def test_pipeline_min_score(fakes):
    fakes.pipeline.min_score = 0.5
    hits = fakes.pipeline.retrieve("médiane")
    assert [h.source for h in hits] == ["Stats/cours.pdf"]


# --- API ---------------------------------------------------------------------------


def test_ask(client):
    response = client.post("/ask", json={"question": "Qu'est-ce que la médiane ?"})
    assert response.status_code == 200
    data = response.json()
    assert "médiane" in data["answer"]
    assert data["sources"][0]["source"] == "Stats/cours.pdf"
    assert data["sources"][0]["page"] == 3


def test_ask_filtre_matiere(client, fakes):
    response = client.post("/ask", json={"question": "liste ?", "matiere": "Python"})
    assert fakes.store.last_matiere == "Python"
    assert [s["matiere"] for s in response.json()["sources"]] == ["Python"]


def test_ask_question_vide(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_health(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["models"] == {"llama3.2:3b": True, "nomic-embed-text": True}
    assert data["chunks_indexes"] == 2


def test_matieres(client):
    assert client.get("/matieres").json() == ["Python", "Stats"]


# --- Compatibilité OpenAI ----------------------------------------------------------


def test_v1_models_une_entree_par_matiere(client):
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["rag-cours", "rag-cours/Python", "rag-cours/Stats"]


def test_v1_chat_completions_sans_streaming(client, fakes):
    body = {
        "model": "rag-cours/Stats",
        "messages": [{"role": "user", "content": "Qu'est-ce que la médiane ?"}],
    }
    data = client.post("/v1/chat/completions", json=body).json()
    content = data["choices"][0]["message"]["content"]
    assert "**Sources :**" in content and "Stats/cours.pdf, page 3" in content
    assert fakes.store.last_matiere == "Stats"


def test_v1_chat_completions_streaming(client):
    body = {
        "model": "rag-cours",
        "stream": True,
        "messages": [
            {"role": "user", "content": "Bonjour"},
            {"role": "assistant", "content": "Bonjour !"},
            {"role": "user", "content": [{"type": "text", "text": "La médiane ?"}]},
        ],
    }
    response = client.post("/v1/chat/completions", json=body)
    lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text.startswith("La médiane partage la série [1].")
    assert "**Sources :**" in text
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_v1_taches_openwebui_sans_rag(client, fakes):
    body = {
        "model": "rag-cours",
        "messages": [{"role": "user", "content": "### Task:\nGénère un titre court."}],
    }
    content = client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]
    assert "Sources" not in content["content"]
    assert "CONTEXTE" not in fakes.llm.calls[-1][-1]["content"]
