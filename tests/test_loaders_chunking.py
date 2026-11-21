from pathlib import Path

import pytest

from app.chunking import chunk_document, split_words
from app.loaders import (
    DEFAULT_MATIERE,
    Document,
    Page,
    clean_text,
    file_sha256,
    infer_matiere,
    iter_files,
    load_document,
)

# --- Découpage -----------------------------------------------------------------


def test_split_words_chevauchement():
    text = " ".join(f"m{i}" for i in range(10))
    chunks = split_words(text, size=4, overlap=2)
    assert chunks == ["m0 m1 m2 m3", "m2 m3 m4 m5", "m4 m5 m6 m7", "m6 m7 m8 m9"]


def test_split_words_texte_court_et_vide():
    assert split_words("un deux", size=5, overlap=1) == ["un deux"]
    assert split_words("   ", size=5, overlap=1) == []


def test_split_words_couvre_tous_les_mots():
    words = [f"w{i}" for i in range(103)]
    chunks = split_words(" ".join(words), size=20, overlap=5)
    assert chunks[-1].split()[-1] == "w102"
    assert all(len(c.split()) <= 20 for c in chunks)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (5, 5), (5, -1)])
def test_split_words_parametres_invalides(size, overlap):
    with pytest.raises(ValueError):
        split_words("a b c", size=size, overlap=overlap)


def test_chunk_document_garde_les_pages():
    doc = Document(
        source="x.pdf",
        matiere="m",
        file_hash="h",
        pages=[Page("a b c d e", 1), Page("f g", 2)],
    )
    chunks = chunk_document(doc, size=3, overlap=1)
    assert [(c.page, c.index) for c in chunks] == [(1, 0), (1, 1), (2, 2)]


# --- Chargement ----------------------------------------------------------------


def test_clean_text():
    assert clean_text("statis-\ntique   descrip­tive\n\n\n\nfin") == (
        "statistique descriptive\n\nfin"
    )


def test_infer_matiere(tmp_path: Path):
    assert infer_matiere(tmp_path / "Stats" / "cours.pdf", tmp_path) == "Stats"
    assert infer_matiere(tmp_path / "Stats" / "td" / "cours.pdf", tmp_path) == "Stats"
    assert infer_matiere(tmp_path / "cours.pdf", tmp_path) == DEFAULT_MATIERE


def test_iter_files_filtre(tmp_path: Path):
    (tmp_path / "Stats").mkdir()
    (tmp_path / "Stats" / "cours.md").write_text("x", encoding="utf-8")
    (tmp_path / "Stats" / "image.png").write_bytes(b"x")
    (tmp_path / "Stats" / "~$brouillon.docx").write_bytes(b"x")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "notes.md").write_text("x", encoding="utf-8")
    assert [p.name for p in iter_files(tmp_path)] == ["cours.md"]
    assert iter_files(tmp_path / "absent") == []


def test_load_markdown(tmp_path: Path):
    path = tmp_path / "Python" / "cours.md"
    path.parent.mkdir()
    path.write_text("# Titre\n\nUne liste est modifiable.", encoding="utf-8")
    doc = load_document(path, tmp_path)
    assert doc.source == "Python/cours.md"
    assert doc.matiere == "Python"
    assert doc.file_hash == file_sha256(path)
    assert doc.pages[0].page == 1
    assert "liste est modifiable" in doc.pages[0].text


def test_load_pdf_page_par_page(tmp_path: Path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "cours.pdf"
    pdf = pymupdf.open()
    for text in ("Page un : la moyenne", "Page deux : la variance"):
        page = pdf.new_page()
        page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()

    doc = load_document(path, tmp_path)
    assert [p.page for p in doc.pages] == [1, 2]
    assert "variance" in doc.pages[1].text


def test_load_docx(tmp_path: Path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "rapport.docx"
    document = docx.Document()
    document.add_paragraph("Introduction du rapport")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Indicateur"
    table.rows[0].cells[1].text = "Médiane"
    document.save(str(path))

    doc = load_document(path, tmp_path)
    assert "Introduction du rapport" in doc.pages[0].text
    assert "Indicateur | Médiane" in doc.pages[0].text
