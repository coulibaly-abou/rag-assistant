"""Lecture des documents (PDF, DOCX, Markdown, texte) page par page."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}
DEFAULT_MATIERE = "general"


@dataclass
class Page:
    text: str
    page: int  # numéro de page, à partir de 1


@dataclass
class Document:
    source: str  # chemin relatif au dossier data/, au format POSIX
    matiere: str
    file_hash: str
    pages: list[Page]


def file_sha256(path: Path) -> str:
    """Empreinte du fichier : sert à ne pas réindexer un fichier inchangé."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    """Nettoie le texte extrait : césures, espaces multiples, lignes vides."""
    text = text.replace("­", "").replace(" ", " ").replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # mot coupé en fin de ligne
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[Page]:
    import pymupdf

    pages: list[Page] = []
    with pymupdf.open(path) as pdf:
        for number, page in enumerate(pdf, start=1):
            pages.append(Page(text=page.get_text("text"), page=number))
    return pages


def load_docx(path: Path) -> list[Page]:
    """Un DOCX n'a pas de pages fixes : tout le texte est rattaché à la page 1."""
    import docx

    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return [Page(text="\n".join(parts), page=1)]


def load_text(path: Path) -> list[Page]:
    return [Page(text=path.read_text(encoding="utf-8", errors="replace"), page=1)]


def infer_matiere(path: Path, data_dir: Path) -> str:
    """La matière est le nom du premier sous-dossier de data/ (ex. data/Statistiques/cours.pdf)."""
    relative = path.relative_to(data_dir)
    return relative.parts[0] if len(relative.parts) > 1 else DEFAULT_MATIERE


def relative_source(path: Path, data_dir: Path) -> str:
    return path.relative_to(data_dir).as_posix()


def iter_files(data_dir: Path) -> list[Path]:
    """Liste triée des fichiers pris en charge (fichiers cachés et temporaires Office exclus)."""
    if not data_dir.exists():
        return []
    files = []
    for path in data_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if any(part.startswith(".") for part in path.relative_to(data_dir).parts):
            continue
        if path.name.startswith("~$"):
            continue
        files.append(path)
    return sorted(files)


def load_document(path: Path, data_dir: Path, file_hash: str | None = None) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw_pages = load_pdf(path)
    elif suffix == ".docx":
        raw_pages = load_docx(path)
    elif suffix in SUPPORTED_EXTENSIONS:
        raw_pages = load_text(path)
    else:
        raise ValueError(f"Format non pris en charge : {path.suffix}")

    pages = [Page(text=clean_text(p.text), page=p.page) for p in raw_pages]
    pages = [p for p in pages if p.text]
    return Document(
        source=relative_source(path, data_dir),
        matiere=infer_matiere(path, data_dir),
        file_hash=file_hash or file_sha256(path),
        pages=pages,
    )
