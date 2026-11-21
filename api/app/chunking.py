"""Découpage du texte en chunks de taille fixe (en mots) avec chevauchement.

Le découpage se fait page par page : chaque chunk garde ainsi son numéro de page exact.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.loaders import Document


@dataclass
class Chunk:
    text: str
    page: int
    index: int  # position du chunk dans le document


def split_words(text: str, size: int, overlap: int) -> list[str]:
    """Découpe `text` en fenêtres de `size` mots qui se chevauchent de `overlap` mots."""
    if size <= 0:
        raise ValueError("size doit être > 0")
    if not 0 <= overlap < size:
        raise ValueError("overlap doit être compris entre 0 et size - 1")

    words = text.split()
    if not words:
        return []

    step = size - overlap
    chunks = []
    start = 0
    while True:
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks


def chunk_document(document: Document, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in document.pages:
        for text in split_words(page.text, size, overlap):
            chunks.append(Chunk(text=text, page=page.page, index=len(chunks)))
    return chunks
