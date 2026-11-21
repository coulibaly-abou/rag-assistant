from app.hybrid import BM25Index, rrf_fuse, tokenize
from app.store import SearchHit


def hit(source: str, index: int, text: str = "", matiere: str = "m", score: float = 0.0):
    return SearchHit(text, source, 1, matiere, score, index)


CHUNKS = [
    hit("python.md", 0, "La méthode loc sélectionne par étiquettes, iloc par positions.", "Python"),
    hit("python.md", 1, "Une liste est modifiable, un tuple est immuable.", "Python"),
    hit("stats.md", 0, "La médiane partage la série ordonnée en deux.", "Stats"),
    hit("stats.md", 1, "La règle des 68-95-99,7 concerne la loi normale.", "Stats"),
]


def test_tokenize_sans_accents_ni_mots_vides():
    assert tokenize("Quelle est la différence entre loc et iloc ?") == [
        "difference",
        "entre",
        "loc",
        "iloc",
    ]
    assert tokenize("Règle des 68-95-99,7") == ["regle", "68", "95", "99", "7"]


def test_bm25_trouve_le_mot_exact():
    index = BM25Index(CHUNKS)
    results = index.search("Différence entre loc et iloc ?", top_k=2)
    assert results[0].key == ("python.md", 0)
    assert results[0].score > 0


def test_bm25_filtre_matiere_et_aucun_resultat():
    index = BM25Index(CHUNKS)
    assert index.search("iloc", top_k=3, matiere="Stats") == []
    assert index.search("photosynthèse", top_k=3) == []
    assert BM25Index([]).search("iloc", top_k=3) == []


def test_rrf_fuse_favorise_les_passages_presents_dans_les_deux_listes():
    a, b, c = hit("f", 0), hit("f", 1), hit("f", 2)
    dense = [a, b, c]
    sparse = [c, b]
    fused = rrf_fuse([dense, sparse], top_k=3)
    # c : 1/63 + 1/61 = 0.03227 ; b : 1/62 + 1/62 = 0.03226 ; a : 1/61 seulement = 0.01639
    assert [h.key for h in fused] == [("f", 2), ("f", 1), ("f", 0)]
    assert fused[0].score > fused[1].score > fused[2].score
    assert len(rrf_fuse([dense, sparse], top_k=1)) == 1


def test_pipeline_hybride(fakes):
    # Le faux store renvoie toujours le même ordre « vectoriel » : python.md#0 en premier.
    # BM25 trouve « liste … modifiable » dans python.md#1, qui passe donc en tête après fusion.
    fakes.store.hits = list(CHUNKS)
    fakes.pipeline.search_mode = "hybrid"
    hits = fakes.pipeline.retrieve("Une liste est-elle modifiable ?", top_k=2)
    assert hits[0].key == ("python.md", 1)
    # l'index BM25 est mis en cache tant que le contenu ne change pas
    index = fakes.pipeline._bm25_index()
    assert fakes.pipeline._bm25_index() is index
