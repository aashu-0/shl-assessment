"""Lexical retrieval quality (BM25 path, no embeddings)."""

from app.catalog import get_catalog
from app.retriever import ANCHOR_ENTITY_IDS, Retriever


def _names(results):
    return " || ".join(a.name.lower() for a in results)


def test_docker_query_surfaces_docker_test():
    r = Retriever(get_catalog())
    results = r.search("Docker containers", top_k=15)
    assert any("docker" in a.name.lower() for a in results)


def test_java_query_surfaces_java_tests():
    r = Retriever(get_catalog())
    results = r.search("senior core java spring developer", top_k=20)
    assert any("java" in a.name.lower() for a in results)


def test_empty_query_returns_nothing():
    r = Retriever(get_catalog())
    assert r.search("", top_k=10) == []


def test_results_are_capped():
    r = Retriever(get_catalog())
    results = r.search("assessment", top_k=5)
    assert len(results) <= 5


def test_retrieve_always_includes_anchors():
    # Anchors (OPQ32r, Verify G+, ...) must be available for the LLM to pick,
    # even for a query with no lexical overlap with their names.
    r = Retriever(get_catalog())
    results = r.retrieve(["we are hiring plant operators, safety critical"], top_k=15)
    ids = {a.entity_id for a in results}
    assert set(ANCHOR_ENTITY_IDS).issubset(ids)


def test_anchor_ids_exist_in_catalog():
    cat = get_catalog()
    for aid in ANCHOR_ENTITY_IDS:
        assert cat.get(aid) is not None, f"anchor {aid} missing from catalog"
