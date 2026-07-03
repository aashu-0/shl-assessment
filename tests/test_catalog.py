"""Catalog loading, test-type mapping and the id-grounding guardrail."""

from app.catalog import KEY_TO_CODE, get_catalog, load_catalog


def test_catalog_loads_nonempty():
    cat = get_catalog()
    assert len(cat) > 300  # ~377 individual test solutions


def test_records_have_real_urls():
    cat = get_catalog()
    for a in cat:
        assert a.url.startswith("http")
        assert a.entity_id
        assert a.name


def test_test_type_mapping():
    cat = get_catalog()
    a = next(iter(cat))
    for k in a.keys:
        assert k in KEY_TO_CODE
    # Every code produced is a single known letter.
    codes = a.test_type.split(",") if a.test_type else []
    assert all(c in KEY_TO_CODE.values() for c in codes)


def test_resolve_ids_drops_unknown():
    cat = get_catalog()
    known = next(iter(cat)).entity_id
    resolved = cat.resolve_ids([known, "does-not-exist-999999", known])
    # Unknown id dropped; duplicate collapsed.
    assert [a.entity_id for a in resolved] == [known]


def test_by_name_lookup():
    cat = get_catalog()
    sample = next(iter(cat))
    assert cat.by_name(sample.name.upper()) is sample
