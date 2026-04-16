import asyncio
from typing import Any

from app.tools.vector_store import LayerVectorStore


def _make_layer(name: str, title: str, abstract: str) -> dict[str, str]:
    return {"name": name, "title": title, "abstract": abstract}


def _make_catalog_row(name: str, de_title: str, en_title: str, de_abstract: str, en_abstract: str, aliases: list[str] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "de_title": de_title,
        "en_title": en_title,
        "de_abstract": de_abstract,
        "en_abstract": en_abstract,
        "aliases": aliases or [],
    }


# Deterministic fake embeddings: one-hot-ish vectors for testing ranking
_FAKE_EMBEDDINGS: dict[str, list[float]] = {}
_CALL_COUNT = 0


async def _fake_embed(texts: list[str], **kwargs) -> list[list[float]]:
    global _CALL_COUNT
    result = []
    for text in texts:
        _CALL_COUNT += 1
        # Create a distinct vector per unique text
        key = text.strip().lower()[:50]
        if key not in _FAKE_EMBEDDINGS:
            vec = [0.0] * 64
            idx = len(_FAKE_EMBEDDINGS) % 64
            vec[idx] = 1.0
            _FAKE_EMBEDDINGS[key] = vec
        result.append(_FAKE_EMBEDDINGS[key])
    return result


def test_vector_store_index_and_search() -> None:
    store = LayerVectorStore()
    layers = [
        _make_layer("city:hospitals", "Krankenhäuser", "Gesundheitsversorgung"),
        _make_layer("city:schools", "Schulen", "Bildungseinrichtungen"),
        _make_layer("env:flood_zones", "Hochwasserzonen", "Überschwemmungsgebiete"),
    ]
    catalog_rows = [
        _make_catalog_row("city:hospitals", "Krankenhäuser", "Hospitals", "Gesundheitsversorgung", "Healthcare facilities"),
        _make_catalog_row("city:schools", "Schulen", "Schools", "Bildungseinrichtungen", "Educational institutions"),
        _make_catalog_row("env:flood_zones", "Hochwasserzonen", "Flood Zones", "Überschwemmungsgebiete", "Flood risk areas"),
    ]

    asyncio.run(store.index_layers(layers, catalog_rows, _fake_embed))
    assert store.is_indexed()
    assert store.layer_count() == 3

    # Search returns results with expected shape
    query_vec = [0.0] * 64
    query_vec[0] = 1.0  # Similar to first indexed layer
    results = store.search(query_vec, top_k=2)
    assert len(results) == 2
    assert "layer_name" in results[0]
    assert "score" in results[0]
    assert "title" in results[0]
    assert "abstract" in results[0]


def test_vector_store_clear_resets_index() -> None:
    store = LayerVectorStore()
    layers = [_make_layer("city:roads", "Strassen", "Strassennetz")]
    catalog_rows = [_make_catalog_row("city:roads", "Strassen", "Roads", "Strassennetz", "Road network")]

    asyncio.run(store.index_layers(layers, catalog_rows, _fake_embed))
    assert store.is_indexed()

    store.clear()
    assert not store.is_indexed()
    assert store.layer_count() == 0


def test_vector_store_search_returns_sorted_by_score() -> None:
    store = LayerVectorStore()
    layers = [
        _make_layer("city:hospitals", "Krankenhäuser", "Gesundheitsversorgung"),
        _make_layer("city:schools", "Schulen", "Bildungseinrichtungen"),
    ]
    catalog_rows = [
        _make_catalog_row("city:hospitals", "Krankenhäuser", "Hospitals", "Gesundheitsversorgung", "Healthcare"),
        _make_catalog_row("city:schools", "Schulen", "Schools", "Bildungseinrichtungen", "Education"),
    ]

    asyncio.run(store.index_layers(layers, catalog_rows, _fake_embed))

    # Use the exact vector of first layer to ensure it scores highest
    query_vec = _FAKE_EMBEDDINGS[list(_FAKE_EMBEDDINGS.keys())[0]]
    results = store.search(query_vec, top_k=2)
    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]


def test_vector_store_top_k_limits_results() -> None:
    store = LayerVectorStore()
    layers = [
        _make_layer(f"ns:layer_{i}", f"Title {i}", f"Abstract {i}")
        for i in range(10)
    ]
    catalog_rows = [
        _make_catalog_row(f"ns:layer_{i}", f"Titel {i}", f"Title {i}", f"Beschreibung {i}", f"Abstract {i}")
        for i in range(10)
    ]

    asyncio.run(store.index_layers(layers, catalog_rows, _fake_embed))
    query_vec = [0.0] * 64
    query_vec[0] = 1.0

    results = store.search(query_vec, top_k=3)
    assert len(results) == 3


def test_vector_store_semantic_match_no_keyword_overlap() -> None:
    """A layer named 'inundation_poly_2025_v3' should match a query about
    'flood risk' when the embedding function maps both to nearby vectors.

    This proves semantic search works even with zero keyword overlap.
    """
    store = LayerVectorStore()
    layers = [
        _make_layer("env:inundation_poly_2025_v3", "Überschwemmungspolygone 2025", "Hydrologische Risikogebiete"),
        _make_layer("city:kindergarten", "Kindergärten", "Kinderbetreuungseinrichtungen"),
        _make_layer("city:parking", "Parkplätze", "Öffentliche Stellplätze"),
    ]
    catalog_rows = [
        _make_catalog_row(
            "env:inundation_poly_2025_v3",
            "Überschwemmungspolygone 2025",
            "Inundation Polygons 2025",
            "Hydrologische Risikogebiete",
            "Hydrological risk zones showing flood inundation areas",
        ),
        _make_catalog_row(
            "city:kindergarten",
            "Kindergärten",
            "Kindergartens",
            "Kinderbetreuungseinrichtungen",
            "Childcare facilities",
        ),
        _make_catalog_row(
            "city:parking",
            "Parkplätze",
            "Parking Lots",
            "Öffentliche Stellplätze",
            "Public parking spaces",
        ),
    ]

    # Controlled embeddings: the inundation layer and the "flood risk" query
    # get nearly identical vectors (cosine distance ~0), while others are distant.
    embedding_map: dict[str, list[float]] = {}

    async def controlled_embed(texts: list[str], **kwargs) -> list[list[float]]:
        result = []
        for text in texts:
            lower = text.lower()
            if "inundation" in lower or "flood" in lower or "überschwemmung" in lower:
                # Flood-related: same direction
                vec = [0.9, 0.1, 0.0, 0.0] + [0.0] * 60
            elif "kindergarten" in lower or "childcare" in lower:
                vec = [0.0, 0.0, 0.9, 0.1] + [0.0] * 60
            elif "parking" in lower or "stellplätze" in lower or "parkplätze" in lower:
                vec = [0.0, 0.0, 0.1, 0.9] + [0.0] * 60
            else:
                vec = [0.1] * 64
            result.append(vec)
        return result

    asyncio.run(store.index_layers(layers, catalog_rows, controlled_embed))

    # Query embedding: "flood risk areas" → same vector direction as inundation layer
    query_vec = [0.9, 0.1, 0.0, 0.0] + [0.0] * 60
    results = store.search(query_vec, top_k=3)

    # inundation layer should be the top result despite no keyword "flood" in layer name
    assert results[0]["layer_name"] == "env:inundation_poly_2025_v3"
    assert results[0]["score"] > 0.9  # high cosine similarity


def test_vector_store_skips_reindex_when_content_unchanged() -> None:
    """index_layers with identical data a second time must skip the embed call."""
    store = LayerVectorStore()
    layers = [
        _make_layer("city:hospitals", "Krankenhäuser", "Gesundheitsversorgung"),
        _make_layer("city:schools", "Schulen", "Bildungseinrichtungen"),
    ]
    catalog_rows = [
        _make_catalog_row("city:hospitals", "Krankenhäuser", "Hospitals", "Gesundheitsversorgung", "Healthcare"),
        _make_catalog_row("city:schools", "Schulen", "Schools", "Bildungseinrichtungen", "Education"),
    ]

    call_count = 0

    async def counting_embed(texts: list[str], **kwargs) -> list[list[float]]:
        nonlocal call_count
        call_count += 1
        return [[0.0] * 64 for _ in texts]

    asyncio.run(store.index_layers(layers, catalog_rows, counting_embed))
    assert call_count == 1
    assert store.layer_count() == 2

    # Second call with same data — embed_fn must NOT be called again
    asyncio.run(store.index_layers(layers, catalog_rows, counting_embed))
    assert call_count == 1  # still 1
    assert store.layer_count() == 2


def test_vector_store_reindexes_when_content_changes() -> None:
    """index_layers must re-embed when the catalog content changes."""
    store = LayerVectorStore()
    layers_v1 = [_make_layer("city:hospitals", "Krankenhäuser", "Gesundheitsversorgung")]
    rows_v1 = [_make_catalog_row("city:hospitals", "Krankenhäuser", "Hospitals", "Gesundheitsversorgung", "Healthcare")]

    layers_v2 = [
        _make_layer("city:hospitals", "Krankenhäuser", "Gesundheitsversorgung"),
        _make_layer("city:schools", "Schulen", "Bildungseinrichtungen"),
    ]
    rows_v2 = [
        _make_catalog_row("city:hospitals", "Krankenhäuser", "Hospitals", "Gesundheitsversorgung", "Healthcare"),
        _make_catalog_row("city:schools", "Schulen", "Schools", "Bildungseinrichtungen", "Education"),
    ]

    call_count = 0

    async def counting_embed(texts: list[str], **kwargs) -> list[list[float]]:
        nonlocal call_count
        call_count += 1
        return [[0.0] * 64 for _ in texts]

    asyncio.run(store.index_layers(layers_v1, rows_v1, counting_embed))
    assert call_count == 1
    assert store.layer_count() == 1

    # Content changed — must re-embed
    asyncio.run(store.index_layers(layers_v2, rows_v2, counting_embed))
    assert call_count == 2
    assert store.layer_count() == 2


def test_vector_store_reindexes_after_clear() -> None:
    """After clear(), the same content must trigger a fresh embed call."""
    store = LayerVectorStore()
    layers = [_make_layer("city:roads", "Strassen", "Strassennetz")]
    rows = [_make_catalog_row("city:roads", "Strassen", "Roads", "Strassennetz", "Road network")]

    call_count = 0

    async def counting_embed(texts: list[str], **kwargs) -> list[list[float]]:
        nonlocal call_count
        call_count += 1
        return [[0.0] * 64 for _ in texts]

    asyncio.run(store.index_layers(layers, rows, counting_embed))
    assert call_count == 1

    store.clear()

    # Same content but index was cleared — must re-embed
    asyncio.run(store.index_layers(layers, rows, counting_embed))
    assert call_count == 2
