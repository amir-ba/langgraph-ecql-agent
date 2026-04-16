import asyncio
from typing import Any

from app.graph.nodes import layer_discoverer_node
from app.tools.vector_store import LayerVectorStore


def _make_layer(name: str, title: str, abstract: str) -> dict[str, str]:
    return {"name": name, "title": title, "abstract": abstract}


def test_layer_discoverer_uses_semantic_search_when_indexed(monkeypatch) -> None:
    """When vector store is indexed, layer_discoverer_node uses semantic search
    instead of fuzzy matching. The top semantic result should be passed to LLM.
    """

    class _FakeLLMResponse:
        layer_name = "env:flood_zones"
        confidence = "high"
        reasoning = "Flood zones match the query"
        score = 0.95

    async def fake_ainvoke_llm(
        *,
        messages,
        response_format,
        agent_state=None,
        model_name=None,
        enable_prompt_cache=None,
        node_name=None,
    ):
        # Verify the top semantic result is in the LLM prompt
        content = messages[1]["content"]
        assert "env:flood_zones" in content
        return _FakeLLMResponse()

    async def fake_get_embeddings(texts, **kwargs):
        return [[0.9, 0.1, 0.0, 0.0] + [0.0] * 60]

    # Create and populate a vector store
    store = LayerVectorStore()
    layers = [
        _make_layer("env:flood_zones", "Hochwasserzonen", "Überschwemmungsgebiete"),
        _make_layer("city:kindergarten", "Kindergärten", "Kinderbetreuung"),
        _make_layer("city:parking", "Parkplätze", "Stellplätze"),
    ]
    catalog_rows = [
        {"name": "env:flood_zones", "de_title": "Hochwasserzonen", "en_title": "Flood Zones",
         "de_abstract": "Überschwemmungsgebiete", "en_abstract": "Flood risk areas", "aliases": []},
        {"name": "city:kindergarten", "de_title": "Kindergärten", "en_title": "Kindergartens",
         "de_abstract": "Kinderbetreuung", "en_abstract": "Childcare", "aliases": []},
        {"name": "city:parking", "de_title": "Parkplätze", "en_title": "Parking",
         "de_abstract": "Stellplätze", "en_abstract": "Parking spaces", "aliases": []},
    ]

    async def controlled_embed(texts, **kwargs):
        result = []
        for text in texts:
            lower = text.lower()
            if "flood" in lower or "hochwasser" in lower or "überschwemmung" in lower:
                result.append([0.9, 0.1, 0.0, 0.0] + [0.0] * 60)
            elif "kindergarten" in lower or "kinderbetreuung" in lower:
                result.append([0.0, 0.0, 0.9, 0.1] + [0.0] * 60)
            else:
                result.append([0.0, 0.0, 0.1, 0.9] + [0.0] * 60)
        return result

    asyncio.run(store.index_layers(layers, catalog_rows, controlled_embed))

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)
    monkeypatch.setattr("app.graph.nodes.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("app.graph.nodes.get_layer_vector_store", lambda: store)
    monkeypatch.setenv("LAYER_DISCOVERY_MODE", "semantic")
    from app.core.settings import get_settings
    get_settings.cache_clear()

    state: dict[str, Any] = {
        "user_query": "show me flood risk areas",
        "layer_subject": "flood zones",
        "available_layers": layers,
        "layer_catalog_rows": catalog_rows,
    }

    result = asyncio.run(layer_discoverer_node(state))

    assert result["selected_layer"] == "env:flood_zones"
    assert result["validation_error"] is None
    assert result["retrieval_mode"] == "semantic"
    assert result["retrieval_top_score"] > 0.9


def test_layer_discoverer_returns_error_when_semantic_index_unavailable(monkeypatch) -> None:
    """If semantic indexing fails and no index is available, node should fail safely."""

    llm_called = False

    async def fake_ainvoke_llm(
        *,
        messages,
        response_format,
        agent_state=None,
        model_name=None,
        enable_prompt_cache=None,
        node_name=None,
    ):
        nonlocal llm_called
        llm_called = True
        raise AssertionError("LLM should not be called when semantic index is unavailable")

    async def failing_embeddings(texts, **kwargs):
        raise ValueError("LLM_BASE_URL must be set for embedding requests")

    store = LayerVectorStore()
    assert not store.is_indexed()

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)
    monkeypatch.setattr("app.graph.nodes.get_embeddings", failing_embeddings)
    monkeypatch.setattr("app.graph.nodes.get_layer_vector_store", lambda: store)
    monkeypatch.setenv("LAYER_DISCOVERY_MODE", "semantic")
    from app.core.settings import get_settings
    get_settings.cache_clear()

    state: dict[str, Any] = {
        "user_query": "show states",
        "layer_subject": "states",
        "available_layers": [
            {"name": "topp:states", "title": "States", "abstract": "State boundaries"},
            {"name": "topp:roads", "title": "Roads", "abstract": "Road network"},
        ],
        "layer_catalog_rows": [],
    }

    result = asyncio.run(layer_discoverer_node(state))

    assert llm_called is False
    assert result["selected_layer"] == ""
    assert result["validation_error"] is not None
    assert "Semantic layer index unavailable" in result["validation_error"]
