import asyncio
from typing import Any

from app.graph.nodes import layer_discoverer_node, wfs_discovery_node


def _make_layer(name: str, title: str, abstract: str) -> dict[str, str]:
    return {"name": name, "title": title, "abstract": abstract}


LAYERS = [
    _make_layer("city:hospitals", "Krankenhäuser", "Gesundheitseinrichtungen der Stadt"),
    _make_layer("city:schools", "Schulen", "Bildungseinrichtungen"),
    _make_layer("city:parking", "Parkplätze", "Stellplätze im Stadtgebiet"),
    _make_layer("env:flood_zones", "Hochwasserzonen", "Überschwemmungsgebiete"),
]

CATALOG_ROWS = [
    {
        "name": "city:hospitals",
        "de_title": "Krankenhäuser",
        "en_title": "Hospitals",
        "de_abstract": "Gesundheitseinrichtungen der Stadt",
        "en_abstract": "City healthcare facilities",
        "aliases": ["hospitals", "krankenhäuser"],
    },
    {
        "name": "city:schools",
        "de_title": "Schulen",
        "en_title": "Schools",
        "de_abstract": "Bildungseinrichtungen",
        "en_abstract": "Educational institutions",
        "aliases": ["schools", "schulen"],
    },
    {
        "name": "city:parking",
        "de_title": "Parkplätze",
        "en_title": "Parking",
        "de_abstract": "Stellplätze im Stadtgebiet",
        "en_abstract": "Parking spaces",
        "aliases": ["parking", "parkplätze"],
    },
    {
        "name": "env:flood_zones",
        "de_title": "Hochwasserzonen",
        "en_title": "Flood Zones",
        "de_abstract": "Überschwemmungsgebiete",
        "en_abstract": "Flood risk areas",
        "aliases": ["flood zones", "hochwasserzonen"],
    },
]


def test_fuzzy_mode_selects_correct_layer(monkeypatch) -> None:
    """In fuzzy mode, layer_discoverer_node ranks layers by string similarity,
    passes top-k to LLM, and returns the selected layer with retrieval_mode='fuzzy'.
    """

    class _FakeLLMResponse:
        layer_name = "city:hospitals"
        confidence = "high"
        reasoning = "Hospitals match the query"
        score = 0.9

    async def fake_ainvoke_llm(
        *,
        messages,
        response_format,
        agent_state=None,
        model_name=None,
        enable_prompt_cache=None,
        node_name=None,
    ):
        # Verify fuzzy top candidates are in the LLM prompt
        content = messages[1]["content"]
        assert "city:hospitals" in content
        return _FakeLLMResponse()

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)
    # Force settings reload
    from app.core.settings import get_settings
    get_settings.cache_clear()

    state: dict[str, Any] = {
        "user_query": "show me hospitals in the city",
        "layer_subject": "hospitals",
        "available_layers": LAYERS,
        "layer_catalog_rows": CATALOG_ROWS,
    }

    result = asyncio.run(layer_discoverer_node(state))

    assert result["selected_layer"] == "city:hospitals"
    assert result["validation_error"] is None
    assert result["retrieval_mode"] == "fuzzy"


def test_fuzzy_mode_scores_catalog_translations_and_aliases(monkeypatch) -> None:
    """Fuzzy scoring should match against catalog bilingual fields and aliases,
    not just the raw WFS title/abstract. An English query 'healthcare' should
    rank 'city:fca_gesundheit' at the top because the catalog has
    en_title='Healthcare Facilities' and alias 'healthcare', even though
    the raw WFS metadata is entirely German.
    """

    llm_received_layers: list[str] = []

    class _FakeLLMResponse:
        layer_name = "city:fca_gesundheit"
        confidence = "high"
        reasoning = "Healthcare match"
        score = 0.9

    async def fake_ainvoke_llm(
        *,
        messages,
        response_format,
        agent_state=None,
        model_name=None,
        enable_prompt_cache=None,
        node_name=None,
    ):
        content = messages[1]["content"]
        llm_received_layers.append(content)
        return _FakeLLMResponse()

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)
    monkeypatch.setenv("MAX_LLM_CANDIDATES", "1")
    from app.core.settings import get_settings
    get_settings.cache_clear()

    # Raw WFS metadata is ENTIRELY German — no English words anywhere
    german_only_layers = [
        _make_layer("city:fca_gesundheit", "Gesundheitseinrichtungen", "Ärzte und Kliniken der Stadt"),
        _make_layer("city:fca_strassen", "Strassennetzwerk", "Hauptverkehrswege"),
        _make_layer("city:fca_bildung", "Bildungseinrichtungen", "Schulen und Universitäten"),
    ]

    catalog_with_translations = [
        {
            "name": "city:fca_gesundheit",
            "de_title": "Gesundheitseinrichtungen",
            "en_title": "Healthcare Facilities",
            "de_abstract": "Ärzte und Kliniken der Stadt",
            "en_abstract": "Doctors and clinics in the city",
            "aliases": ["healthcare", "gesundheit", "kliniken"],
        },
        {
            "name": "city:fca_strassen",
            "de_title": "Strassennetzwerk",
            "en_title": "Road Network",
            "de_abstract": "Hauptverkehrswege",
            "en_abstract": "Main traffic routes",
            "aliases": ["roads", "strassen"],
        },
        {
            "name": "city:fca_bildung",
            "de_title": "Bildungseinrichtungen",
            "en_title": "Educational Institutions",
            "de_abstract": "Schulen und Universitäten",
            "en_abstract": "Schools and universities",
            "aliases": ["education", "schulen", "bildung"],
        },
    ]

    state: dict[str, Any] = {
        "user_query": "show healthcare facilities",
        "layer_subject": "healthcare",
        "available_layers": german_only_layers,
        "layer_catalog_rows": catalog_with_translations,
    }

    result = asyncio.run(layer_discoverer_node(state))

    assert result["selected_layer"] == "city:fca_gesundheit"
    # The LLM must have received fca_gesundheit as top candidate
    assert "city:fca_gesundheit" in llm_received_layers[0]
