"""Graph node tests.

Migration note: analyzed intent now uses non-backward-compatible id-bound
spatial_targets and spatial_predicates only.
"""

import asyncio

import pytest
from pydantic import ValidationError

from app.core.schemas import AnalyzedIntent, ECQLGeneration, SpatialPredicateBindingDef, SpatialTargetDef
from app.graph.nodes import (
    _normalize_query_text,
    _score_layer_against_query,
    ecql_generator_node,
    geocoder_context_node,
    layer_discoverer_node,
    schema_extractor_node,
    synthesizer_node,
    unified_router_analyzer_node,
    wfs_discovery_node,
    wfs_executor_node,
)
from app.graph.state import build_initial_state


def test_build_initial_state_sets_defaults() -> None:
    state = build_initial_state("find parks in berlin")

    assert state["user_query"] == "find parks in berlin"
    assert state["intent"] == "irrelevant"
    assert state["final_response"] is None
    assert state["spatial_targets"] is None
    assert state["spatial_predicates"] is None
    assert state["layer_subject"] is None
    assert state["attribute_hints"] == []
    assert state["spatial_contexts"] == []
    assert state["unresolved_target_ids"] == []
    assert state["retry_count"] == 0
    assert state["validation_error"] is None
    assert state["aggregate_usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "request_count": 0,
    }


def test_unified_router_analyzer_node_returns_spatial_entities(monkeypatch) -> None:
    parsed_intent = AnalyzedIntent(
        intent="spatial_query",
        general_response=None,
        layer_subject="schools",
        attribute_hints=["capacity > 100"],
        spatial_targets=[
            {
                "id": "r1",
                "kind": "spatial_reference",
                "value": "Bonn",
                "role": "primary_area",
                "required": True,
            }
        ],
        spatial_predicates=[
            {
                "id": "p1",
                "predicate": "DWITHIN",
                "target_ids": ["r1"],
                "distance": 5,
                "units": "kilometers",
                "join_with_next": "AND",
                "required": True,
            }
        ],
    )

    async def fake_ainvoke_llm(
        *,
        messages,
        output_schema=None,
        response_format=None,
        agent_state=None,
        model_name=None,
        node_name=None,
    ):
        assert "geospatial AI assistant" in messages[0]["content"]
        assert output_schema is AnalyzedIntent
        assert response_format is None
        return parsed_intent

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    updates = asyncio.run(unified_router_analyzer_node({"user_query": "show schools near bonn"}))

    assert updates == {
        "intent": "spatial_query",
        "final_response": None,
        "spatial_targets": [
            {
                "id": "r1",
                "kind": "spatial_reference",
                "value": "Bonn",
                "role": "primary_area",
                "required": True,
            }
        ],
        "spatial_predicates": [
            {
                "id": "p1",
                "predicate": "DWITHIN",
                "target_ids": ["r1"],
                "distance": 5.0,
                "units": "kilometers",
                "join_with_next": "AND",
                "required": True,
            }
        ],
        "layer_subject": "schools",
        "attribute_hints": ["capacity > 100"],
    }


def test_analyzed_intent_accepts_spatial_targets_and_predicates() -> None:
    parsed = AnalyzedIntent.model_validate(
        {
            "intent": "spatial_query",
            "spatial_targets": [
                {
                    "id": "g1",
                    "kind": "explicit_geometry",
                    "value": "POINT(7.1 50.7)",
                    "role": "primary_area",
                    "required": True,
                },
                {
                    "id": "r1",
                    "kind": "spatial_reference",
                    "value": "Bonn",
                    "role": "secondary_area",
                    "required": True,
                },
            ],
            "spatial_predicates": [
                {
                    "id": "p1",
                    "predicate": "INTERSECTS",
                    "target_ids": ["g1"],
                    "join_with_next": "AND",
                    "required": True,
                },
                {
                    "id": "p2",
                    "predicate": "DWITHIN",
                    "target_ids": ["g1", "r1"],
                    "distance": 5,
                    "units": "kilometers",
                    "join_with_next": "AND",
                    "required": True,
                },
            ],
        }
    )

    # RED expectation for new schema fields: they must be materialized, not ignored.
    assert parsed.spatial_targets is not None
    assert [target.id for target in parsed.spatial_targets] == ["g1", "r1"]
    assert parsed.spatial_predicates is not None
    assert [predicate.id for predicate in parsed.spatial_predicates] == ["p1", "p2"]


def test_analyzed_intent_rejects_legacy_fields() -> None:
    try:
        AnalyzedIntent.model_validate(
            {
                "intent": "spatial_query",
                "layer_subject": "hospitals",
                "spatial_reference": "Bonn",
            }
        )
    except Exception:
        pass
    else:
        raise AssertionError("Legacy fields must be rejected in non-backward-compatible API")


# ---------------------------------------------------------------------------
# Slice 1 — Schema cross-field validators
# ---------------------------------------------------------------------------

def test_analyzed_intent_rejects_dwithin_without_distance() -> None:
    with pytest.raises(ValidationError):
        SpatialPredicateBindingDef(
            id="p1",
            predicate="DWITHIN",
            target_ids=["r1"],
            distance=None,
            units="meters",
        )


def test_analyzed_intent_rejects_dwithin_without_units() -> None:
    with pytest.raises(ValidationError):
        SpatialPredicateBindingDef(
            id="p1",
            predicate="DWITHIN",
            target_ids=["r1"],
            distance=100.0,
            units=None,
        )


def test_analyzed_intent_rejects_dwithin_with_zero_distance() -> None:
    with pytest.raises(ValidationError):
        SpatialPredicateBindingDef(
            id="p1",
            predicate="DWITHIN",
            target_ids=["r1"],
            distance=0.0,
            units="meters",
        )


def test_analyzed_intent_rejects_predicate_with_empty_target_ids() -> None:
    with pytest.raises(ValidationError):
        SpatialPredicateBindingDef(
            id="p1",
            predicate="INTERSECTS",
            target_ids=[],
        )


def test_analyzed_intent_rejects_blank_target_id() -> None:
    with pytest.raises(ValidationError):
        SpatialTargetDef(
            id="   ",
            kind="spatial_reference",
            value="Berlin",
        )


def test_analyzed_intent_rejects_blank_predicate_id() -> None:
    with pytest.raises(ValidationError):
        SpatialPredicateBindingDef(
            id="",
            predicate="INTERSECTS",
            target_ids=["r1"],
        )


def test_unified_router_analyzer_node_returns_irrelevant_response(monkeypatch) -> None:
    parsed_intent = AnalyzedIntent(
        intent="irrelevant",
        general_response="Hello! How can I help with geospatial analysis today?",
        layer_subject=None,
        attribute_hints=None,
        spatial_targets=[],
        spatial_predicates=[],
    )

    async def fake_ainvoke_llm(
        *,
        messages,
        output_schema=None,
        response_format=None,
        agent_state=None,
        model_name=None,
        node_name=None,
    ):
        assert output_schema is AnalyzedIntent
        assert response_format is None
        return parsed_intent

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    updates = asyncio.run(unified_router_analyzer_node({"user_query": "hello"}))

    assert updates == {
        "intent": "irrelevant",
        "final_response": {
            "summary": "Hello! How can I help with geospatial analysis today?"
        },
        "spatial_targets": [],
        "spatial_predicates": [],
        "layer_subject": None,
        "attribute_hints": [],
    }


def test_geocoder_context_node_uses_deterministic_geocoder(monkeypatch) -> None:
    class _Geocoder:
        async def forward_fulltext(self, query: str, max_results: int, epsg: int):
            assert query == "Berlin"
            assert max_results == 1
            assert epsg == 4326
            return {"Result": [{"Coordinate": "52.5200,13.4050", "Name": "Berlin"}]}

        async def suggest(self, query: str, max_results: int = 1):
            return {"SuggestResult": [{"id": "1", "locationType": "City", "label": query, "score": 1.0}]}

    monkeypatch.setattr("app.graph.nodes.GeocoderClient", lambda **kwargs: _Geocoder())

    updates = asyncio.run(
        geocoder_context_node(
            {
                "spatial_targets": [
                    {
                        "id": "r1",
                        "kind": "spatial_reference",
                        "value": "Berlin",
                        "required": True,
                    }
                ],
                "spatial_predicates": [
                    {
                        "id": "p1",
                        "predicate": "INTERSECTS",
                        "target_ids": ["r1"],
                        "join_with_next": "AND",
                        "required": True,
                    }
                ],
            }
        )
    )

    context = updates["spatial_contexts"][0]
    assert isinstance(context["bbox"], list)
    assert len(context["bbox"]) == 4
    assert context["crs"] == "EPSG:4326"
    assert context["geometry_type"] == "Polygon"


def test_geocoder_context_node_resolves_id_bound_targets_and_filters_optional_unresolved(monkeypatch) -> None:
    class _Geocoder:
        async def forward_fulltext(self, query: str, max_results: int, epsg: int):
            if query == "Berlin":
                return {"Result": [{"Coordinate": "52.5200,13.4050", "Name": "Berlin"}]}
            return {"Result": []}

        async def suggest(self, query: str, max_results: int = 1):
            if query == "Berlin":
                return {"SuggestResult": [{"id": "1", "locationType": "City", "label": query, "score": 1.0}]}
            return {"SuggestResult": []}

    monkeypatch.setattr("app.graph.nodes.GeocoderClient", lambda **kwargs: _Geocoder())

    updates = asyncio.run(
        geocoder_context_node(
            {
                "spatial_targets": [
                    {"id": "r1", "kind": "spatial_reference", "value": "Berlin", "required": True},
                    {"id": "r2", "kind": "spatial_reference", "value": "Nowhere", "required": False},
                ],
                "spatial_predicates": [
                    {
                        "id": "p1",
                        "predicate": "INTERSECTS",
                        "target_ids": ["r1"],
                        "join_with_next": "AND",
                        "required": True,
                    },
                    {
                        "id": "p2",
                        "predicate": "DWITHIN",
                        "target_ids": ["r2"],
                        "distance": 1,
                        "units": "kilometers",
                        "join_with_next": "AND",
                        "required": False,
                    },
                ],
            }
        )
    )

    assert updates["validation_error"] is None
    assert len(updates["spatial_contexts"]) == 1
    assert updates["spatial_contexts"][0]["target_id"] == "r1"
    assert [p["id"] for p in updates["spatial_predicates"]] == ["p1"]


def test_geocoder_context_node_returns_validation_error_for_required_unresolved_target(monkeypatch) -> None:
    class _Geocoder:
        async def forward_fulltext(self, query: str, max_results: int, epsg: int):
            return {"Result": []}

        async def suggest(self, query: str, max_results: int = 1):
            return {"SuggestResult": []}

    monkeypatch.setattr("app.graph.nodes.GeocoderClient", lambda **kwargs: _Geocoder())

    updates = asyncio.run(
        geocoder_context_node(
            {
                "spatial_targets": [
                    {"id": "r1", "kind": "spatial_reference", "value": "Unknown Place", "required": True},
                ],
                "spatial_predicates": [
                    {
                        "id": "p1",
                        "predicate": "INTERSECTS",
                        "target_ids": ["r1"],
                        "join_with_next": "AND",
                        "required": True,
                    }
                ],
            }
        )
    )

    assert updates["validation_error"] is not None
    assert updates["validation_error"].startswith("location_unresolved:")
    assert "Unknown Place" in updates["validation_error"]


def test_layer_discoverer_node_selects_layer(monkeypatch) -> None:
    class _Layer:
        layer_name = "topp:states"
        confidence = "high"

    async def fake_ainvoke_llm(
        *,
        messages,
        response_format,
        agent_state=None,
        model_name=None,
        enable_prompt_cache=None,
        node_name=None,
    ):
        assert "Layer catalog markdown" in messages[1]["content"]
        assert response_format.__name__ == "LayerSelection"
        return _Layer()

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    updates = asyncio.run(layer_discoverer_node(
        {
            "user_query": "show states",
            "layer_catalog_rows": [
                {"name": "topp:states", "de_title": "Grenzen", "en_title": "States",
                 "de_abstract": "Staatsgrenzen", "en_abstract": "State boundaries", "aliases": ["states"]},
                {"name": "topp:roads", "de_title": "Strassen", "en_title": "Roads",
                 "de_abstract": "Strassennetz", "en_abstract": "Road network", "aliases": ["roads"]},
            ],
            "available_layers": [
                {"name": "topp:states", "title": "States", "abstract": "State boundaries"},
                {"name": "topp:roads", "title": "Roads", "abstract": "Road network"},
            ],
        }
    ))

    assert updates["selected_layer"] == "topp:states"
    assert updates["validation_error"] is None


def test_layer_discoverer_node_uses_layer_subject_primary_single_match(monkeypatch) -> None:
    class _Layer:
        layer_name = "city:hospitals"
        confidence = "high"

    async def fake_ainvoke_llm(
        *,
        messages,
        response_format,
        agent_state=None,
        model_name=None,
        enable_prompt_cache=None,
        node_name=None,
    ):
        assert "Layer subject:\nhospitals" in messages[1]["content"]
        return _Layer()

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    updates = asyncio.run(layer_discoverer_node(
        {
            "user_query": "find hospitals near bonn",
            "layer_subject": "hospitals",
            "layer_catalog_rows": [
                {"name": "city:hospitals", "de_title": "Krankenhäuser", "en_title": "Hospitals",
                 "de_abstract": "Gesundheitseinrichtungen", "en_abstract": "Healthcare facilities",
                 "aliases": ["hospitals"]},
                {"name": "city:roads", "de_title": "Strassen", "en_title": "Roads",
                 "de_abstract": "Strassennetz", "en_abstract": "Road network", "aliases": ["roads"]},
            ],
            "available_layers": [
                {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare facilities"},
                {"name": "city:roads", "title": "Roads", "abstract": "Road network"},
            ],
        }
    ))

    assert updates["selected_layer"] == "city:hospitals"
    assert updates["validation_error"] is None


def test_layer_discoverer_node_uses_filtered_candidates_for_llm_tiebreak(monkeypatch) -> None:
    class _Layer:
        layer_name = "city:hospitals_general"
        confidence = "high"

    async def fake_ainvoke_llm(
        *,
        messages,
        response_format,
        agent_state=None,
        model_name=None,
        enable_prompt_cache=None,
        node_name=None,
    ):
        prompt = messages[1]["content"]
        assert "city:hospitals_general" in prompt
        assert "city:hospitals_specialized" in prompt
        assert "city:roads" in prompt
        assert response_format.__name__ == "LayerSelection"
        return _Layer()

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    updates = asyncio.run(layer_discoverer_node(
        {
            "user_query": "find hospitals",
            "layer_subject": "hospitals",
            "layer_catalog_rows": [
                {"name": "city:hospitals_general", "de_title": "Allgemeine Krankenhäuser",
                 "en_title": "Hospitals", "de_abstract": "", "en_abstract": "All hospitals",
                 "aliases": ["hospitals"]},
                {"name": "city:hospitals_specialized", "de_title": "Spezialkliniken",
                 "en_title": "Specialized Hospitals", "de_abstract": "", "en_abstract": "Specialized care",
                 "aliases": ["specialized"]},
                {"name": "city:roads", "de_title": "Strassen", "en_title": "Roads",
                 "de_abstract": "Strassennetz", "en_abstract": "Road network", "aliases": ["roads"]},
            ],
            "available_layers": [
                {
                    "name": "city:hospitals_general",
                    "title": "General Hospitals",
                    "abstract": "All hospitals",
                },
                {
                    "name": "city:hospitals_specialized",
                    "title": "Specialized Hospitals",
                    "abstract": "Specialized care",
                },
                {"name": "city:roads", "title": "Roads", "abstract": "Road network"},
            ],
        }
    ))

    assert updates["selected_layer"] == "city:hospitals_general"
    assert updates["validation_error"] is None


def test_layer_discoverer_node_falls_back_to_full_list_when_subject_has_no_match(monkeypatch) -> None:
    class _Layer:
        layer_name = "city:roads"
        confidence = "high"

    async def fake_ainvoke_llm(
        *,
        messages,
        response_format,
        agent_state=None,
        model_name=None,
        enable_prompt_cache=None,
        node_name=None,
    ):
        prompt = messages[1]["content"]
        assert "Layer subject:\nschools" in prompt
        assert "city:hospitals" in prompt
        assert "city:roads" in prompt
        assert response_format.__name__ == "LayerSelection"
        return _Layer()

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    updates = asyncio.run(layer_discoverer_node(
        {
            "user_query": "find schools",
            "layer_subject": "schools",
            "layer_catalog_rows": [
                {"name": "city:hospitals", "de_title": "Krankenhäuser", "en_title": "Hospitals",
                 "de_abstract": "", "en_abstract": "Healthcare facilities", "aliases": ["hospitals"]},
                {"name": "city:roads", "de_title": "Strassen", "en_title": "Roads",
                 "de_abstract": "Strassennetz", "en_abstract": "Road network", "aliases": ["roads"]},
            ],
            "available_layers": [
                {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare facilities"},
                {"name": "city:roads", "title": "Roads", "abstract": "Road network"},
            ],
        }
    ))

    assert updates["selected_layer"] == "city:roads"
    assert updates["validation_error"] is None


def test_layer_discoverer_node_gracefully_falls_back_on_low_confidence(monkeypatch) -> None:
    class _Layer:
        layer_name = "city:roads"
        confidence = "low"

    async def fake_ainvoke_llm(
        *,
        messages,
        response_format,
        agent_state=None,
        enable_prompt_cache=None,
        node_name=None,
    ):
        assert "Layer catalog markdown" in messages[1]["content"]
        return _Layer()

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    updates = asyncio.run(layer_discoverer_node(
        {
            "user_query": "find hospitals",
            "layer_subject": "hospitals",
            "layer_catalog_rows": [
                {"name": "city:roads", "de_title": "Strassen", "en_title": "Roads",
                 "de_abstract": "Strassennetz", "en_abstract": "Road network", "aliases": ["roads"]},
                {"name": "city:lakes", "de_title": "Seen", "en_title": "Lakes",
                 "de_abstract": "Gewässer", "en_abstract": "Water bodies", "aliases": ["lakes"]},
            ],
            "available_layers": [
                {"name": "city:roads", "title": "Roads", "abstract": "Road network"},
                {"name": "city:lakes", "title": "Lakes", "abstract": "Water bodies"},
            ],
        }
    ))

    assert updates["selected_layer"] == ""
    assert updates["validation_error"] is not None
    assert "Low-confidence layer retrieval" in updates["validation_error"]


def test_schema_extractor_node_calls_get_layer_schema(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.graph.nodes.get_settings",
        lambda: type(
            "S",
            (),
            {
                "geoserver_wfs_url": "https://wfs",
                "geoserver_wfs_username": "demo-user",
                "geoserver_wfs_password": "demo-pass",
            },
        )(),
    )

    async def fake_get_layer_schema(*, wfs_url: str, type_name: str, username: str, password: str, http_client=None):
        assert wfs_url == "https://wfs"
        assert type_name == "topp:states"
        assert username == "demo-user"
        assert password == "demo-pass"
        return {"STATE_NAME": "xsd:string"}, "the_geom"

    monkeypatch.setattr("app.graph.nodes.get_layer_schema", fake_get_layer_schema)

    updates = asyncio.run(schema_extractor_node({"selected_layer": "topp:states"}))

    assert updates["layer_schema"]["STATE_NAME"] == "xsd:string"
    assert updates["geometry_column"] == "the_geom"
    assert updates["validation_error"] is None


def test_ecql_generator_node_increments_retry_and_returns_ecql(monkeypatch) -> None:
    async def fake_ainvoke_llm(*, messages, response_format, agent_state=None, node_name=None):
        prompt = messages[1]["content"]
        assert "Previous validation error" in messages[1]["content"]
        assert "Attribute hints" in prompt
        assert "capacity > 100" in prompt
        assert response_format is ECQLGeneration
        return ECQLGeneration(reasoning="Use geometry and population", ecql_string="PERSONS > 1000")

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    updates = asyncio.run(ecql_generator_node(
        {
            "user_query": "states with high population",
            "layer_schema": {"PERSONS": "xsd:int"},
            "geometry_column": "the_geom",
            "spatial_contexts": [
                {
                    "target_id": "g1",
                    "crs": "EPSG:4326",
                    "bbox": [-10, -10, 10, 10],
                    "geometry_wkt": "POLYGON ((-10 -10, 10 -10, 10 10, -10 10, -10 -10))",
                    "geometry_type": "Polygon",
                },
            ],
            "spatial_predicates": [
                {
                    "id": "p1",
                    "predicate": "INTERSECTS",
                    "target_ids": ["g1"],
                    "join_with_next": "AND",
                    "required": True,
                }
            ],
            "attribute_hints": ["capacity > 100"],
            "validation_error": "Unknown attribute",
            "retry_count": 1,
        }
    ))

    assert "INTERSECTS(the_geom" in updates["generated_ecql"]
    assert "PERSONS > 1000" in updates["generated_ecql"]
    assert updates["retry_count"] == 2


def test_ecql_generator_node_builds_multi_spatial_ecql(monkeypatch) -> None:
    async def fake_ainvoke_llm(*, messages, response_format, agent_state=None, node_name=None):
        assert response_format is ECQLGeneration
        return ECQLGeneration(reasoning="Use attributes only", ecql_string="PERSONS > 1000")

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    updates = asyncio.run(ecql_generator_node(
        {
            "user_query": "states near bbox and point",
            "layer_schema": {"PERSONS": "xsd:int"},
            "geometry_column": "the_geom",
            "spatial_contexts": [
                {
                    "target_id": "g1",
                    "crs": "EPSG:4326",
                    "bbox": [7.0, 50.0, 7.5, 50.5],
                    "geometry_wkt": (
                        "POLYGON ((7.0 50.0, 7.5 50.0, 7.5 50.5, 7.0 50.5, 7.0 50.0))"
                    ),
                    "geometry_type": "Polygon",
                },
                {
                    "target_id": "g2",
                    "crs": "EPSG:4326",
                    "bbox": [7.19, 50.19, 7.21, 50.21],
                    "geometry_wkt": "POINT (7.2 50.2)",
                    "geometry_type": "Point",
                },
            ],
            "spatial_predicates": [
                {
                    "id": "p1",
                    "predicate": "INTERSECTS",
                    "target_ids": ["g1"],
                    "join_with_next": "AND",
                    "required": True,
                },
                {
                    "id": "p2",
                    "predicate": "DWITHIN",
                    "target_ids": ["g2"],
                    "distance": 100,
                    "units": "meters",
                    "join_with_next": "AND",
                    "required": True,
                },
            ],
            "attribute_hints": [],
            "validation_error": None,
            "retry_count": 0,
        }
    ))

    assert "INTERSECTS(the_geom, SRID=4326;POLYGON" in updates["generated_ecql"]
    assert "DWITHIN(the_geom, SRID=4326;POINT (7.2 50.2), 100, meters)" in updates["generated_ecql"]
    assert "PERSONS > 1000" in updates["generated_ecql"]


def test_ecql_generator_node_avoids_duplicate_bbox_for_duplicated_reference_context(monkeypatch) -> None:
    async def fake_ainvoke_llm(*, messages, response_format, agent_state=None, node_name=None):
        assert response_format is ECQLGeneration
        return ECQLGeneration(reasoning="Use attributes only", ecql_string="manager = 'NBG-Maps'")

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    hamburg_context = {
        "target_id": "r1",
        "source": "reference",
        "crs": "EPSG:4326",
        "bbox": [9.621463608102793, 53.303866335746676, 10.519778892222314, 53.837305712419166],
        "geometry_wkt": (
            "POLYGON ((10.519778892222314 53.303866335746676, "
            "10.519778892222314 53.837305712419166, "
            "9.621463608102793 53.837305712419166, "
            "9.621463608102793 53.303866335746676, "
            "10.519778892222314 53.303866335746676))"
        ),
        "geometry_type": "Polygon",
    }

    updates = asyncio.run(ecql_generator_node(
        {
            "user_query": "Show me all archeive new development areas managed by NBG-Maps in hamburg",
            "layer_schema": {"manager": "xsd:string"},
            "geometry_column": "wkb_geometry",
            "spatial_contexts": [hamburg_context],
            "spatial_predicates": [
                {
                    "id": "p1",
                    "predicate": "INTERSECTS",
                    "target_ids": ["r1"],
                    "join_with_next": "AND",
                    "required": True,
                }
            ],
            "attribute_hints": ["manager = 'NBG-Maps'"],
            "validation_error": None,
            "retry_count": 0,
        }
    ))

    generated_ecql = updates["generated_ecql"]
    assert "BBOX(wkb_geometry" not in generated_ecql
    assert "INTERSECTS(wkb_geometry, SRID=4326;POLYGON" in generated_ecql
    assert "manager = 'NBG-Maps'" in generated_ecql


def test_synthesizer_node_returns_llm_summary(monkeypatch) -> None:
    async def fake_ainvoke_llm(*, messages, response_format=None, agent_state=None, model_name=None, node_name=None):
        assert response_format is None
        assert "Feature count" in messages[1]["content"]
        return "Found 2 matching features in the requested area."

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    updates = asyncio.run(synthesizer_node(
        {
            "user_query": "show matching states",
            "wfs_result": {
                "type": "FeatureCollection",
                "features": [
                    {"properties": {"STATE_NAME": "A"}},
                    {"properties": {"STATE_NAME": "B"}},
                ],
            },
        }
    ))

    assert updates == {
        "final_response": {
            "summary": "Found 2 matching features in the requested area."
        },
    }


def test_wfs_discovery_node_populates_available_layers(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.graph.nodes.get_settings",
        lambda: type(
            "S",
            (),
            {
                "geoserver_wfs_url": "https://wfs",
                "geoserver_wfs_username": "demo-user",
                "geoserver_wfs_password": "demo-pass",
            },
        )(),
    )

    async def fake_discover_layers(*, wfs_url: str, username: str, password: str, http_client=None):
        assert wfs_url == "https://wfs"
        assert username == "demo-user"
        assert password == "demo-pass"
        return [{"name": "topp:states", "title": "States", "abstract": "Boundaries"}]

    async def fake_ensure_markdown_layer_catalog(*, layers, catalog_path, stale_after_hours):
        return "# GeoServer Layer Catalog\n\n- **Layer ID:** `topp:states`\n", [
            {"name": "topp:states", "de_title": "Grenzen", "en_title": "States",
             "de_abstract": "Staatsgrenzen", "en_abstract": "State boundaries", "aliases": ["states"]},
        ]

    monkeypatch.setattr("app.graph.nodes.discover_layers", fake_discover_layers)
    monkeypatch.setattr("app.graph.nodes.ensure_markdown_layer_catalog", fake_ensure_markdown_layer_catalog)

    updates = asyncio.run(wfs_discovery_node({"user_query": "find states"}))

    assert updates["available_layers"][0]["name"] == "topp:states"
    assert updates["validation_error"] is None
    assert "layer_catalog_rows" in updates
    assert isinstance(updates["layer_catalog_rows"], list)
    assert updates["layer_catalog_rows"][0]["name"] == "topp:states"
    assert updates["layer_catalog_rows"][0]["en_title"] == "States"


def test_wfs_executor_node_returns_wfs_result(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.graph.nodes.get_settings",
        lambda: type(
            "S",
            (),
            {
                "geoserver_wfs_url": "https://wfs",
                "geoserver_wfs_username": "demo-user",
                "geoserver_wfs_password": "demo-pass",
                "geoserver_wfs_srs_name": "EPSG:3857",
            },
        )(),
    )

    async def fake_execute_wfs_query(
        *,
        wfs_url: str,
        type_name: str,
        cql_filter: str,
        count: int,
        srs_name: str,
        username: str,
        password: str,
        http_client=None,
    ):
        assert wfs_url == "https://wfs"
        assert type_name == "topp:states"
        assert cql_filter == "PERSONS > 1000"
        assert count == 1000
        assert srs_name == "EPSG:3857"
        assert username == "demo-user"
        assert password == "demo-pass"
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr("app.graph.nodes.execute_wfs_query", fake_execute_wfs_query)

    updates = asyncio.run(wfs_executor_node(
        {
            "selected_layer": "topp:states",
            "generated_ecql": "PERSONS > 1000",
        }
    ))

    assert updates["validation_error"] is None
    assert updates["wfs_result"]["type"] == "FeatureCollection"


def test_geocoder_context_node_no_spatial_targets_returns_empty_context() -> None:
    updates = asyncio.run(geocoder_context_node({"user_query": "show schools"}))

    assert updates == {"spatial_contexts": []}


# ---------------------------------------------------------------------------
# Slice 2 — pyproj Transformer caching
# ---------------------------------------------------------------------------

def test_build_bbox_for_point_does_not_reconstruct_transformer_on_second_call() -> None:
    from app.graph.nodes import _get_transformer

    _get_transformer.cache_clear()

    _get_transformer(4326, 3857)
    _get_transformer(3857, 4326)
    _get_transformer(4326, 3857)  # second call — should be a cache hit

    info = _get_transformer.cache_info()
    assert info.hits >= 1, f"Expected at least one cache hit, got {info}"


# ---------------------------------------------------------------------------
# Slice 3 — HTTP clients out of AgentState
# ---------------------------------------------------------------------------

def test_agent_state_does_not_contain_http_client_fields() -> None:
    from app.graph.state import AgentState
    import typing
    hints = typing.get_type_hints(AgentState)
    assert "geocoder_http_client" not in hints, "geocoder_http_client must not be in AgentState"
    assert "wfs_http_client" not in hints, "wfs_http_client must not be in AgentState"


def test_build_initial_state_accepts_no_http_client_args() -> None:
    import inspect
    from app.graph.state import build_initial_state
    sig = inspect.signature(build_initial_state)
    assert "geocoder_http_client" not in sig.parameters
    assert "wfs_http_client" not in sig.parameters
    state = build_initial_state("test query")
    assert "geocoder_http_client" not in state
    assert "wfs_http_client" not in state


# ---------------------------------------------------------------------------
# Slice 6 — _distance_filter_for_target returns largest distance
# ---------------------------------------------------------------------------

def test_distance_filter_for_target_returns_largest_distance_when_multiple_predicates() -> None:
    from app.graph.nodes import _distance_filter_for_target

    predicates = [
        {
            "id": "p1",
            "predicate": "DWITHIN",
            "target_ids": ["r1"],
            "distance": 500.0,
            "units": "meters",
        },
        {
            "id": "p2",
            "predicate": "DWITHIN",
            "target_ids": ["r1"],
            "distance": 5.0,
            "units": "kilometers",  # 5000 m > 500 m
        },
    ]

    result = _distance_filter_for_target("r1", predicates)

    assert result is not None
    assert result["distance"] == 5.0
    assert result["units"] == "kilometers"


# ---------------------------------------------------------------------------
# Phase 2 — Selector Scoring
# ---------------------------------------------------------------------------


def test_score_layer_against_query_returns_high_for_exact_title_match() -> None:
    layer = {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare facilities"}
    score = _score_layer_against_query(layer, "hospitals")
    assert score >= 0.8


def test_score_layer_against_query_returns_low_for_unrelated() -> None:
    layer = {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare facilities"}
    score = _score_layer_against_query(layer, "rivers")
    assert score < 0.4


def test_score_layer_against_query_ranks_matching_layer_higher() -> None:
    hospitals = {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare facilities"}
    roads = {"name": "city:roads", "title": "Roads", "abstract": "Road network"}

    score_hospitals = _score_layer_against_query(hospitals, "hospitals")
    score_roads = _score_layer_against_query(roads, "hospitals")
    assert score_hospitals > score_roads


def test_normalize_query_text_strips_gis_stopwords() -> None:
    normalized = _normalize_query_text("Show me the Layer for Schools")
    assert "layer" not in normalized.lower()
    assert "school" in normalized.lower()


def test_normalize_query_text_lowercases_and_strips_punctuation() -> None:
    normalized = _normalize_query_text("Hospitals!!!")
    assert normalized == "hospitals"


def test_layer_discoverer_node_returns_fallback_without_llm_when_scores_too_low(monkeypatch) -> None:
    """When no layer scores above min_retrieval_score, fallback immediately without LLM."""
    llm_called = False

    async def fake_ainvoke_llm(**kwargs):
        nonlocal llm_called
        llm_called = True
        raise AssertionError("LLM should not be called when scores are too low")

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)
    monkeypatch.setattr(
        "app.graph.nodes.get_settings",
        lambda: type("S", (), {
            "min_retrieval_score": 0.5,
            "max_llm_candidates": 10,
            "current_model": "gpt-4.1",
        })(),
    )

    updates = asyncio.run(layer_discoverer_node(
        {
            "user_query": "find xyzzy quantum flux capacitors",
            "layer_subject": "xyzzy quantum flux capacitors",
            "layer_catalog_rows": [
                {"name": "city:hospitals", "de_title": "Krankenhäuser", "en_title": "Hospitals",
                 "de_abstract": "", "en_abstract": "Healthcare", "aliases": ["hospitals"]},
                {"name": "city:roads", "de_title": "Strassen", "en_title": "Roads",
                 "de_abstract": "Strassennetz", "en_abstract": "Road network", "aliases": ["roads"]},
            ],
            "available_layers": [
                {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare"},
                {"name": "city:roads", "title": "Roads", "abstract": "Road network"},
            ],
        }
    ))

    assert llm_called is False
    assert updates["selected_layer"] == ""
    assert updates["validation_error"] is not None
    assert updates["retrieval_top_score"] is not None
    assert updates["retrieval_top_score"] < 0.5


def test_layer_discoverer_node_populates_scores_on_success(monkeypatch) -> None:
    """When a high-score layer is found, retrieval_top_score and retrieval_score_gap should be populated."""
    class _Layer:
        layer_name = "city:hospitals"
        confidence = "high"
        reasoning = "Exact match"
        score = 0.95

    async def fake_ainvoke_llm(**kwargs):
        return _Layer()

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)
    monkeypatch.setattr(
        "app.graph.nodes.get_settings",
        lambda: type("S", (), {
            "min_retrieval_score": 0.15,
            "max_llm_candidates": 10,
            "current_model": "gpt-4.1",
        })(),
    )

    updates = asyncio.run(layer_discoverer_node(
        {
            "user_query": "find hospitals",
            "layer_subject": "hospitals",
            "layer_catalog_rows": [
                {"name": "city:hospitals", "de_title": "Krankenhäuser", "en_title": "Hospitals",
                 "de_abstract": "", "en_abstract": "Healthcare", "aliases": ["hospitals"]},
                {"name": "city:roads", "de_title": "Strassen", "en_title": "Roads",
                 "de_abstract": "Strassennetz", "en_abstract": "Road network", "aliases": ["roads"]},
            ],
            "available_layers": [
                {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare"},
                {"name": "city:roads", "title": "Roads", "abstract": "Road network"},
            ],
        }
    ))

    assert updates["selected_layer"] == "city:hospitals"
    assert updates["retrieval_top_score"] is not None
    assert updates["retrieval_top_score"] > 0.15
    assert updates["retrieval_score_gap"] is not None


# ---------------------------------------------------------------------------
# Phase 1b — Catalog slicing: only top-N rows reach LLM
# ---------------------------------------------------------------------------


def test_layer_discoverer_node_sends_only_top_n_rows_to_llm(monkeypatch) -> None:
    """Rows for layers NOT in available_layers must not appear in the LLM catalog prompt."""
    captured_prompt: list[str] = []

    class _Layer:
        layer_name = "city:hospitals"
        confidence = "high"

    async def fake_ainvoke_llm(*, messages, response_format, agent_state=None,
                                enable_prompt_cache=None, node_name=None, **kwargs):
        captured_prompt.append(messages[1]["content"])
        return _Layer()

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    updates = asyncio.run(layer_discoverer_node(
        {
            "user_query": "find hospitals",
            "layer_subject": "hospitals",
            # catalog_rows has 4 rows but only 2 layers are "available"
            "layer_catalog_rows": [
                {"name": "city:hospitals", "de_title": "Krankenhäuser", "en_title": "Hospitals",
                 "de_abstract": "", "en_abstract": "Healthcare", "aliases": ["hospitals"]},
                {"name": "city:roads", "de_title": "Strassen", "en_title": "Roads",
                 "de_abstract": "Strassennetz", "en_abstract": "Road network", "aliases": ["roads"]},
                # These two are stale rows that no longer exist in the server
                {"name": "city:parks", "de_title": "Parks", "en_title": "Parks",
                 "de_abstract": "Grünanlagen", "en_abstract": "City parks", "aliases": ["parks"]},
                {"name": "city:lakes", "de_title": "Seen", "en_title": "Lakes",
                 "de_abstract": "Gewässer", "en_abstract": "Water bodies", "aliases": ["lakes"]},
            ],
            "available_layers": [
                {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare"},
                {"name": "city:roads", "title": "Roads", "abstract": "Road network"},
            ],
        }
    ))

    assert updates["selected_layer"] == "city:hospitals"
    assert len(captured_prompt) == 1
    prompt = captured_prompt[0]
    # Only available layers should appear in catalog portion
    assert "city:hospitals" in prompt
    assert "city:roads" in prompt
    # Stale rows absent from available_layers must NOT appear
    assert "city:parks" not in prompt
    assert "city:lakes" not in prompt


def test_layer_discoverer_node_falls_back_to_basic_catalog_when_no_rows_in_state(monkeypatch) -> None:
    """When layer_catalog_rows is absent, render_basic_markdown_catalog(top_layers) is used."""
    captured_prompt: list[str] = []

    class _Layer:
        layer_name = "city:hospitals"
        confidence = "high"

    async def fake_ainvoke_llm(*, messages, response_format, agent_state=None,
                                enable_prompt_cache=None, node_name=None, **kwargs):
        captured_prompt.append(messages[1]["content"])
        return _Layer()

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    updates = asyncio.run(layer_discoverer_node(
        {
            "user_query": "find hospitals",
            "layer_subject": "hospitals",
            # no layer_catalog_rows in state
            "available_layers": [
                {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare"},
                {"name": "city:roads", "title": "Roads", "abstract": "Road network"},
            ],
        }
    ))

    assert updates["selected_layer"] == "city:hospitals"
    assert len(captured_prompt) == 1
    # Basic catalog still includes layer names
    assert "city:hospitals" in captured_prompt[0]


# ---------------------------------------------------------------------------
# Phase 3 — Per-node token tracking
# ---------------------------------------------------------------------------


def test_ainvoke_llm_tracks_per_node_usage(monkeypatch) -> None:
    """When node_name is passed, aggregate_usage.by_node is populated per node."""
    import asyncio as _asyncio
    from app.core.llm import ainvoke_llm
    from app.graph.state import build_initial_state

    class _FakeUsage:
        def get(self, key, default=0):
            return {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}.get(key, default)

    class _FakeResponse:
        usage = _FakeUsage()
        choices = [type("C", (), {"message": type("M", (), {"content": "hello"})()})()]

    async def fake_acompletion(**kwargs):
        return _FakeResponse()

    monkeypatch.setattr("app.core.llm.acompletion", fake_acompletion)
    monkeypatch.setattr(
        "app.core.llm.get_settings",
        lambda: type("S", (), {
            "current_model": "gpt-4.1",
            "llm_base_url": "",
            "llm_api_key": "",
            "openai_api_key": "key",
            "llm_prompt_cache_enabled": False,
        })(),
    )

    state = build_initial_state("test")

    _asyncio.run(ainvoke_llm(
        messages=[{"role": "user", "content": "hello"}],
        agent_state=state,
        node_name="ecql_generator_node",
    ))
    _asyncio.run(ainvoke_llm(
        messages=[{"role": "user", "content": "hello"}],
        agent_state=state,
        node_name="ecql_generator_node",
    ))
    _asyncio.run(ainvoke_llm(
        messages=[{"role": "user", "content": "hello"}],
        agent_state=state,
        node_name="synthesizer_node",
    ))

    agg = state["aggregate_usage"]
    assert agg["prompt_tokens"] == 300
    assert agg["completion_tokens"] == 60
    assert agg["request_count"] == 3

    by_node = agg["by_node"]
    assert by_node["ecql_generator_node"]["prompt_tokens"] == 200
    assert by_node["ecql_generator_node"]["request_count"] == 2
    assert by_node["synthesizer_node"]["prompt_tokens"] == 100
    assert by_node["synthesizer_node"]["request_count"] == 1


def test_ainvoke_llm_skips_by_node_when_no_node_name(monkeypatch) -> None:
    """Without node_name, by_node should not be created."""
    import asyncio as _asyncio
    from app.core.llm import ainvoke_llm
    from app.graph.state import build_initial_state

    class _FakeUsage:
        def get(self, key, default=0):
            return {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}.get(key, default)

    class _FakeResponse:
        usage = _FakeUsage()
        choices = [type("C", (), {"message": type("M", (), {"content": "hi"})()})()]

    async def fake_acompletion(**kwargs):
        return _FakeResponse()

    monkeypatch.setattr("app.core.llm.acompletion", fake_acompletion)
    monkeypatch.setattr(
        "app.core.llm.get_settings",
        lambda: type("S", (), {
            "current_model": "gpt-4.1",
            "llm_base_url": "",
            "llm_api_key": "",
            "openai_api_key": "key",
            "llm_prompt_cache_enabled": False,
        })(),
    )

    state = build_initial_state("test")
    _asyncio.run(ainvoke_llm(messages=[{"role": "user", "content": "hi"}], agent_state=state))

    assert "by_node" not in state["aggregate_usage"]

