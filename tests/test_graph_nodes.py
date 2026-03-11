from app.core.schemas import AnalyzedIntent, ECQLGeneration
from app.graph.nodes import (
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
    assert state["spatial_reference"] is None
    assert state["spatial_relationship"] is None
    assert state["layer_subject"] is None
    assert state["attribute_hints"] == []
    assert state["spatial_context_bbox"] is None
    assert state["retry_count"] == 0
    assert state["validation_error"] is None


def test_unified_router_analyzer_node_returns_spatial_entities(monkeypatch) -> None:
    parsed_intent = AnalyzedIntent(
        intent="spatial_query",
        general_response=None,
        spatial_reference="Bonn",
        spatial_relationship="within 5km",
        layer_subject="schools",
        attribute_hints=["capacity > 100"],
    )

    def fake_invoke_llm(*, messages, output_schema=None, response_format=None):
        assert "geospatial AI assistant" in messages[0]["content"]
        assert output_schema is AnalyzedIntent
        assert response_format is None
        return parsed_intent

    monkeypatch.setattr("app.graph.nodes.invoke_llm", fake_invoke_llm)

    updates = unified_router_analyzer_node({"user_query": "show schools near bonn"})

    assert updates == {
        "intent": "spatial_query",
        "final_response": None,
        "spatial_reference": "Bonn",
        "spatial_relationship": "within 5km",
        "layer_subject": "schools",
        "attribute_hints": ["capacity > 100"],
    }


def test_unified_router_analyzer_node_returns_general_response(monkeypatch) -> None:
    parsed_intent = AnalyzedIntent(
        intent="general_chat",
        general_response="Hello! How can I help with geospatial analysis today?",
        spatial_reference=None,
        spatial_relationship=None,
        layer_subject=None,
        attribute_hints=None,
    )

    def fake_invoke_llm(*, messages, output_schema=None, response_format=None):
        assert output_schema is AnalyzedIntent
        assert response_format is None
        return parsed_intent

    monkeypatch.setattr("app.graph.nodes.invoke_llm", fake_invoke_llm)

    updates = unified_router_analyzer_node({"user_query": "hello"})

    assert updates == {
        "intent": "general_chat",
        "final_response": {
            "summary": "Hello! How can I help with geospatial analysis today?",
            "geojson": None,
        },
        "spatial_reference": None,
        "spatial_relationship": None,
        "layer_subject": None,
        "attribute_hints": [],
    }


def test_geocoder_context_node_uses_deterministic_geocoder(monkeypatch) -> None:
    class _Geocoder:
        def forward_fulltext(self, query: str, max_results: int, epsg: int):
            assert query == "Berlin"
            assert max_results == 1
            assert epsg == 4326
            return {"Result": [{"BoundingBox": [13.1, 52.3, 13.8, 52.7], "Name": "Berlin"}]}

    monkeypatch.setattr("app.graph.nodes.GeocoderClient", lambda: _Geocoder())

    updates = geocoder_context_node({"spatial_reference": "Berlin"})

    assert updates["spatial_context_bbox"] == [13.1, 52.3, 13.8, 52.7]


def test_layer_discoverer_node_selects_layer(monkeypatch) -> None:
    class _Layer:
        layer_name = "topp:states"

    def fake_invoke_llm(*, messages, response_format):
        assert "Available layers" in messages[1]["content"]
        assert response_format.__name__ == "LayerSelection"
        return _Layer()

    monkeypatch.setattr("app.graph.nodes.invoke_llm", fake_invoke_llm)

    updates = layer_discoverer_node(
        {
            "user_query": "show states",
            "available_layers": [
                {"name": "topp:states", "title": "States", "abstract": "State boundaries"},
                {"name": "topp:roads", "title": "Roads", "abstract": "Road network"},
            ],
        }
    )

    assert updates["selected_layer"] == "topp:states"
    assert updates["validation_error"] is None


def test_layer_discoverer_node_uses_layer_subject_primary_single_match(monkeypatch) -> None:
    def should_not_call_llm(*, messages, response_format):
        raise AssertionError("LLM should not be called when layer_subject has a single deterministic match")

    monkeypatch.setattr("app.graph.nodes.invoke_llm", should_not_call_llm)

    updates = layer_discoverer_node(
        {
            "user_query": "find hospitals near bonn",
            "layer_subject": "hospitals",
            "available_layers": [
                {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare facilities"},
                {"name": "city:roads", "title": "Roads", "abstract": "Road network"},
            ],
        }
    )

    assert updates["selected_layer"] == "city:hospitals"
    assert updates["validation_error"] is None


def test_layer_discoverer_node_uses_filtered_candidates_for_llm_tiebreak(monkeypatch) -> None:
    class _Layer:
        layer_name = "city:hospitals_general"

    def fake_invoke_llm(*, messages, response_format):
        prompt = messages[1]["content"]
        assert "city:hospitals_general" in prompt
        assert "city:hospitals_specialized" in prompt
        assert "city:roads" not in prompt
        assert response_format.__name__ == "LayerSelection"
        return _Layer()

    monkeypatch.setattr("app.graph.nodes.invoke_llm", fake_invoke_llm)

    updates = layer_discoverer_node(
        {
            "user_query": "find hospitals",
            "layer_subject": "hospitals",
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
    )

    assert updates["selected_layer"] == "city:hospitals_general"
    assert updates["validation_error"] is None


def test_layer_discoverer_node_falls_back_to_full_list_when_subject_has_no_match(monkeypatch) -> None:
    class _Layer:
        layer_name = "city:roads"

    def fake_invoke_llm(*, messages, response_format):
        prompt = messages[1]["content"]
        assert "Layer subject:\nschools" in prompt
        assert "city:hospitals" in prompt
        assert "city:roads" in prompt
        assert response_format.__name__ == "LayerSelection"
        return _Layer()

    monkeypatch.setattr("app.graph.nodes.invoke_llm", fake_invoke_llm)

    updates = layer_discoverer_node(
        {
            "user_query": "find schools",
            "layer_subject": "schools",
            "available_layers": [
                {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare facilities"},
                {"name": "city:roads", "title": "Roads", "abstract": "Road network"},
            ],
        }
    )

    assert updates["selected_layer"] == "city:roads"
    assert updates["validation_error"] is None


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

    def fake_get_layer_schema(*, wfs_url: str, type_name: str, username: str, password: str):
        assert wfs_url == "https://wfs"
        assert type_name == "topp:states"
        assert username == "demo-user"
        assert password == "demo-pass"
        return {"STATE_NAME": "xsd:string"}, "the_geom"

    monkeypatch.setattr("app.graph.nodes.get_layer_schema", fake_get_layer_schema)

    updates = schema_extractor_node({"selected_layer": "topp:states"})

    assert updates["layer_schema"]["STATE_NAME"] == "xsd:string"
    assert updates["geometry_column"] == "the_geom"
    assert updates["validation_error"] is None


def test_ecql_generator_node_increments_retry_and_returns_ecql(monkeypatch) -> None:
    def fake_invoke_llm(*, messages, response_format):
        prompt = messages[1]["content"]
        assert "Previous validation error" in messages[1]["content"]
        assert "Spatial context bbox" in prompt
        assert "Attribute hints" in prompt
        assert "capacity > 100" in prompt
        assert "-10" in prompt
        assert "10" in prompt
        assert response_format is ECQLGeneration
        return ECQLGeneration(reasoning="Use geometry and population", ecql_string="PERSONS > 1000")

    monkeypatch.setattr("app.graph.nodes.invoke_llm", fake_invoke_llm)

    updates = ecql_generator_node(
        {
            "user_query": "states with high population",
            "layer_schema": {"PERSONS": "xsd:int"},
            "geometry_column": "the_geom",
            "spatial_context_bbox": [-10, -10, 10, 10],
            "attribute_hints": ["capacity > 100"],
            "validation_error": "Unknown attribute",
            "retry_count": 1,
        }
    )

    assert updates["generated_ecql"] == "PERSONS > 1000"
    assert updates["retry_count"] == 2


def test_synthesizer_node_returns_llm_summary(monkeypatch) -> None:
    def fake_invoke_llm(*, messages, response_format=None):
        assert response_format is None
        assert "Feature count" in messages[1]["content"]
        return "Found 2 matching features in the requested area."

    monkeypatch.setattr("app.graph.nodes.invoke_llm", fake_invoke_llm)

    updates = synthesizer_node(
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
    )

    assert updates == {
        "final_response": {
            "summary": "Found 2 matching features in the requested area.",
            "geojson": {
                "type": "FeatureCollection",
                "features": [
                    {"properties": {"STATE_NAME": "A"}},
                    {"properties": {"STATE_NAME": "B"}},
                ],
            },
        }
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

    def fake_discover_layers(*, wfs_url: str, username: str, password: str):
        assert wfs_url == "https://wfs"
        assert username == "demo-user"
        assert password == "demo-pass"
        return [{"name": "topp:states", "title": "States", "abstract": "Boundaries"}]

    monkeypatch.setattr("app.graph.nodes.discover_layers", fake_discover_layers)

    updates = wfs_discovery_node({"user_query": "find states"})

    assert updates["available_layers"][0]["name"] == "topp:states"
    assert updates["validation_error"] is None


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
            },
        )(),
    )

    def fake_execute_wfs_query(*, wfs_url: str, type_name: str, cql_filter: str, count: int, username: str, password: str):
        assert wfs_url == "https://wfs"
        assert type_name == "topp:states"
        assert cql_filter == "PERSONS > 1000"
        assert count == 1000
        assert username == "demo-user"
        assert password == "demo-pass"
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr("app.graph.nodes.execute_wfs_query", fake_execute_wfs_query)

    updates = wfs_executor_node(
        {
            "selected_layer": "topp:states",
            "generated_ecql": "PERSONS > 1000",
        }
    )

    assert updates["validation_error"] is None
    assert updates["wfs_result"]["type"] == "FeatureCollection"


def test_geocoder_context_node_no_spatial_reference_returns_empty_context() -> None:
    updates = geocoder_context_node({"user_query": "show schools"})

    assert updates == {"spatial_context_bbox": None}
