import asyncio

from app.core.schemas import AnalyzedIntent, ECQLGeneration, SpatialFilterDef
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
    assert state.get("spatial_filters") is None
    assert state["layer_subject"] is None
    assert state["attribute_hints"] == []
    assert state["spatial_contexts"] == []
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
        spatial_reference="Bonn",
        spatial_filters=[SpatialFilterDef(predicate="DWITHIN", distance=5, units="kilometers")],
        layer_subject="schools",
        attribute_hints=["capacity > 100"],
    )

    async def fake_ainvoke_llm(
        *,
        messages,
        output_schema=None,
        response_format=None,
        agent_state=None,
        model_name=None,
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
        "spatial_reference": "Bonn",
        "explicit_coordinates": None,
        "explicit_bbox": None,
        "spatial_filters": [
            {
                "predicate": "DWITHIN",
                "distance": 5.0,
                "units": "kilometers",
            }
        ],
        "layer_subject": "schools",
        "attribute_hints": ["capacity > 100"],
    }


def test_unified_router_analyzer_node_returns_irrelevant_response(monkeypatch) -> None:
    parsed_intent = AnalyzedIntent(
        intent="irrelevant",
        general_response="Hello! How can I help with geospatial analysis today?",
        spatial_reference=None,
        spatial_filters=None,
        layer_subject=None,
        attribute_hints=None,
    )

    async def fake_ainvoke_llm(
        *,
        messages,
        output_schema=None,
        response_format=None,
        agent_state=None,
        model_name=None,
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
        "spatial_reference": None,
        "explicit_coordinates": None,
        "explicit_bbox": None,
        "spatial_filters": None,
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

    updates = asyncio.run(geocoder_context_node({"spatial_reference": "Berlin"}))

    context = updates["spatial_contexts"][0]
    assert isinstance(context["bbox"], list)
    assert len(context["bbox"]) == 4
    assert context["crs"] == "EPSG:4326"
    assert context["geometry_type"] == "Polygon"


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
    ):
        assert "Layer catalog markdown" in messages[1]["content"]
        assert response_format.__name__ == "LayerSelection"
        return _Layer()

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    updates = asyncio.run(layer_discoverer_node(
        {
            "user_query": "show states",
            "layer_catalog_markdown": "# GeoServer Layer Catalog\n\n- **Layer ID:** `topp:states`",
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
    ):
        assert "Layer subject:\nhospitals" in messages[1]["content"]
        return _Layer()

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    updates = asyncio.run(layer_discoverer_node(
        {
            "user_query": "find hospitals near bonn",
            "layer_subject": "hospitals",
            "layer_catalog_markdown": "# GeoServer Layer Catalog\n\n- **Layer ID:** `city:hospitals`",
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
            "layer_catalog_markdown": (
                "# GeoServer Layer Catalog\n\n"
                "- **Layer ID:** `city:hospitals_general`\n"
                "  - **EN Translation:** Hospitals\n"
                "- **Layer ID:** `city:hospitals_specialized`\n"
                "  - **EN Translation:** Specialized Hospitals\n"
                "- **Layer ID:** `city:roads`\n"
                "  - **EN Translation:** Roads"
            ),
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
            "layer_catalog_markdown": (
                "# GeoServer Layer Catalog\n\n"
                "- **Layer ID:** `city:hospitals`\n"
                "  - **EN Translation:** Hospitals\n"
                "- **Layer ID:** `city:roads`\n"
                "  - **EN Translation:** Roads"
            ),
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
    ):
        assert "Layer catalog markdown" in messages[1]["content"]
        return _Layer()

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    updates = asyncio.run(layer_discoverer_node(
        {
            "user_query": "find hospitals",
            "layer_subject": "hospitals",
            "layer_catalog_markdown": "# GeoServer Layer Catalog\n\n- **Layer ID:** `city:roads`",
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
    async def fake_ainvoke_llm(*, messages, response_format, agent_state=None):
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
                    "crs": "EPSG:4326",
                    "bbox": [-10, -10, 10, 10],
                    "geometry_wkt": "POLYGON ((-10 -10, 10 -10, 10 10, -10 10, -10 -10))",
                    "geometry_type": "Polygon",
                },
            ],
            "spatial_filters": [{"predicate": "INTERSECTS"}],
            "attribute_hints": ["capacity > 100"],
            "validation_error": "Unknown attribute",
            "retry_count": 1,
        }
    ))

    assert "INTERSECTS(the_geom" in updates["generated_ecql"]
    assert "PERSONS > 1000" in updates["generated_ecql"]
    assert updates["retry_count"] == 2


def test_ecql_generator_node_builds_multi_spatial_ecql(monkeypatch) -> None:
    async def fake_ainvoke_llm(*, messages, response_format, agent_state=None):
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
                    "crs": "EPSG:4326",
                    "bbox": [7.0, 50.0, 7.5, 50.5],
                    "geometry_wkt": (
                        "POLYGON ((7.0 50.0, 7.5 50.0, 7.5 50.5, 7.0 50.5, 7.0 50.0))"
                    ),
                    "geometry_type": "Polygon",
                },
                {
                    "crs": "EPSG:4326",
                    "bbox": [7.19, 50.19, 7.21, 50.21],
                    "geometry_wkt": "POINT (7.2 50.2)",
                    "geometry_type": "Point",
                },
            ],
            "spatial_filters": [
                None,
                {"predicate": "DWITHIN", "distance": 100, "units": "meters"},
            ],
            "attribute_hints": [],
            "validation_error": None,
            "retry_count": 0,
        }
    ))

    assert "BBOX(the_geom, 7.0, 50.0, 7.5, 50.5, 'EPSG:4326')" in updates["generated_ecql"]
    assert "DWITHIN(the_geom, SRID=4326;POINT (7.2 50.2), 100, meters)" in updates["generated_ecql"]
    assert "PERSONS > 1000" in updates["generated_ecql"]


def test_ecql_generator_node_avoids_duplicate_bbox_for_duplicated_reference_context(monkeypatch) -> None:
    async def fake_ainvoke_llm(*, messages, response_format, agent_state=None):
        assert response_format is ECQLGeneration
        return ECQLGeneration(reasoning="Use attributes only", ecql_string="manager = 'NBG-Maps'")

    monkeypatch.setattr("app.graph.nodes.ainvoke_llm", fake_ainvoke_llm)

    hamburg_context = {
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
            "spatial_filters": [{"predicate": "INTERSECTS"}],
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
    async def fake_ainvoke_llm(*, messages, response_format=None, agent_state=None, model_name=None):
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

    monkeypatch.setattr("app.graph.nodes.discover_layers", fake_discover_layers)

    updates = asyncio.run(wfs_discovery_node({"user_query": "find states"}))

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


def test_geocoder_context_node_no_spatial_reference_returns_empty_context() -> None:
    updates = asyncio.run(geocoder_context_node({"user_query": "show schools"}))

    assert updates == {"spatial_contexts": []}
