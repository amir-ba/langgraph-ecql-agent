import asyncio

from app.graph.builder import (
    build_graph,
    executor_router,
    layer_selector_router,
    route_after_geocoder,
    route_after_analysis,
    route_after_generator,
    validator_router,
)


def test_route_after_analysis_returns_expected_target() -> None:
    assert route_after_analysis({"intent": "spatial_query", "layer_subject": "hospitals"}) == "geocoder"
    assert route_after_analysis({"intent": "spatial_query", "layer_subject": None}) == "end"
    assert route_after_analysis({"intent": "irrelevant"}) == "end"


def test_route_after_geocoder_handles_required_target_resolution_failures() -> None:
    assert route_after_geocoder({"validation_error": None}) == "discoverer"
    assert route_after_geocoder({"validation_error": "location_unresolved:\"r1\""}) == "fallback"


def test_validator_router_handles_retry_and_fallback() -> None:
    assert validator_router({"validation_error": "bad ecql", "retry_count": 1}) == "generator"
    assert validator_router({"validation_error": "bad ecql", "retry_count": 3}) == "fallback"
    assert validator_router({"validation_error": None, "retry_count": 1}) == "executor"


def test_layer_selector_router_handles_low_confidence_fallback() -> None:
    assert layer_selector_router({"validation_error": "Low-confidence layer retrieval", "selected_layer": ""}) == "fallback"
    assert layer_selector_router({"validation_error": None, "selected_layer": "topp:states"}) == "schema"


def test_executor_router_handles_retry_and_fallback() -> None:
    assert executor_router({"validation_error": "ows exception", "retry_count": 2}) == "generator"
    assert executor_router({"validation_error": "ows exception", "retry_count": 3}) == "fallback"
    assert executor_router({"validation_error": None, "retry_count": 1}) == "synthesizer"


# Slice 5 — Guard empty ECQL → fallback
def test_graph_routes_to_fallback_when_ecql_is_empty() -> None:
    assert route_after_generator({"generated_ecql": ""}) == "fallback"
    assert route_after_generator({"generated_ecql": "   "}) == "fallback"
    assert route_after_generator({}) == "fallback"


def test_graph_routes_to_fallback_when_ecql_is_none_string() -> None:
    assert route_after_generator({"generated_ecql": "NONE"}) == "fallback"
    assert route_after_generator({"generated_ecql": "none"}) == "fallback"
    assert route_after_generator({"generated_ecql": "NULL"}) == "fallback"


def test_graph_routes_to_validator_when_ecql_is_present() -> None:
    assert route_after_generator({"generated_ecql": "PERSONS > 1000"}) == "validator"


def test_build_graph_spatial_path_success() -> None:
    call_order: list[str] = []

    async def router_node(_):
        call_order.append("router_analyzer")
        return {"intent": "spatial_query", "layer_subject": "states"}

    async def geocoder_node(_):
        call_order.append("geocoder")
        return {"spatial_contexts": [{"crs": "EPSG:4326", "bbox": [0, 0, 1, 1]}]}

    async def discoverer_node(_):
        call_order.append("discoverer")
        return {"available_layers": [{"name": "topp:states", "title": "States", "abstract": ""}]}

    async def layer_selector_node(_):
        call_order.append("layer_selector")
        return {"selected_layer": "topp:states"}

    async def schema_node(_):
        call_order.append("schema")
        return {"layer_schema": {"PERSONS": "xsd:int"}, "geometry_column": "the_geom"}

    async def generator_node(state):
        call_order.append("generator")
        return {"generated_ecql": "PERSONS > 1000", "retry_count": int(state.get("retry_count", 0)) + 1}

    async def validator_node(_):
        call_order.append("validator")
        return {"validation_error": None}

    async def executor_node(_):
        call_order.append("executor")
        return {"wfs_result": {"features": []}, "validation_error": None}

    async def synthesizer_node(_):
        call_order.append("synthesizer")
        return {"final_response": {"summary": "ok"}}

    graph = build_graph(
        {
            "router_analyzer": router_node,
            "geocoder": geocoder_node,
            "discoverer": discoverer_node,
            "layer_selector": layer_selector_node,
            "schema": schema_node,
            "generator": generator_node,
            "validator": validator_node,
            "executor": executor_node,
            "synthesizer": synthesizer_node,
        }
    )

    result = asyncio.run(graph.ainvoke({"user_query": "states near berlin", "retry_count": 0}))

    assert result["final_response"]["summary"] == "ok"
    assert call_order == [
        "router_analyzer",
        "geocoder",
        "discoverer",
        "layer_selector",
        "schema",
        "generator",
        "validator",
        "executor",
        "synthesizer",
    ]


def test_build_graph_validator_retry_loop() -> None:
    validator_calls = 0
    generator_calls = 0

    async def router_node(_):
        return {"intent": "spatial_query", "layer_subject": "states"}

    async def geocoder_node(_):
        return {"spatial_contexts": [{"crs": "EPSG:4326", "bbox": [0, 0, 1, 1]}]}

    async def discoverer_node(_):
        return {"available_layers": [{"name": "topp:states", "title": "States", "abstract": ""}]}

    async def layer_selector_node(_):
        return {"selected_layer": "topp:states"}

    async def schema_node(_):
        return {"layer_schema": {"PERSONS": "xsd:int"}, "geometry_column": "the_geom"}

    async def generator_node(state):
        nonlocal generator_calls
        generator_calls += 1
        return {"generated_ecql": "PERSONS > 1000", "retry_count": int(state.get("retry_count", 0)) + 1}

    async def validator_node(_):
        nonlocal validator_calls
        validator_calls += 1
        if validator_calls == 1:
            return {"validation_error": "Unknown attribute"}
        return {"validation_error": None}

    async def executor_node(_):
        return {"wfs_result": {"features": []}, "validation_error": None}

    async def synthesizer_node(_):
        return {"final_response": {"summary": "ok", "geojson": {"type": "FeatureCollection", "features": []}}}

    graph = build_graph(
        {
            "router_analyzer": router_node,
            "geocoder": geocoder_node,
            "discoverer": discoverer_node,
            "layer_selector": layer_selector_node,
            "schema": schema_node,
            "generator": generator_node,
            "validator": validator_node,
            "executor": executor_node,
            "synthesizer": synthesizer_node,
        }
    )

    result = asyncio.run(graph.ainvoke({"user_query": "states near berlin", "retry_count": 0}))

    assert result["final_response"]["summary"] == "ok"
    assert generator_calls == 2
    assert validator_calls == 2


def test_build_graph_executor_retry_loop() -> None:
    executor_calls = 0
    generator_calls = 0

    async def router_node(_):
        return {"intent": "spatial_query", "layer_subject": "states"}

    async def geocoder_node(_):
        return {"spatial_contexts": [{"crs": "EPSG:4326", "bbox": [0, 0, 1, 1]}]}

    async def discoverer_node(_):
        return {"available_layers": [{"name": "topp:states", "title": "States", "abstract": ""}]}

    async def layer_selector_node(_):
        return {"selected_layer": "topp:states"}

    async def schema_node(_):
        return {"layer_schema": {"PERSONS": "xsd:int"}, "geometry_column": "the_geom"}

    async def generator_node(state):
        nonlocal generator_calls
        generator_calls += 1
        return {"generated_ecql": "PERSONS > 1000", "retry_count": int(state.get("retry_count", 0)) + 1}

    async def validator_node(_):
        return {"validation_error": None}

    async def executor_node(_):
        nonlocal executor_calls
        executor_calls += 1
        if executor_calls == 1:
            return {"validation_error": "OWS ExceptionReport"}
        return {"wfs_result": {"features": []}, "validation_error": None}

    async def synthesizer_node(_):
        return {"final_response": {"summary": "ok"}}

    graph = build_graph(
        {
            "router_analyzer": router_node,
            "geocoder": geocoder_node,
            "discoverer": discoverer_node,
            "layer_selector": layer_selector_node,
            "schema": schema_node,
            "generator": generator_node,
            "validator": validator_node,
            "executor": executor_node,
            "synthesizer": synthesizer_node,
        }
    )

    result = asyncio.run(graph.ainvoke({"user_query": "states near berlin", "retry_count": 0}))

    assert result["final_response"]["summary"] == "ok"
    assert executor_calls == 2
    assert generator_calls == 2


def test_build_graph_irrelevant_ends_after_router() -> None:
    call_order: list[str] = []

    async def router_node(_):
        call_order.append("router_analyzer")
        return {
            "intent": "irrelevant",
            "final_response": {"summary": "Hello!"},
            "layer_subject": None,
        }

    graph = build_graph({"router_analyzer": router_node})

    result = asyncio.run(graph.ainvoke({"user_query": "hello", "retry_count": 0}))

    assert result["final_response"]["summary"] == "Hello!"
    assert call_order == ["router_analyzer"]


def test_build_graph_routes_to_fallback_after_layer_selector_low_confidence() -> None:
    call_order: list[str] = []

    async def router_node(_):
        call_order.append("router_analyzer")
        return {"intent": "spatial_query", "layer_subject": "hospitals"}

    async def geocoder_node(_):
        call_order.append("geocoder")
        return {"spatial_contexts": [{"crs": "EPSG:4326", "bbox": [0, 0, 1, 1]}]}

    async def discoverer_node(_):
        call_order.append("discoverer")
        return {"available_layers": [{"name": "city:roads", "title": "Roads", "abstract": "Road network"}]}

    async def layer_selector_node(_):
        call_order.append("layer_selector")
        return {
            "selected_layer": "",
            "validation_error": "Low-confidence layer retrieval",
            "final_response": {"summary": "Could not confidently map your request to a layer."},
        }

    async def fallback_node(_):
        call_order.append("fallback")
        return {"final_response": {"summary": "Could not confidently map your request to a layer."}}

    graph = build_graph(
        {
            "router_analyzer": router_node,
            "geocoder": geocoder_node,
            "discoverer": discoverer_node,
            "layer_selector": layer_selector_node,
            "fallback": fallback_node,
        }
    )

    result = asyncio.run(graph.ainvoke({"user_query": "find hospitals", "retry_count": 0}))

    assert result["final_response"]["summary"] == "Could not confidently map your request to a layer."
    assert call_order == [
        "router_analyzer",
        "geocoder",
        "discoverer",
        "layer_selector",
        "fallback",
    ]
