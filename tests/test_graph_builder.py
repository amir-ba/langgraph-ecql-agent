from app.graph.builder import build_graph, executor_router, route_after_analysis, validator_router


def test_route_after_analysis_returns_expected_target() -> None:
    assert route_after_analysis({"intent": "spatial_query", "layer_subject": "hospitals"}) == "geocoder"
    assert route_after_analysis({"intent": "spatial_query", "layer_subject": None}) == "end"
    assert route_after_analysis({"intent": "general_chat"}) == "end"
    assert route_after_analysis({"intent": "irrelevant"}) == "end"


def test_validator_router_handles_retry_and_fallback() -> None:
    assert validator_router({"validation_error": "bad ecql", "retry_count": 1}) == "generator"
    assert validator_router({"validation_error": "bad ecql", "retry_count": 3}) == "fallback"
    assert validator_router({"validation_error": None, "retry_count": 1}) == "executor"


def test_executor_router_handles_retry_and_fallback() -> None:
    assert executor_router({"validation_error": "ows exception", "retry_count": 2}) == "generator"
    assert executor_router({"validation_error": "ows exception", "retry_count": 3}) == "fallback"
    assert executor_router({"validation_error": None, "retry_count": 1}) == "synthesizer"


def test_build_graph_spatial_path_success() -> None:
    call_order: list[str] = []

    def router_node(_):
        call_order.append("router_analyzer")
        return {"intent": "spatial_query", "layer_subject": "states"}

    def geocoder_node(_):
        call_order.append("geocoder")
        return {"spatial_context_bbox": [0, 0, 1, 1]}

    def discoverer_node(_):
        call_order.append("discoverer")
        return {"available_layers": [{"name": "topp:states", "title": "States", "abstract": ""}]}

    def layer_selector_node(_):
        call_order.append("layer_selector")
        return {"selected_layer": "topp:states"}

    def schema_node(_):
        call_order.append("schema")
        return {"layer_schema": {"PERSONS": "xsd:int"}, "geometry_column": "the_geom"}

    def generator_node(state):
        call_order.append("generator")
        return {"generated_ecql": "PERSONS > 1000", "retry_count": int(state.get("retry_count", 0)) + 1}

    def validator_node(_):
        call_order.append("validator")
        return {"validation_error": None}

    def executor_node(_):
        call_order.append("executor")
        return {"wfs_result": {"features": []}, "validation_error": None}

    def synthesizer_node(_):
        call_order.append("synthesizer")
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

    result = graph.invoke({"user_query": "states near berlin", "retry_count": 0})

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

    def router_node(_):
        return {"intent": "spatial_query", "layer_subject": "states"}

    def geocoder_node(_):
        return {"spatial_context_bbox": [0, 0, 1, 1]}

    def discoverer_node(_):
        return {"available_layers": [{"name": "topp:states", "title": "States", "abstract": ""}]}

    def layer_selector_node(_):
        return {"selected_layer": "topp:states"}

    def schema_node(_):
        return {"layer_schema": {"PERSONS": "xsd:int"}, "geometry_column": "the_geom"}

    def generator_node(state):
        nonlocal generator_calls
        generator_calls += 1
        return {"generated_ecql": "PERSONS > 1000", "retry_count": int(state.get("retry_count", 0)) + 1}

    def validator_node(_):
        nonlocal validator_calls
        validator_calls += 1
        if validator_calls == 1:
            return {"validation_error": "Unknown attribute"}
        return {"validation_error": None}

    def executor_node(_):
        return {"wfs_result": {"features": []}, "validation_error": None}

    def synthesizer_node(_):
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

    result = graph.invoke({"user_query": "states near berlin", "retry_count": 0})

    assert result["final_response"]["summary"] == "ok"
    assert generator_calls == 2
    assert validator_calls == 2


def test_build_graph_executor_retry_loop() -> None:
    executor_calls = 0
    generator_calls = 0

    def router_node(_):
        return {"intent": "spatial_query", "layer_subject": "states"}

    def geocoder_node(_):
        return {"spatial_context_bbox": [0, 0, 1, 1]}

    def discoverer_node(_):
        return {"available_layers": [{"name": "topp:states", "title": "States", "abstract": ""}]}

    def layer_selector_node(_):
        return {"selected_layer": "topp:states"}

    def schema_node(_):
        return {"layer_schema": {"PERSONS": "xsd:int"}, "geometry_column": "the_geom"}

    def generator_node(state):
        nonlocal generator_calls
        generator_calls += 1
        return {"generated_ecql": "PERSONS > 1000", "retry_count": int(state.get("retry_count", 0)) + 1}

    def validator_node(_):
        return {"validation_error": None}

    def executor_node(_):
        nonlocal executor_calls
        executor_calls += 1
        if executor_calls == 1:
            return {"validation_error": "OWS ExceptionReport"}
        return {"wfs_result": {"features": []}, "validation_error": None}

    def synthesizer_node(_):
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

    result = graph.invoke({"user_query": "states near berlin", "retry_count": 0})

    assert result["final_response"]["summary"] == "ok"
    assert executor_calls == 2
    assert generator_calls == 2


def test_build_graph_general_chat_ends_after_router() -> None:
    call_order: list[str] = []

    def router_node(_):
        call_order.append("router_analyzer")
        return {
            "intent": "general_chat",
            "final_response": {"summary": "Hello!", "geojson": None},
            "layer_subject": None,
        }

    graph = build_graph({"router_analyzer": router_node})

    result = graph.invoke({"user_query": "hello", "retry_count": 0})

    assert result["final_response"]["summary"] == "Hello!"
    assert call_order == ["router_analyzer"]
