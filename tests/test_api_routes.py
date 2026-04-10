import json

from fastapi.testclient import TestClient

from main import app


def _parse_sse_frames(body: str) -> list[tuple[str, dict[str, object]]]:
    frames: list[tuple[str, dict[str, object]]] = []
    for raw_frame in body.strip().split("\n\n"):
        if not raw_frame.strip():
            continue
        event_name = "message"
        data_payload: dict[str, object] = {}
        for line in raw_frame.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            if line.startswith("data: "):
                data_payload = json.loads(line.removeprefix("data: "))
        frames.append((event_name, data_payload))
    return frames


def test_spatial_chat_streams_graph_updates_as_sse(monkeypatch) -> None:
    class FakeGraph:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def astream(self, inputs, stream_mode, config):
            self.calls.append({"inputs": inputs, "stream_mode": stream_mode, "config": config})
            yield {"router": {"relevance": "spatial"}}
            yield {"schema": {"geometry_column": "the_geom"}}
            yield {
                "synthesizer": {
                    "final_response": {
                        "summary": "Found 2 matching features in the requested area."
                    }
                }
            }

    fake_graph = FakeGraph()
    monkeypatch.setattr("app.api.routes.graph", fake_graph)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/spatial-chat",
            json={"query": "Find hospitals in Berlin", "thread_id": "123"},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = "".join(response.iter_text())

    frames = _parse_sse_frames(body)

    assert frames[0] == ("status", {"thread_id": "123", "status": "starting"})
    assert frames[1] == ("update", {"thread_id": "123", "update": {"router": {"relevance": "spatial"}}})
    assert frames[2] == ("update", {"thread_id": "123", "update": {"schema": {"geometry_column": "the_geom"}}})
    assert frames[3] == (
        "final",
        {
            "thread_id": "123",
            "final_response": {
                "summary": "Found 2 matching features in the requested area."
            },
        },
    )
    assert frames[4] == ("done", {"thread_id": "123", "status": "completed"})

    assert fake_graph.calls[0]["inputs"]["user_query"] == "Find hospitals in Berlin"
    assert fake_graph.calls[0]["inputs"]["retry_count"] == 0
    assert fake_graph.calls[0]["stream_mode"] == "updates"
    config = fake_graph.calls[0]["config"]
    assert config["configurable"]["thread_id"] == "123"
    assert "geocoder_http_client" in config["configurable"]
    assert "wfs_http_client" in config["configurable"]


def test_spatial_chat_requires_query_and_thread_id() -> None:
    with TestClient(app) as client:
        response = client.post("/api/spatial-chat", json={"query": "Find hospitals in Berlin"})

    assert response.status_code == 422


def test_spatial_chat_streams_error_event_when_graph_fails(monkeypatch) -> None:
    class FakeGraph:
        async def astream(self, inputs, stream_mode, config):
            raise RuntimeError("boom")
            yield

    monkeypatch.setattr("app.api.routes.graph", FakeGraph())

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/spatial-chat",
            json={"query": "Find hospitals in Berlin", "thread_id": "err-1"},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

    frames = _parse_sse_frames(body)
    assert frames[0] == ("status", {"thread_id": "err-1", "status": "starting"})
    assert frames[1] == (
        "error",
        {"thread_id": "err-1", "status": "failed", "message": "boom"},
    )


def test_spatial_chat_emits_final_event_with_null_geojson_when_unavailable(monkeypatch) -> None:
    class FakeGraph:
        async def astream(self, inputs, stream_mode, config):
            yield {
                "router_analyzer": {
                    "intent": "general_chat",
                    "final_response": {
                        "summary": "Hello!"
                    },
                }
            }

    monkeypatch.setattr("app.api.routes.graph", FakeGraph())

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/spatial-chat",
            json={"query": "Hello", "thread_id": "chat-1"},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

    frames = _parse_sse_frames(body)
    assert frames[0] == ("status", {"thread_id": "chat-1", "status": "starting"})
    assert frames[1][0] == "update"
    assert frames[2] == (
        "final",
        {
            "thread_id": "chat-1",
            "final_response": {
                "summary": "Hello!"
            },
        },
    )
    assert frames[3] == ("done", {"thread_id": "chat-1", "status": "completed"})


def test_spatial_chat_redacts_available_layers_to_count(monkeypatch) -> None:
    class FakeGraph:
        async def astream(self, inputs, stream_mode, config):
            yield {
                "wfs_discovery": {
                    "available_layers": [
                        {"name": "layer_1", "title": "Layer One", "abstract": ""},
                        {"name": "layer_2", "title": "Layer Two", "abstract": ""},
                    ],
                    "layer_catalog_markdown": "# Catalog\n- layer_1\n- layer_2",
                    "layer_catalog_rows": [
                        {"name": "layer_1", "de_title": "Ebene Eins", "en_title": "Layer One"},
                        {"name": "layer_2", "de_title": "Ebene Zwei", "en_title": "Layer Two"},
                    ],
                }
            }

    monkeypatch.setattr("app.api.routes.graph", FakeGraph())

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/spatial-chat",
            json={"query": "find layers", "thread_id": "layers-1"},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

    frames = _parse_sse_frames(body)
    assert frames[0] == ("status", {"thread_id": "layers-1", "status": "starting"})
    assert frames[1] == (
        "update",
        {
            "thread_id": "layers-1",
            "update": {"wfs_discovery": {"available_layers_count": 2}},
        },
    )