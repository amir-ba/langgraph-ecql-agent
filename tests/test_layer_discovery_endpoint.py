from fastapi.testclient import TestClient

from main import app


def test_layer_discovery_endpoint_returns_matches(monkeypatch) -> None:
    async def fake_wfs_discovery_node(state, config=None):
        return {
            "available_layers": [
                {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare"},
            ],
            "layer_catalog_markdown": "# catalog",
            "layer_catalog_rows": [
                {
                    "name": "city:hospitals",
                    "de_title": "Krankenhäuser",
                    "en_title": "Hospitals",
                    "de_abstract": "Gesundheit",
                    "en_abstract": "Healthcare",
                    "aliases": ["hospitals"],
                }
            ],
            "validation_error": None,
        }

    async def fake_layer_discoverer_node(state, config=None):
        return {
            "selected_layer": "city:hospitals",
            "validation_error": None,
            "retrieval_mode": "semantic",
            "retrieval_top_score": 0.93,
            "retrieval_score_gap": 0.42,
            "retrieval_reason": "top=city:hospitals score=0.930 gap=0.420",
        }

    monkeypatch.setattr("app.api.routes.wfs_discovery_node", fake_wfs_discovery_node)
    monkeypatch.setattr("app.api.routes.layer_discoverer_node", fake_layer_discoverer_node)

    with TestClient(app) as client:
        resp = client.post("/api/layer-discovery", json={"query": "hospitals nearby"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["layer_name"] == "city:hospitals"
    assert data["validation_error"] is None
    assert data["retrieval_mode"] == "semantic"
    assert data["retrieval_top_score"] > 0.5


def test_layer_discovery_endpoint_rejects_empty_query() -> None:
    with TestClient(app) as client:
        resp = client.post("/api/layer-discovery", json={"query": ""})
    assert resp.status_code == 422


def test_layer_discovery_endpoint_returns_validation_error_when_selection_fails(monkeypatch) -> None:
    async def fake_wfs_discovery_node(state, config=None):
        return {
            "available_layers": [],
            "layer_catalog_markdown": "",
            "layer_catalog_rows": [],
            "validation_error": None,
        }

    async def fake_layer_discoverer_node(state, config=None):
        return {
            "selected_layer": "",
            "validation_error": "Low-confidence layer retrieval",
            "retrieval_mode": "semantic",
            "retrieval_top_score": 0.1,
            "retrieval_score_gap": 0.0,
            "retrieval_reason": "semantic-index-unavailable",
        }

    monkeypatch.setattr("app.api.routes.wfs_discovery_node", fake_wfs_discovery_node)
    monkeypatch.setattr("app.api.routes.layer_discoverer_node", fake_layer_discoverer_node)

    with TestClient(app) as client:
        resp = client.post("/api/layer-discovery", json={"query": "anything"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["layer_name"] == ""
    assert data["validation_error"] == "Low-confidence layer retrieval"
