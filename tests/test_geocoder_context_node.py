import asyncio
from app.graph.nodes import geocoder_context_node

class DummyGeocoder:
    async def forward_fulltext(self, query: str, max_results: int, epsg: int):
        return {
            "Result": [
                {
                    "Coordinate": "49.362911,8.687586",
                    "MatchLevel": "City",
                    "MatchQuality": {"City": 1},
                }
            ]
        }

    async def suggest(self, query: str, max_results: int = 1):
        return {
            "SuggestResult": [
                {
                    "id": "1",
                    "locationType": "City",
                    "label": query,
                    "score": 1.0
                }
            ]
        }

def test_geocoder_context_node_explicit_distance(monkeypatch):
    # Explicit user intent distance (e.g., 7km)
    state = {
        "spatial_reference": "Heidelberg",
        "spatial_filters": [{"predicate": "DWITHIN", "distance": 7, "units": "kilometers"}],
        "geocoder_http_client": None,
    }
    monkeypatch.setattr("app.graph.nodes.GeocoderClient", lambda **kwargs: DummyGeocoder())
    # Patch transform and buffer logic (to be implemented)
    # For now, just check output structure and that a bbox is returned
    updates = asyncio.run(geocoder_context_node(state))
    assert "spatial_contexts" in updates
    bbox = updates["spatial_contexts"][0]["bbox"]
    assert isinstance(bbox, list) and len(bbox) == 4
    # City-level geocoding returns a buffered polygon context in EPSG:4326.
    assert updates["spatial_contexts"][0]["crs"] == "EPSG:4326"
    assert updates["spatial_contexts"][0]["geometry_type"] == "Polygon"


def test_geocoder_context_node_handles_multiple_predicates(monkeypatch):
    state = {
        "explicit_bbox": [[7.0, 50.0, 7.5, 50.5]],
        "explicit_coordinates": [[7.2, 50.2]],
        "spatial_filters": [
            None,
            {"predicate": "DWITHIN", "distance": 100, "units": "meters"},
        ],
    }

    updates = asyncio.run(geocoder_context_node(state))

    contexts = updates.get("spatial_contexts")
    assert isinstance(contexts, list)
    assert len(contexts) == 2

    assert contexts[0]["geometry_type"] == "Polygon"
    assert isinstance(contexts[0]["bbox"], list)
    assert len(contexts[0]["bbox"]) == 4

    assert contexts[1]["geometry_type"] == "Point"
    assert isinstance(contexts[1]["bbox"], list)
    assert len(contexts[1]["bbox"]) == 4
