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

def test_geocoder_context_node_bbox_epsg3857(monkeypatch):
    # Explicit user intent distance (e.g., 7km)
    state = {
        "spatial_reference": "Heidelberg",
        "spatial_filters": [{"predicate": "DWITHIN", "distance": 7, "units": "kilometers"}],
        "geocoder_http_client": None,
    }
    monkeypatch.setattr("app.graph.nodes.GeocoderClient", lambda **kwargs: DummyGeocoder())
    updates = asyncio.run(geocoder_context_node(state))
    context = updates["spatial_contexts"][0]
    bbox = context["bbox"]
    assert isinstance(bbox, list) and len(bbox) == 4
    assert context["crs"] == "EPSG:4326"
    assert context["geometry_type"] == "Polygon"
    assert context["geometry_wkt"].startswith("SRID=4326;POLYGON")
