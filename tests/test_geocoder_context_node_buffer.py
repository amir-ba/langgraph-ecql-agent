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

def test_geocoder_context_node_city_matchlevel(monkeypatch):
    # No explicit distance, should use city (10km)
    state = {
        "spatial_reference": "Heidelberg",
        "geocoder_http_client": None,
    }
    monkeypatch.setattr("app.graph.nodes.GeocoderClient", lambda **kwargs: DummyGeocoder())
    updates = asyncio.run(geocoder_context_node(state))
    bbox = updates["spatial_context"]["bbox"]
    assert isinstance(bbox, list) and len(bbox) == 4
    assert updates["spatial_context"]["crs"] == "EPSG:4326"
    assert updates["spatial_context"]["geometry_type"] == "Polygon"

def test_geocoder_context_node_fallback_default(monkeypatch):
    # No explicit distance, no matchlevel, should use fallback (5km)
    class DummyGeocoderNoLevel:
        async def suggest(self, query: str, max_results: int = 1):
            return {
                "SuggestResult": [
                    {
                        "id": "1",
                        "locationType": "Unknown",
                        "label": query,
                        "score": 1.0
                    }
                ]
            }
        async def forward_fulltext(self, query: str, max_results: int, epsg: int):
            return {
                "Result": [
                    {
                        "Coordinate": "49.362911,8.687586",
                        # No MatchLevel
                        "MatchQuality": {},
                    }
                ]
            }
    state = {
        "spatial_reference": "Heidelberg",
        "geocoder_http_client": None,
    }
    monkeypatch.setattr("app.graph.nodes.GeocoderClient", lambda **kwargs: DummyGeocoderNoLevel())
    updates = asyncio.run(geocoder_context_node(state))
    bbox = updates["spatial_context"]["bbox"]
    assert isinstance(bbox, list) and len(bbox) == 4
    assert updates["spatial_context"]["crs"] == "EPSG:4326"
    assert updates["spatial_context"]["geometry_type"] == "Point"
