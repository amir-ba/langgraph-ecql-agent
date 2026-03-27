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
    state = {
        "spatial_targets": [
            {"id": "r1", "kind": "spatial_reference", "value": "Heidelberg", "required": True}
        ],
        "spatial_predicates": [
            {
                "id": "p1",
                "predicate": "DWITHIN",
                "target_ids": ["r1"],
                "distance": 7,
                "units": "kilometers",
                "join_with_next": "AND",
                "required": True,
            }
        ],
    }
    monkeypatch.setattr("app.graph.nodes.GeocoderClient", lambda **kwargs: DummyGeocoder())
    updates = asyncio.run(geocoder_context_node(state))
    context = updates["spatial_contexts"][0]
    bbox = context["bbox"]
    assert isinstance(bbox, list) and len(bbox) == 4
    assert context["crs"] == "EPSG:4326"
    assert context["geometry_type"] == "Polygon"
    assert context["geometry_wkt"].startswith("SRID=4326;POLYGON")
