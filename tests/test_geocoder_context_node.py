"""Geocoder context tests for id-bound spatial targets and predicates."""

import asyncio

from app.graph.nodes import geocoder_context_node


class DummyGeocoder:
    async def forward_fulltext(self, query: str, max_results: int, epsg: int):
        if query == "Heidelberg":
            return {
                "Result": [
                    {
                        "Coordinate": "49.362911,8.687586",
                        "MatchLevel": "City",
                        "MatchQuality": {"City": 1},
                    }
                ]
            }
        return {"Result": []}

    async def suggest(self, query: str, max_results: int = 1):
        if query == "Heidelberg":
            return {
                "SuggestResult": [
                    {
                        "id": "1",
                        "locationType": "City",
                        "label": query,
                        "score": 1.0,
                    }
                ]
            }
        return {"SuggestResult": []}


def test_geocoder_context_node_explicit_distance(monkeypatch):
    state = {
        "spatial_targets": [
            {
                "id": "r1",
                "kind": "spatial_reference",
                "value": "Heidelberg",
                "required": True,
            }
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
    assert "spatial_contexts" in updates
    bbox = updates["spatial_contexts"][0]["bbox"]
    assert isinstance(bbox, list) and len(bbox) == 4
    assert updates["spatial_contexts"][0]["crs"] == "EPSG:4326"
    assert updates["spatial_contexts"][0]["geometry_type"] == "Polygon"


def test_geocoder_context_node_handles_multiple_predicates():
    state = {
        "spatial_targets": [
            {
                "id": "g1",
                "kind": "explicit_geometry",
                "value": "POLYGON((7.0 50.0, 7.5 50.0, 7.5 50.5, 7.0 50.5, 7.0 50.0))",
                "required": True,
            },
            {
                "id": "g2",
                "kind": "explicit_geometry",
                "value": "POINT(7.2 50.2)",
                "required": True,
            },
        ],
        "spatial_predicates": [
            {
                "id": "p1",
                "predicate": "INTERSECTS",
                "target_ids": ["g1"],
                "join_with_next": "AND",
                "required": True,
            },
            {
                "id": "p2",
                "predicate": "DWITHIN",
                "target_ids": ["g2"],
                "distance": 100,
                "units": "meters",
                "join_with_next": "AND",
                "required": True,
            },
        ],
    }

    updates = asyncio.run(geocoder_context_node(state))

    contexts = updates.get("spatial_contexts")
    assert isinstance(contexts, list)
    assert len(contexts) == 2

    assert contexts[0]["target_id"] == "g1"
    assert contexts[0]["geometry_type"] == "Polygon"
    assert isinstance(contexts[0]["bbox"], list)
    assert len(contexts[0]["bbox"]) == 4

    assert contexts[1]["target_id"] == "g2"
    assert contexts[1]["geometry_type"] == "Point"
    assert isinstance(contexts[1]["bbox"], list)
    assert len(contexts[1]["bbox"]) == 4


def test_geocoder_context_node_uses_explicit_geometry_without_geocoding(monkeypatch):
    class _FailingGeocoder:
        async def forward_fulltext(self, query: str, max_results: int, epsg: int):
            raise AssertionError("forward_fulltext should not be called")

        async def suggest(self, query: str, max_results: int = 1):
            raise AssertionError("suggest should not be called")

    monkeypatch.setattr("app.graph.nodes.GeocoderClient", lambda **kwargs: _FailingGeocoder())

    state = {
        "spatial_targets": [
            {
                "id": "g1",
                "kind": "explicit_geometry",
                "value": "POLYGON((7.0 50.0, 7.5 50.0, 7.5 50.5, 7.0 50.5, 7.0 50.0))",
                "required": True,
            }
        ],
        "spatial_predicates": [
            {
                "id": "p1",
                "predicate": "INTERSECTS",
                "target_ids": ["g1"],
                "join_with_next": "AND",
                "required": True,
            }
        ],
    }

    updates = asyncio.run(geocoder_context_node(state))
    contexts = updates.get("spatial_contexts")
    assert isinstance(contexts, list)
    assert len(contexts) == 1
    assert contexts[0]["source"] == "explicit_geometry"
    assert contexts[0]["geometry_type"] == "Polygon"
