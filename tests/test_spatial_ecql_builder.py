"""Spatial ECQL builder tests for id-bound target/predicate mapping."""

from app.tools.spatial_ecql_builder import build_spatial_ecql


def test_build_spatial_ecql_returns_none_without_context() -> None:
    assert build_spatial_ecql("the_geom", None, None) is None


# Slice 4 — Guard empty geom_column
def test_build_spatial_ecql_returns_none_for_empty_geom_column() -> None:
    spatial_contexts = [
        {
            "target_id": "g1",
            "crs": "EPSG:4326",
            "bbox": [13.1, 52.3, 13.8, 52.7],
            "geometry_wkt": "POINT (13.45 52.5)",
            "geometry_type": "Point",
        }
    ]
    assert build_spatial_ecql("", spatial_contexts, None) is None
    assert build_spatial_ecql("   ", spatial_contexts, None) is None


def test_build_spatial_ecql_builds_bbox_only_without_predicates() -> None:
    spatial_contexts = [
        {
            "target_id": "g1",
            "crs": "EPSG:4326",
            "bbox": [13.1, 52.3, 13.8, 52.7],
            "geometry_wkt": "POINT (13.45 52.5)",
            "geometry_type": "Point",
        }
    ]

    ecql = build_spatial_ecql("the_geom", spatial_contexts, None)

    assert ecql == "BBOX(the_geom, 13.1, 52.3, 13.8, 52.7, 'EPSG:4326')"


def test_build_spatial_ecql_builds_distance_predicate_for_dwithin() -> None:
    spatial_contexts = [
        {
            "target_id": "g1",
            "crs": "EPSG:4326",
            "bbox": [13.1, 52.3, 13.8, 52.7],
            "geometry_wkt": "POINT (13.45 52.5)",
            "geometry_type": "Point",
        }
    ]
    spatial_predicates = [
        {
            "id": "p1",
            "predicate": "DWITHIN",
            "target_ids": ["g1"],
            "distance": 5,
            "units": "kilometers",
            "join_with_next": "AND",
            "required": True,
        }
    ]

    ecql = build_spatial_ecql("the_geom", spatial_contexts, spatial_predicates)

    assert ecql is not None
    assert "BBOX(the_geom, 13.1, 52.3, 13.8, 52.7, 'EPSG:4326')" not in ecql
    assert "DWITHIN(the_geom, SRID=4326;POINT (13.45 52.5), 5, kilometers)" in ecql


def test_build_spatial_ecql_uses_polygon_for_topological_predicate() -> None:
    spatial_contexts = [
        {
            "target_id": "g1",
            "crs": "EPSG:4326",
            "bbox": [13.1, 52.3, 13.8, 52.7],
            "geometry_wkt": "POLYGON ((13.1 52.3, 13.8 52.3, 13.8 52.7, 13.1 52.7, 13.1 52.3))",
            "geometry_type": "Polygon",
        }
    ]
    spatial_predicates = [
        {
            "id": "p1",
            "predicate": "INTERSECTS",
            "target_ids": ["g1"],
            "join_with_next": "AND",
            "required": True,
        }
    ]

    ecql = build_spatial_ecql("the_geom", spatial_contexts, spatial_predicates)

    assert ecql == "INTERSECTS(the_geom, SRID=4326;POLYGON ((13.1 52.3, 13.8 52.3, 13.8 52.7, 13.1 52.7, 13.1 52.3)))"


def test_build_spatial_ecql_builds_multi_predicates_with_and() -> None:
    spatial_contexts = [
        {
            "target_id": "g1",
            "crs": "EPSG:4326",
            "bbox": [7.0, 50.0, 7.5, 50.5],
            "geometry_wkt": "POLYGON ((7.0 50.0, 7.5 50.0, 7.5 50.5, 7.0 50.5, 7.0 50.0))",
            "geometry_type": "Polygon",
        },
        {
            "target_id": "g2",
            "crs": "EPSG:4326",
            "bbox": [7.19, 50.19, 7.21, 50.21],
            "geometry_wkt": "POINT (7.2 50.2)",
            "geometry_type": "Point",
        },
    ]
    spatial_predicates = [
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
    ]

    ecql = build_spatial_ecql("the_geom", spatial_contexts, spatial_predicates)

    assert ecql is not None
    assert " AND " in ecql
    assert "INTERSECTS(the_geom, SRID=4326;POLYGON" in ecql
    assert "DWITHIN(the_geom, SRID=4326;POINT (7.2 50.2), 100, meters)" in ecql


def test_build_spatial_ecql_deterministically_applies_binary_target_binding() -> None:
    spatial_contexts = [
        {
            "target_id": "g1",
            "source": "explicit_geometry",
            "crs": "EPSG:4326",
            "bbox": [7.1991, 50.1994, 7.2009, 50.2006],
            "geometry_wkt": "POINT (7.2 50.2)",
            "geometry_type": "Point",
        },
        {
            "target_id": "r1",
            "source": "reference",
            "crs": "EPSG:4326",
            "bbox": [9.5691, 50.8971, 9.6050, 50.9198],
            "geometry_wkt": "POLYGON ((9.5691 50.8971, 9.6050 50.8971, 9.6050 50.9198, 9.5691 50.9198, 9.5691 50.8971))",
            "geometry_type": "Polygon",
        },
    ]
    spatial_predicates = [
        {
            "id": "p1",
            "predicate": "DWITHIN",
            "target_ids": ["g1", "r1"],
            "distance": 250,
            "units": "meters",
            "join_with_next": "AND",
            "required": True,
        }
    ]

    ecql = build_spatial_ecql("the_geom", spatial_contexts, spatial_predicates)

    assert ecql is not None
    assert "DWITHIN(the_geom, SRID=4326;POINT (7.2 50.2), 250, meters)" in ecql
    assert "DWITHIN(the_geom, SRID=4326;POLYGON ((9.5691 50.8971" in ecql


def test_build_spatial_ecql_uses_or_join_operator() -> None:
    spatial_contexts = [
        {
            "target_id": "g1",
            "crs": "EPSG:4326",
            "bbox": [7.0, 50.0, 7.5, 50.5],
            "geometry_wkt": "POLYGON ((7.0 50.0, 7.5 50.0, 7.5 50.5, 7.0 50.5, 7.0 50.0))",
            "geometry_type": "Polygon",
        },
        {
            "target_id": "g2",
            "crs": "EPSG:4326",
            "bbox": [8.0, 51.0, 8.5, 51.5],
            "geometry_wkt": "POLYGON ((8.0 51.0, 8.5 51.0, 8.5 51.5, 8.0 51.5, 8.0 51.0))",
            "geometry_type": "Polygon",
        },
    ]
    spatial_predicates = [
        {
            "id": "p1",
            "predicate": "INTERSECTS",
            "target_ids": ["g1"],
            "join_with_next": "OR",
            "required": True,
        },
        {
            "id": "p2",
            "predicate": "INTERSECTS",
            "target_ids": ["g2"],
            "join_with_next": "AND",
            "required": True,
        },
    ]

    ecql = build_spatial_ecql("the_geom", spatial_contexts, spatial_predicates)

    assert ecql is not None
    assert " OR " in ecql


def test_build_spatial_ecql_skips_optional_unresolved_predicate() -> None:
    spatial_contexts = [
        {
            "target_id": "g1",
            "crs": "EPSG:4326",
            "bbox": [7.0, 50.0, 7.5, 50.5],
            "geometry_wkt": "POLYGON ((7.0 50.0, 7.5 50.0, 7.5 50.5, 7.0 50.5, 7.0 50.0))",
            "geometry_type": "Polygon",
        }
    ]
    spatial_predicates = [
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
            "target_ids": ["missing-target"],
            "distance": 100,
            "units": "meters",
            "join_with_next": "AND",
            "required": False,
        },
    ]

    ecql = build_spatial_ecql("the_geom", spatial_contexts, spatial_predicates)

    assert ecql is not None
    assert "INTERSECTS(the_geom, SRID=4326;POLYGON" in ecql
    assert "missing-target" not in ecql


def test_build_spatial_ecql_returns_none_for_required_unresolved_predicate() -> None:
    spatial_contexts = [
        {
            "target_id": "g1",
            "crs": "EPSG:4326",
            "bbox": [7.0, 50.0, 7.5, 50.5],
            "geometry_wkt": "POLYGON ((7.0 50.0, 7.5 50.0, 7.5 50.5, 7.0 50.5, 7.0 50.0))",
            "geometry_type": "Polygon",
        }
    ]
    spatial_predicates = [
        {
            "id": "p1",
            "predicate": "INTERSECTS",
            "target_ids": ["missing-target"],
            "join_with_next": "AND",
            "required": True,
        }
    ]

    assert build_spatial_ecql("the_geom", spatial_contexts, spatial_predicates) is None
