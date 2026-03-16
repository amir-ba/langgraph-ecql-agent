from app.tools.spatial_ecql_builder import build_spatial_ecql


def test_build_spatial_ecql_returns_none_without_context() -> None:
    assert build_spatial_ecql("the_geom", None, None) is None


def test_build_spatial_ecql_builds_bbox_only_without_filter() -> None:
    spatial_context = {
        "crs": "EPSG:4326",
        "bbox": [13.1, 52.3, 13.8, 52.7],
        "geometry_wkt": "POINT (13.45 52.5)",
        "geometry_type": "Point",
    }

    ecql = build_spatial_ecql("the_geom", spatial_context, None)

    assert ecql == "BBOX(the_geom, 13.1, 52.3, 13.8, 52.7, 'EPSG:4326')"


def test_build_spatial_ecql_builds_distance_predicate_for_dwithin() -> None:
    spatial_context = {
        "crs": "EPSG:4326",
        "bbox": [13.1, 52.3, 13.8, 52.7],
        "geometry_wkt": "POINT (13.45 52.5)",
        "geometry_type": "Point",
    }
    spatial_filter = {
        "predicate": "DWITHIN",
        "distance": 5,
        "units": "kilometers",
    }

    ecql = build_spatial_ecql("the_geom", spatial_context, spatial_filter)

    assert ecql is not None
    assert "BBOX(the_geom, 13.1, 52.3, 13.8, 52.7, 'EPSG:4326')" in ecql
    assert "DWITHIN(the_geom, POINT (13.45 52.5), 5, kilometers)" in ecql


def test_build_spatial_ecql_uses_polygon_for_topological_predicate() -> None:
    spatial_context = {
        "crs": "EPSG:4326",
        "bbox": [13.1, 52.3, 13.8, 52.7],
        "geometry_wkt": "POLYGON ((13.1 52.3, 13.8 52.3, 13.8 52.7, 13.1 52.7, 13.1 52.3))",
        "geometry_type": "Polygon",
    }
    spatial_filter = {"predicate": "INTERSECTS"}

    ecql = build_spatial_ecql("the_geom", spatial_context, spatial_filter)

    assert ecql is not None
    assert "BBOX(the_geom, 13.1, 52.3, 13.8, 52.7, 'EPSG:4326')" in ecql
    assert "INTERSECTS(the_geom, POLYGON ((13.1 52.3, 13.8 52.3, 13.8 52.7, 13.1 52.7, 13.1 52.3)))" in ecql


def test_build_spatial_ecql_falls_back_to_bbox_when_distance_missing() -> None:
    spatial_context = {
        "crs": "EPSG:4326",
        "bbox": [13.1, 52.3, 13.8, 52.7],
        "geometry_wkt": "POINT (13.45 52.5)",
        "geometry_type": "Point",
    }
    spatial_filter = {"predicate": "DWITHIN", "units": "kilometers"}

    ecql = build_spatial_ecql("the_geom", spatial_context, spatial_filter)

    assert ecql == "BBOX(the_geom, 13.1, 52.3, 13.8, 52.7, 'EPSG:4326')"
