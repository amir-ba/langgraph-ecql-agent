from app.tools.spatial_ecql_builder import build_spatial_ecql


def test_build_spatial_ecql_returns_none_without_context() -> None:
    assert build_spatial_ecql("the_geom", None, None) is None


def test_build_spatial_ecql_builds_bbox_only_without_filter() -> None:
    spatial_contexts = [{
        "crs": "EPSG:4326",
        "bbox": [13.1, 52.3, 13.8, 52.7],
        "geometry_wkt": "POINT (13.45 52.5)",
        "geometry_type": "Point",
    }]

    ecql = build_spatial_ecql("the_geom", spatial_contexts, None)

    assert ecql == "BBOX(the_geom, 13.1, 52.3, 13.8, 52.7, 'EPSG:4326')"


def test_build_spatial_ecql_builds_distance_predicate_for_dwithin() -> None:
    spatial_contexts = [{
        "crs": "EPSG:4326",
        "bbox": [13.1, 52.3, 13.8, 52.7],
        "geometry_wkt": "POINT (13.45 52.5)",
        "geometry_type": "Point",
    }]
    spatial_filters = [
        {
        "predicate": "DWITHIN",
        "distance": 5,
        "units": "kilometers",
        }
    ]

    ecql = build_spatial_ecql("the_geom", spatial_contexts, spatial_filters)

    assert ecql is not None
    assert "BBOX(the_geom, 13.1, 52.3, 13.8, 52.7, 'EPSG:4326')" not in ecql
    assert "DWITHIN(the_geom, SRID=4326;POINT (13.45 52.5), 5, kilometers)" in ecql


def test_build_spatial_ecql_uses_polygon_for_topological_predicate() -> None:
    spatial_contexts = [{
        "crs": "EPSG:4326",
        "bbox": [13.1, 52.3, 13.8, 52.7],
        "geometry_wkt": "POLYGON ((13.1 52.3, 13.8 52.3, 13.8 52.7, 13.1 52.7, 13.1 52.3))",
        "geometry_type": "Polygon",
    }]
    spatial_filters = [{"predicate": "INTERSECTS"}]

    ecql = build_spatial_ecql("the_geom", spatial_contexts, spatial_filters)

    assert ecql is not None
    assert "BBOX(the_geom, 13.1, 52.3, 13.8, 52.7, 'EPSG:4326')" not in ecql
    assert ecql == "INTERSECTS(the_geom, SRID=4326;POLYGON ((13.1 52.3, 13.8 52.3, 13.8 52.7, 13.1 52.7, 13.1 52.3)))"


def test_build_spatial_ecql_uses_point_only_for_topological_predicate() -> None:
    spatial_contexts = [{
        "source": "explicit_point",
        "crs": "EPSG:4326",
        "bbox": [7.9999, 49.9999, 8.0001, 50.0001],
        "geometry_wkt": "POINT(8.0 50.0)",
        "geometry_type": "Point",
    }]
    spatial_filters = [{"predicate": "INTERSECTS"}]

    ecql = build_spatial_ecql("wkb_geometry", spatial_contexts, spatial_filters)

    assert ecql == "INTERSECTS(wkb_geometry, SRID=4326;POINT(8.0 50.0))"


def test_build_spatial_ecql_falls_back_to_bbox_when_distance_missing() -> None:
    spatial_contexts = [{
        "crs": "EPSG:4326",
        "bbox": [13.1, 52.3, 13.8, 52.7],
        "geometry_wkt": "POINT (13.45 52.5)",
        "geometry_type": "Point",
    }]
    spatial_filters = [{"predicate": "DWITHIN", "units": "kilometers"}]

    ecql = build_spatial_ecql("the_geom", spatial_contexts, spatial_filters)

    assert ecql == "BBOX(the_geom, 13.1, 52.3, 13.8, 52.7, 'EPSG:4326')"


def test_build_spatial_ecql_builds_multi_predicates_with_and() -> None:
    spatial_contexts = [
        {
            "crs": "EPSG:4326",
            "bbox": [7.0, 50.0, 7.5, 50.5],
            "geometry_wkt": (
                "POLYGON ((7.0 50.0, 7.5 50.0, 7.5 50.5, 7.0 50.5, 7.0 50.0))"
            ),
            "geometry_type": "Polygon",
        },
        {
            "crs": "EPSG:4326",
            "bbox": [7.19, 50.19, 7.21, 50.21],
            "geometry_wkt": "POINT (7.2 50.2)",
            "geometry_type": "Point",
        },
    ]
    spatial_filters = [
        None,
        {
            "predicate": "DWITHIN",
            "distance": 100,
            "units": "meters",
        },
    ]

    ecql = build_spatial_ecql("the_geom", spatial_contexts, spatial_filters)

    assert ecql is not None
    assert " AND " in ecql
    assert "BBOX(the_geom, 7.0, 50.0, 7.5, 50.5, 'EPSG:4326')" in ecql
    assert "BBOX(the_geom, 7.19, 50.19, 7.21, 50.21, 'EPSG:4326')" not in ecql
    assert "DWITHIN(the_geom, SRID=4326;POINT (7.2 50.2), 100, meters)" in ecql


def test_build_spatial_ecql_prefers_explicit_bbox_over_reference_bbox_only_clause() -> None:
    spatial_contexts = [
        {
            "source": "explicit_bbox",
            "crs": "EPSG:4326",
            "bbox": [7.0, 50.0, 7.5, 50.5],
            "geometry_wkt": (
                "POLYGON ((7.0 50.0, 7.5 50.0, 7.5 50.5, 7.0 50.5, 7.0 50.0))"
            ),
            "geometry_type": "Polygon",
        },
        {
            "source": "explicit_point",
            "crs": "EPSG:4326",
            "bbox": [7.1991, 50.1994, 7.2009, 50.2006],
            "geometry_wkt": "POINT (7.2 50.2)",
            "geometry_type": "Point",
        },
        {
            "source": "reference",
            "crs": "EPSG:4326",
            "bbox": [9.5691, 50.8971, 9.6050, 50.9198],
            "geometry_wkt": (
                "POLYGON ((9.5691 50.8971, 9.6050 50.8971, 9.6050 50.9198, "
                "9.5691 50.9198, 9.5691 50.8971))"
            ),
            "geometry_type": "Polygon",
        },
    ]
    spatial_filters = [
        None,
        {
            "predicate": "DWITHIN",
            "distance": 100,
            "units": "meters",
        },
        None,
    ]

    ecql = build_spatial_ecql("the_geom", spatial_contexts, spatial_filters)

    assert ecql is not None
    assert "BBOX(the_geom, 7.0, 50.0, 7.5, 50.5, 'EPSG:4326')" in ecql
    assert "DWITHIN(the_geom, SRID=4326;POINT (7.2 50.2), 100, meters)" in ecql
    assert "BBOX(the_geom, 9.5691, 50.8971, 9.605, 50.9198, 'EPSG:4326')" not in ecql


def test_build_spatial_ecql_drops_bbox_only_when_stronger_clause_has_same_bbox() -> None:
    spatial_contexts = [
        {
            "source": "reference",
            "crs": "EPSG:4326",
            "bbox": [9.621463608102793, 53.303866335746676, 10.519778892222314, 53.837305712419166],
            "geometry_wkt": (
                "POLYGON ((10.519778892222314 53.303866335746676, "
                "10.519778892222314 53.837305712419166, "
                "9.621463608102793 53.837305712419166, "
                "9.621463608102793 53.303866335746676, "
                "10.519778892222314 53.303866335746676))"
            ),
            "geometry_type": "Polygon",
        },
        {
            "source": "reference",
            "crs": "EPSG:4326",
            "bbox": [9.621463608102793, 53.303866335746676, 10.519778892222314, 53.837305712419166],
            "geometry_wkt": (
                "POLYGON ((10.519778892222314 53.303866335746676, "
                "10.519778892222314 53.837305712419166, "
                "9.621463608102793 53.837305712419166, "
                "9.621463608102793 53.303866335746676, "
                "10.519778892222314 53.303866335746676))"
            ),
            "geometry_type": "Polygon",
        },
    ]
    spatial_filters = [
        None,
        {"predicate": "INTERSECTS"},
    ]

    ecql = build_spatial_ecql("wkb_geometry", spatial_contexts, spatial_filters)

    assert ecql is not None
    assert "BBOX(wkb_geometry" not in ecql
    assert "INTERSECTS(wkb_geometry, SRID=4326;POLYGON" in ecql


def test_build_spatial_ecql_uses_bbox_only_for_fallback_germany_context() -> None:
    spatial_contexts = [
        {
            "source": "fallback_germany_bbox",
            "crs": "EPSG:4326",
            "bbox": [5.86632, 47.27011, 15.04193, 55.09916],
            "geometry_wkt": (
                "POLYGON ((5.86632 47.27011, 15.04193 47.27011, "
                "15.04193 55.09916, 5.86632 55.09916, 5.86632 47.27011))"
            ),
            "geometry_type": "Polygon",
        }
    ]
    spatial_filters = [{"predicate": "INTERSECTS"}]

    ecql = build_spatial_ecql("the_geom", spatial_contexts, spatial_filters)

    assert ecql == "BBOX(the_geom, 5.86632, 47.27011, 15.04193, 55.09916, 'EPSG:4326')"
