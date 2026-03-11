from app.tools.ecql_validator import validate_ecql


def test_validate_ecql_returns_error_for_invalid_syntax() -> None:
    is_valid, error = validate_ecql(
        ecql_string="BBOX(the_geom, -10, -10, 10, 10",
        layer_schema={"STATE_NAME": "xsd:string", "PERSONS": "xsd:int"},
        geometry_column="the_geom",
    )

    assert is_valid is False
    assert error is not None
    assert "syntax" in error.lower()


def test_validate_ecql_returns_error_for_unknown_schema_attribute() -> None:
    is_valid, error = validate_ecql(
        ecql_string="FAKE_DATA > 1000",
        layer_schema={"STATE_NAME": "xsd:string", "PERSONS": "xsd:int"},
        geometry_column="the_geom",
    )

    assert is_valid is False
    assert error is not None
    assert "unknown attribute" in error.lower()
    assert "FAKE_DATA" in error


def test_validate_ecql_returns_error_when_spatial_predicate_uses_wrong_geometry_column() -> None:
    is_valid, error = validate_ecql(
        ecql_string="INTERSECTS(geom, POINT(7 50))",
        layer_schema={"STATE_NAME": "xsd:string", "PERSONS": "xsd:int"},
        geometry_column="the_geom",
    )

    assert is_valid is False
    assert error is not None
    assert "geometry column" in error.lower()
    assert "the_geom" in error


def test_validate_ecql_accepts_valid_spatial_filter() -> None:
    is_valid, error = validate_ecql(
        ecql_string="BBOX(the_geom, -10, -10, 10, 10) AND PERSONS > 1000",
        layer_schema={"STATE_NAME": "xsd:string", "PERSONS": "xsd:int"},
        geometry_column="the_geom",
    )

    assert is_valid is True
    assert error is None
