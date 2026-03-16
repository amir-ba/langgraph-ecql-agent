import asyncio

from app.graph.nodes import ecql_validation_node


def test_ecql_validation_node_sets_validation_error_from_validator() -> None:
    state = {
        "generated_ecql": "INTERSECTS(geom, POINT(7 50))",
        "layer_schema": {"STATE_NAME": "xsd:string", "PERSONS": "xsd:int"},
        "geometry_column": "the_geom",
    }

    updates = asyncio.run(ecql_validation_node(state))

    assert updates["validation_error"] is not None
    assert "geometry column" in updates["validation_error"].lower()


def test_ecql_validation_node_rejects_non_constraining_expression() -> None:
    state = {
        "generated_ecql": "INCLUDE",
        "layer_schema": {"STATE_NAME": "xsd:string", "PERSONS": "xsd:int"},
        "geometry_column": "the_geom",
    }

    updates = asyncio.run(ecql_validation_node(state))

    assert updates["validation_error"] is not None
    assert "constrain" in updates["validation_error"].lower()
