from collections.abc import Iterator

from pygeofilter import ast
from pygeofilter.parsers.ecql import parse


def _iter_nodes(node: object) -> Iterator[object]:
    yield node

    if isinstance(node, list | tuple | set):
        for item in node:
            yield from _iter_nodes(item)
        return

    if isinstance(node, dict):
        for item in node.values():
            yield from _iter_nodes(item)
        return

    values = getattr(node, "__dict__", None)
    if not values:
        return

    for value in values.values():
        yield from _iter_nodes(value)


def validate_ecql(ecql_string: str, layer_schema: dict[str, str], geometry_column: str) -> tuple[bool, str | None]:
    try:
        root = parse(ecql_string)
    except Exception as exc:  # pragma: no cover - parser error types vary by expression
        return False, f"ECQL syntax error: {exc}"

    allowed_attributes = set(layer_schema)
    allowed_attributes.add(geometry_column)

    for node in _iter_nodes(root):
        if isinstance(node, ast.Attribute) and node.name not in allowed_attributes:
            return False, f"Unknown attribute in ECQL: {node.name}"

        if isinstance(node, ast.BBox | ast.GeometryIntersects):
            lhs = node.lhs
            if not isinstance(lhs, ast.Attribute) or lhs.name != geometry_column:
                return False, (
                    f"Spatial predicate must use geometry column '{geometry_column}', "
                    f"got '{getattr(lhs, 'name', lhs)}'"
                )

    return True, None
