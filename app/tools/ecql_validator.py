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


def _is_scalar_literal(value: object) -> bool:
    return isinstance(value, str | int | float | bool)


def _is_self_comparison(node: object) -> bool:
    if not isinstance(
        node,
        ast.Equal | ast.NotEqual | ast.GreaterThan | ast.GreaterEqual | ast.LessThan | ast.LessEqual,
    ):
        return False

    return (
        isinstance(node.lhs, ast.Attribute)
        and isinstance(node.rhs, ast.Attribute)
        and node.lhs.name == node.rhs.name
    )


def _constant_truth_value(node: object) -> bool | None:
    # INCLUDE/EXCLUDE are explicit boolean constants in ECQL.
    if isinstance(node, ast.Include):
        return not node.not_

    if isinstance(node, ast.Not):
        sub_value = _constant_truth_value(node.sub_node)
        if sub_value is None:
            return None
        return not sub_value

    if isinstance(node, ast.And):
        lhs = _constant_truth_value(node.lhs)
        rhs = _constant_truth_value(node.rhs)
        if lhs is False or rhs is False:
            return False
        if lhs is True and rhs is True:
            return True
        return None

    if isinstance(node, ast.Or):
        lhs = _constant_truth_value(node.lhs)
        rhs = _constant_truth_value(node.rhs)
        if lhs is True or rhs is True:
            return True
        if lhs is False and rhs is False:
            return False
        return None

    if isinstance(
        node,
        ast.Equal | ast.NotEqual | ast.GreaterThan | ast.GreaterEqual | ast.LessThan | ast.LessEqual,
    ):
        if _is_self_comparison(node):
            if isinstance(node, ast.Equal | ast.GreaterEqual | ast.LessEqual):
                return True
            return False

        if _is_scalar_literal(node.lhs) and _is_scalar_literal(node.rhs):
            if isinstance(node, ast.Equal):
                return node.lhs == node.rhs
            if isinstance(node, ast.NotEqual):
                return node.lhs != node.rhs
            if isinstance(node, ast.GreaterThan):
                return node.lhs > node.rhs
            if isinstance(node, ast.GreaterEqual):
                return node.lhs >= node.rhs
            if isinstance(node, ast.LessThan):
                return node.lhs < node.rhs
            if isinstance(node, ast.LessEqual):
                return node.lhs <= node.rhs

    return None


def _has_attribute_reference(node: object) -> bool:
    return any(isinstance(candidate, ast.Attribute) for candidate in _iter_nodes(node))


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

    if not _has_attribute_reference(root):
        return False, (
            "ECQL must include at least one layer attribute or geometry predicate "
            "to constrain results"
        )

    truth_value = _constant_truth_value(root)
    if truth_value is True:
        return False, "ECQL is non-constraining (always true); provide a real filter predicate"
    if truth_value is False:
        return False, "ECQL is non-constraining (always false); provide a real filter predicate"

    return True, None
