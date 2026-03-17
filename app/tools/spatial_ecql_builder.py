from __future__ import annotations

from typing import Any
from shapely.geometry import box
from shapely import wkt as shapely_wkt


_BBoxSignature = tuple[float, float, float, float, str]


def _extract_srid(crs: str) -> int | None:
    normalized = str(crs).strip().upper()
    if normalized.startswith("EPSG:"):
        code = normalized.split(":", 1)[1].strip()
        if code.isdigit():
            return int(code)
    return None


def _strip_ewkt_prefix(wkt: str) -> str:
    stripped = wkt.strip()
    if stripped.upper().startswith("SRID=") and ";" in stripped:
        return stripped.split(";", 1)[1].strip()
    return stripped


def _ensure_ewkt(wkt: str, crs: str) -> str:
    stripped = wkt.strip()
    if stripped.upper().startswith("SRID="):
        return stripped
    srid = _extract_srid(crs)
    if srid is None:
        return stripped
    return f"SRID={srid};{stripped}"


def build_spatial_ecql(
    geom_column: str,
    spatial_contexts: list[dict[str, Any]] | None,
    spatial_filters: list[dict[str, Any] | None] | None,
) -> str | None:
    contexts = _normalize_contexts(spatial_contexts)
    if not contexts:
        return None

    filters = _normalize_filters(spatial_filters, len(contexts))
    clause_specs: list[tuple[str, bool, str, _BBoxSignature | None]] = []
    for context, spatial_filter_entry in zip(contexts, filters):
        clause, is_bbox_only = _build_single_spatial_ecql(geom_column, context, spatial_filter_entry)
        if clause:
            source = str(context.get("source") or "").strip().lower()
            clause_specs.append((clause, is_bbox_only, source, _bbox_signature_from_context(context)))

    clauses = _prune_clauses(clause_specs)

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return " AND ".join(f"({clause})" for clause in clauses)


def _prune_clauses(clause_specs: list[tuple[str, bool, str, _BBoxSignature | None]]) -> list[str]:
    # Deduplicate identical clauses while preserving order.
    unique_specs: list[tuple[str, bool, str, _BBoxSignature | None]] = []
    seen: set[str] = set()
    for clause, is_bbox_only, source, bbox_signature in clause_specs:
        if clause in seen:
            continue
        seen.add(clause)
        unique_specs.append((clause, is_bbox_only, source, bbox_signature))

    covered_bbox_signatures = {
        bbox_signature
        for _, is_bbox_only, _, bbox_signature in unique_specs
        if not is_bbox_only and bbox_signature is not None
    }

    filtered_specs: list[tuple[str, bool, str, _BBoxSignature | None]] = []
    for clause, is_bbox_only, source, bbox_signature in unique_specs:
        if is_bbox_only and bbox_signature is not None and bbox_signature in covered_bbox_signatures:
            continue
        filtered_specs.append((clause, is_bbox_only, source, bbox_signature))

    bbox_only_specs = [spec for spec in filtered_specs if spec[1]]
    if len(bbox_only_specs) <= 1:
        return [clause for clause, _, _, _ in filtered_specs]

    # If explicit bbox contexts exist, keep them and drop reference-derived bbox-only noise.
    preferred_bbox_clauses = {
        clause
        for clause, _, source, _ in bbox_only_specs
        if source == "explicit_bbox"
    }
    if not preferred_bbox_clauses:
        preferred_bbox_clauses = {
            clause
            for clause, _, source, _ in bbox_only_specs
            if source.startswith("explicit_")
        }

    if not preferred_bbox_clauses:
        # Fall back to the first bbox-only clause to avoid over-constraining by noisy geocoding.
        preferred_bbox_clauses = {bbox_only_specs[0][0]}

    pruned: list[str] = []
    for clause, is_bbox_only, _, _ in filtered_specs:
        if is_bbox_only and clause not in preferred_bbox_clauses:
            continue
        pruned.append(clause)
    return pruned


def _bbox_signature_from_context(spatial_context: dict[str, Any]) -> _BBoxSignature | None:
    """
    Extracts a bbox signature from a spatial context dict.
    Used for deduplication in spatial_contexts array.
    """
    bbox = spatial_context.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    if not all(isinstance(value, int | float) for value in bbox):
        return None

    crs = str(spatial_context.get("crs") or "EPSG:4326").strip().upper()
    return (
        round(float(bbox[0]), 12),
        round(float(bbox[1]), 12),
        round(float(bbox[2]), 12),
        round(float(bbox[3]), 12),
        crs,
    )


def _normalize_contexts(
    spatial_contexts: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """
    Normalize spatial_contexts to ensure all are dicts.
    Accepts a list of spatial context dicts or None.
    Returns a filtered list of dicts.
    """
    if spatial_contexts is None:
        return []
    return [context for context in spatial_contexts if isinstance(context, dict)]


def _normalize_filters(
    spatial_filters: list[dict[str, Any] | None] | None,
    count: int,
) -> list[dict[str, Any] | None]:
    if count <= 0:
        return []
    if spatial_filters is None:
        return [None for _ in range(count)]
    filters: list[dict[str, Any] | None] = [
        entry if isinstance(entry, dict) else None for entry in spatial_filters
    ]
    if len(filters) < count:
        filters.extend([None for _ in range(count - len(filters))])
    return filters[:count]


def _build_single_spatial_ecql(
    geom_column: str,
    spatial_context: dict[str, Any],
    spatial_filter: dict[str, Any] | None,
) -> tuple[str | None, bool]:
    """
    Builds a single ECQL clause from a spatial context dict and filter.
    Used within spatial_contexts array processing.
    """
    bbox = spatial_context.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None, False

    crs = str(spatial_context.get("crs") or "EPSG:4326")
    bbox_ecql = f"BBOX({geom_column}, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, '{crs}')"

    source = str(spatial_context.get("source") or "").strip().lower()
    if source == "fallback_germany_bbox":
        return bbox_ecql, True

    if not spatial_filter:
        return bbox_ecql, True

    predicate = str(spatial_filter.get("predicate") or "").upper()
    wkt = spatial_context.get("geometry_wkt")
    geometry_type = str(spatial_context.get("geometry_type") or "")
    ewkt = _ensure_ewkt(wkt, crs) if isinstance(wkt, str) else None
    raw_wkt = _strip_ewkt_prefix(ewkt) if isinstance(ewkt, str) else None

    if predicate in {"DWITHIN", "BEYOND"}:
        distance = spatial_filter.get("distance")
        units = spatial_filter.get("units")
        if isinstance(distance, int | float) and distance > 0 and isinstance(units, str) and isinstance(ewkt, str):
            fine_ecql = f"{predicate}({geom_column}, {ewkt}, {distance}, {units})"
            if geometry_type.strip().lower() == "point" or (isinstance(raw_wkt, str) and raw_wkt.upper().startswith("POINT")):
                return fine_ecql, False
            return f"({bbox_ecql}) AND ({fine_ecql})", False
        return bbox_ecql, True

    topological = {
        "INTERSECTS",
        "WITHIN",
        "CONTAINS",
        "DISJOINT",
        "CROSSES",
        "OVERLAPS",
        "TOUCHES",
        "EQUALS",
    }
    if predicate in topological and isinstance(ewkt, str):
        fine_ecql = f"{predicate}({geom_column}, {ewkt})"

        # Point geometries do not benefit from an additional bbox prefilter.
        if geometry_type.strip().lower() == "point" or (isinstance(raw_wkt, str) and raw_wkt.upper().startswith("POINT")):
            return fine_ecql, False

        # Avoid duplicating BBOX when the topological geometry already matches the bbox envelope.
        if geometry_type in {"Polygon", "MultiPolygon"}:
            try:
                bbox_poly = box(*bbox)
                if not isinstance(raw_wkt, str):
                    return f"({bbox_ecql}) AND ({fine_ecql})", False
                wkt_geom = shapely_wkt.loads(raw_wkt)
                if wkt_geom.equals(bbox_poly):
                    return fine_ecql, False
            except Exception:
                pass

        # Apply predicate with bbox prefilter for non-envelope-equal cases.
        return f"({bbox_ecql}) AND ({fine_ecql})", False

    return bbox_ecql, True
