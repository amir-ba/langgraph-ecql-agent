from __future__ import annotations

from typing import Any
from shapely.geometry import box, shape
from shapely import wkt as shapely_wkt


def build_spatial_ecql(
    geom_column: str,
    spatial_context: dict[str, Any] | None,
    spatial_filter: dict[str, Any] | None,
) -> str | None:

    if not spatial_context:
        return None

    bbox = spatial_context.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    crs = str(spatial_context.get("crs") or "EPSG:4326")
    bbox_ecql = f"BBOX({geom_column}, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, '{crs}')"


    if not spatial_filter:
        return bbox_ecql

    predicate = str(spatial_filter.get("predicate") or "").upper()
    wkt = spatial_context.get("geometry_wkt")
    geometry_type = str(spatial_context.get("geometry_type") or "")

    if predicate in {"DWITHIN", "BEYOND"}:
        distance = spatial_filter.get("distance")
        units = spatial_filter.get("units")
        if isinstance(distance, int | float) and distance > 0 and isinstance(units, str) and isinstance(wkt, str):
            fine_ecql = f"{predicate}({geom_column}, {wkt}, {distance}, {units})"
            return f"({bbox_ecql}) AND ({fine_ecql})"
        return bbox_ecql


    topological = {
        "INTERSECTS",
        "WITHIN",
        "CONTAINS",
        "DISJOINT",
        "CROSSES",
        "OVERLAPS",
        "TOUCHES",
        "EQUALS"  

    }
    if predicate in topological and isinstance(wkt, str):
            # 1. BBox Redundancy Check: Only applies if the WKT is actually a polygon
        if geometry_type in {"Polygon", "MultiPolygon"}:
            try:
                bbox_poly = box(*bbox)
                wkt_geom = shapely_wkt.loads(wkt)
                if predicate == "WITHIN" and wkt_geom.equals(bbox_poly):
                    # Skip redundant WITHIN if WKT polygon is exactly the bbox
                    return bbox_ecql
            except Exception:
                pass
        
        # 2. Universal Topological Filter: Apply the predicate to ALL geometry types
        fine_ecql = f"{predicate}({geom_column}, {wkt})"
        return f"({bbox_ecql}) AND ({fine_ecql})"
    
    return bbox_ecql
