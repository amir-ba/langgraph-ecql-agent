from __future__ import annotations

import json
import logging
from typing import Any, cast

import httpx
from pydantic import BaseModel, Field
from pyproj import Transformer
from shapely.geometry import Point, box

from app.core.llm import ainvoke_llm
from app.core.schemas import AnalyzedIntent, ECQLGeneration
from app.core.settings import get_settings
from app.graph.state import AgentState, FinalResponsePayload
from app.tools.ecql_validator import validate_ecql
from app.tools.geocoder import GeocoderClient
from app.tools.spatial_ecql_builder import build_spatial_ecql
from app.tools.wfs_client import (
    discover_layers,
    execute_wfs_query,
    filter_layers_by_subject,
    get_layer_schema,
)
from app.tools.layer_catalog import (
    ensure_markdown_layer_catalog,
    render_basic_markdown_catalog,
)


logger = logging.getLogger(__name__)


class LayerSelection(BaseModel):
    layer_name: str = Field(description="The selected WFS layer name")
    confidence: str = Field(
        default="high",
        description="Confidence in the selected layer: high, medium, or low",
    )


def _as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, default=str)


def _build_final_response(summary: str) -> FinalResponsePayload:
    return {
        "summary": summary,
    }


def _distance_to_meters(distance: float, units: str) -> float | None:
    factor = {
        "meters": 1.0,
        "kilometers": 1000.0,
        "feet": 0.3048,
        "statute miles": 1609.344,
        "nautical miles": 1852.0,
    }.get(units.lower())
    if factor is None:
        return None
    return distance * factor


def _normalize_bbox(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) == 4:
        try:
            return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except (TypeError, ValueError):
            return None
    return None


def _normalize_geojson_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    return None


async def unified_router_analyzer_node(state: AgentState) -> dict[str, Any]:
    user_query = state.get("user_query", "")
    logger.debug("[unified_router_analyzer_node] input user_query=%s", user_query)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a geospatial AI assistant. Determine if the user is asking "
                "for spatial data or just chatting. If spatial, extract the location, "
                "target features (layer subject), spatial relationship, and attributes. "
                "Current location is Germany. Current year is 2026."
            ),
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    result = cast(
        AnalyzedIntent,
        await ainvoke_llm(messages=messages, output_schema=AnalyzedIntent, agent_state=state),
    )
    final_payload: FinalResponsePayload | None = None
    if result.intent == "irrelevant":
        summary = (result.general_response or "").strip()
        if not summary:
            summary = "I can help with spatial queries against available map layers."
        final_payload = _build_final_response(summary=summary)

    output = {
        "intent": result.intent,
        "final_response": final_payload,
        "spatial_reference": result.spatial_reference,
        "explicit_coordinates": result.explicit_coordinates,
        "explicit_bbox": _normalize_bbox(result.explicit_bbox),
        "spatial_filter": (
            result.spatial_filter.model_dump(exclude_none=True)
            if result.spatial_filter is not None
            else None
        ),
        "layer_subject": result.layer_subject,
        "attribute_hints": result.attribute_hints or [],
    }
    logger.debug("[unified_router_analyzer_node] output=%s", _as_json(output))
    return output


async def geocoder_context_node(state: AgentState) -> dict[str, Any]:
    query = state.get("spatial_reference") or ""

    logger.debug("[geocoder_context_node] input query=%s", query)

    # 1. BYPASS 1: Explicit Bounding Box
    explicit_bbox = state.get("explicit_bbox")
    if isinstance(explicit_bbox, list) and len(explicit_bbox) == 4:
        min_x, min_y, max_x, max_y = explicit_bbox
        
        # Construct a WKT Polygon out of the 4 BBOX corners
        # Note: WKT Polygons must be closed (first point == last point)
        wkt_polygon = (
            f"POLYGON(({min_x} {min_y}, {min_x} {max_y}, "
            f"{max_x} {max_y}, {max_x} {min_y}, {min_x} {min_y}))"
        )
        
        output = {
            "spatial_context": {
                "crs": "EPSG:4326",
                "bbox": [min_x, min_y, max_x, max_y],
                "geometry_wkt": wkt_polygon,
                "geometry_type": "Polygon"
            }
        }
        return output

    # 2. BYPASS 2: Explicit Point Coordinates (from previous step)
    explicit_coordinates = state.get("explicit_coordinates")
    if isinstance(explicit_coordinates, list) and len(explicit_coordinates) == 2:
        lon, lat = explicit_coordinates
        
        # Create a Micro-Buffer (~11 meters) to make a valid BBOX
        epsilon = 0.0001
        bbox =[lon - epsilon, lat - epsilon, lon + epsilon, lat + epsilon]
        
        output = {
            "spatial_context": {
                "crs": "EPSG:4326",
                "bbox": bbox,
                "geometry_wkt": f"POINT({lon} {lat})",
                "geometry_type": "Point"
            }
        }
        return output
    if not query:
        output = {"spatial_context": None}
        logger.debug("[geocoder_context_node] output=%s", _as_json(output))
        return output
    geocoder_http_client = cast(httpx.AsyncClient | None, state.get("geocoder_http_client"))
    geocoder = GeocoderClient(http_client=geocoder_http_client)

    # 1. Use /suggest endpoint to get locationType
    suggest_payload = await geocoder.suggest(query=query, max_results=1)
    suggest_results = suggest_payload.get("SuggestResult") or []
    location_type = None
    if suggest_results:
        location_type = suggest_results[0].get("locationType")

    # 2. Use fulltext search to get coordinates
    payload = await geocoder.forward_fulltext(query=query, max_results=1, epsg=4326)
    results = payload.get("Result") or []
    first_result = results[0] if results else {}

    

    # Otherwise, try to use Coordinate and buffer logic
    coord_str = first_result.get("Coordinate")
    if not coord_str:
        output = {"spatial_context": None}
        logger.debug("[geocoder_context_node] output=%s", _as_json(output))
        return output

    # Parse lat,lon
    try:
        lat_str, lon_str = coord_str.split(",")
        lat, lon = float(lat_str.strip()), float(lon_str.strip())
    except Exception:  # pragma: no cover - defensive parsing for provider payload differences
        output = {"spatial_context": None}
        logger.debug("[geocoder_context_node] output=%s", _as_json(output))
        return output

    # 1. Explicit user intent from structured predicate (DWITHIN/BEYOND)
    buffer_m = None
    spatial_filter = state.get("spatial_filter")
    if isinstance(spatial_filter, dict):
        predicate = str(spatial_filter.get("predicate") or "").upper()
        distance = spatial_filter.get("distance")
        units = spatial_filter.get("units")
        if predicate in {"DWITHIN", "BEYOND"} and isinstance(distance, int | float) and isinstance(units, str):
            buffer_m = _distance_to_meters(float(distance), units)

    # 2. Heuristic by locationType from /suggest
    if buffer_m is None:
        locationtype_to_buffer = {
            "Address": 500,
            "Street": 2000,
            "Zip": 20000,
            "City": 50000,
            "POI": 1000,
            "Unknown": 5000,
        }
        if location_type in locationtype_to_buffer:
            buffer_m = locationtype_to_buffer[location_type]

    # 3. Fallback default
    if buffer_m is None:
        buffer_m = 5000
    logger.info("[geocoder_context_node] buffer_m=%s", buffer_m)

    # Buffer in metric CRS and convert back to EPSG:4326 for ECQL BBOX with explicit CRS.
    transformer_to_3857 = Transformer.from_crs(4326, 3857, always_xy=True)
    transformer_to_4326 = Transformer.from_crs(3857, 4326, always_xy=True)
    x, y = transformer_to_3857.transform(lon, lat)

    minx = x - buffer_m
    maxx = x + buffer_m
    miny = y - buffer_m
    maxy = y + buffer_m

    llon, llat = transformer_to_4326.transform(min(minx, maxx), min(miny, maxy))
    ulon, ulat = transformer_to_4326.transform(max(minx, maxx), max(miny, maxy))
    bbox_4326 = [llon, llat, ulon, ulat]
    # Determine geometry type and WKT based on location_type
    if location_type in {"POI", "Address", "Unknown"}:
        geometry_wkt = Point(lon, lat).wkt
        geometry_type = "Point"
    else:
        # Create a polygon from the bbox
        polygon = box(*bbox_4326)
        geometry_wkt = polygon.wkt
        geometry_type = "Polygon"
    output = {
        "spatial_context": {
            "crs": "EPSG:4326",
            "bbox": bbox_4326,
            "geometry_wkt": geometry_wkt,
            "geometry_type": geometry_type,
        }
    }
    logger.debug("[geocoder_context_node] output=%s", _as_json(output))
    return output


async def wfs_discovery_node(state: AgentState) -> dict[str, list[dict[str, str]] | str | None]:
    logger.debug("[wfs_discovery_node] input received")
    settings = get_settings()
    wfs_http_client = cast(httpx.AsyncClient | None, state.get("wfs_http_client"))
    layers = await discover_layers(
        wfs_url=settings.geoserver_wfs_url,
        http_client=wfs_http_client,
        username=settings.geoserver_wfs_username,
        password=settings.geoserver_wfs_password,
    )
    if not layers:
        output = {
            "available_layers": [],
            "validation_error": "No WFS layers discovered from GetCapabilities",
        }
        logger.debug("[wfs_discovery_node] output=%s", _as_json(output))
        return output

    catalog_markdown = ""
    try:
        catalog_markdown = await ensure_markdown_layer_catalog(
            layers=layers,
            catalog_path=getattr(settings, "layer_catalog_markdown_path", "layer_catalog.md"),
            stale_after_hours=getattr(settings, "layer_catalog_stale_after_hours", 8),
        )
    except Exception as exc:
        logger.warning("[wfs_discovery_node] markdown catalog refresh failed: %s", exc)
        catalog_markdown = render_basic_markdown_catalog(layers)

    output = {
        "available_layers": layers,
        "layer_catalog_markdown": catalog_markdown,
        "validation_error": None,
    }
    logger.debug("[wfs_discovery_node] output=%s", _as_json(output))
    return output


async def layer_discoverer_node(state: AgentState) -> dict[str, Any]:
    available_layers = state.get("available_layers", [])
    layer_subject = state.get("layer_subject")
    logger.debug(
        "[layer_discoverer_node] input user_query=%s layer_subject=%s available_layers_count=%s",
        state.get("user_query", ""),
        layer_subject,
        len(available_layers) if isinstance(available_layers, list) else 0,
    )
    if not available_layers:
        output = {
            "selected_layer": "",
            "validation_error": "No WFS layers available for selection",
        }
        logger.debug("[layer_discoverer_node] output=%s", _as_json(output))
        return output

    catalog_markdown = str(state.get("layer_catalog_markdown") or "")
    if not catalog_markdown:
        catalog_markdown = render_basic_markdown_catalog(available_layers)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert GIS layer selector. "
                "Choose exactly one layer_name from the markdown catalog. "
                "Do not invent layer names. "
                "Set confidence to low if no catalog entry clearly matches the request."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query:\n{state.get('user_query', '')}\n\n"
                f"Layer subject:\n{layer_subject or ''}\n\n"
                f"Layer catalog markdown:\n{catalog_markdown}\n\n"
                f"Available layer names:\n{_as_json([str(layer.get('name', '')) for layer in available_layers])}"
            ),
        },
    ]

    result = cast(
        LayerSelection,
        await ainvoke_llm(messages=messages, response_format=LayerSelection, agent_state=state),
    )

    valid_layer_names = {
        str(layer.get("name", ""))
        for layer in available_layers
        if str(layer.get("name", "")).strip()
    }
    chosen_layer = str(result.layer_name or "").strip()
    confidence = str(result.confidence or "").strip().lower()

    if (not chosen_layer) or (chosen_layer not in valid_layer_names) or confidence == "low":
        output = {
            "selected_layer": "",
            "validation_error": (
                "Low-confidence layer retrieval: unable to confidently map "
                "layer_subject to available WFS layers"
            ),
            "final_response": _build_final_response(
                summary=(
                    "I could not confidently determine which map layer matches your request. "
                    "Please mention a clearer feature type (for example, hospitals, roads, or schools)."
                )
            ),
            "retrieval_mode": "catalog",
            "retrieval_top_score": None,
            "retrieval_score_gap": None,
            "candidate_layers_for_llm_count": len(available_layers),
        }
        logger.debug("[layer_discoverer_node] output=%s", _as_json(output))
        return output

    output = {
        "selected_layer": chosen_layer,
        "validation_error": None,
        "retrieval_mode": "catalog",
        "retrieval_top_score": None,
        "retrieval_score_gap": None,
        "candidate_layers_for_llm_count": len(available_layers),
    }
    logger.debug("[layer_discoverer_node] output=%s", _as_json(output))
    return output


async def schema_extractor_node(state: AgentState) -> dict[str, Any]:
    selected_layer = state.get("selected_layer", "")
    logger.debug("[schema_extractor_node] input selected_layer=%s", selected_layer)
    if not selected_layer:
        output = {
            "layer_schema": {},
            "geometry_column": "",
            "validation_error": "No selected_layer provided for schema extraction",
        }
        logger.debug("[schema_extractor_node] output=%s", _as_json(output))
        return output

    settings = get_settings()
    wfs_http_client = cast(httpx.AsyncClient | None, state.get("wfs_http_client"))
    attributes, geometry_column = await get_layer_schema(
        wfs_url=settings.geoserver_wfs_url,
        type_name=selected_layer,
        http_client=wfs_http_client,
        username=settings.geoserver_wfs_username,
        password=settings.geoserver_wfs_password,
    )

    output = {
        "layer_schema": attributes,
        "geometry_column": geometry_column,
        "validation_error": None,
    }
    logger.debug("[schema_extractor_node] output=%s", _as_json(output))
    return output


async def ecql_generator_node(state: AgentState) -> dict[str, str | int]:
    logger.debug(
        "[ecql_generator_node] input user_query=%s retry_count=%s",
        state.get("user_query", ""),
        state.get("retry_count", 0),
    )
    spatial_ecql = build_spatial_ecql(
        geom_column=state.get("geometry_column", ""),
        spatial_context=cast(dict[str, Any] | None, state.get("spatial_context")),
        spatial_filter=cast(dict[str, Any] | None, state.get("spatial_filter")),
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in GeoServer ECQL. Your task is to generate ONLY the "
                "ATTRIBUTE filtering portion of the ECQL expression based on the user's hints. "
                "DO NOT generate spatial filters (like BBOX, DWITHIN, or INTERSECTS). "
                "Use only the provided schema attributes. "
                "If no attribute filtering is requested or needed, return the exact string 'NONE'. "
                "ECQL String Matching Rules:\n"
                "- For \"starts with\": Use the LIKE operator with a wildcard at the end: name LIKE 'Nordrhein%'\n"
                "- For \"contains\": Use wildcards on both ends: name LIKE '%Nordrhein%'\n"
                "- For exact match: Use equals: name = 'Nordrhein'\n"
                "- Strings MUST be wrapped in single quotes.\n"
                "Do not return trivial filters like INCLUDE or 1=1."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query:\n{state.get('user_query', '')}\n\n"
                f"Layer schema:\n{_as_json(state.get('layer_schema', {}))}\n\n"
                f"Attribute hints:\n{_as_json(state.get('attribute_hints',[]))}\n\n"
                f"Previous validation error:\n{state.get('validation_error')}"
            ),
        }
    ]

    result = cast(
        ECQLGeneration,
        await ainvoke_llm(messages=messages, response_format=ECQLGeneration, agent_state=state),
    )

    attribute_ecql = result.ecql_string.strip()
    if attribute_ecql.upper() == "NONE":
        attribute_ecql = ""

    final_ecql_parts: list[str] = []
    if spatial_ecql:
        final_ecql_parts.append(f"({spatial_ecql})")
    if attribute_ecql:
        final_ecql_parts.append(f"({attribute_ecql})")

    final_ecql = " AND ".join(final_ecql_parts) if final_ecql_parts else ""

    output = {
        "generated_ecql": final_ecql,
        "retry_count": int(state.get("retry_count", 0)) + 1,
    }
    logger.debug("[ecql_generator_node] output=%s", _as_json(output))
    return output


async def ecql_validation_node(state: AgentState) -> dict[str, str | None]:
    logger.debug(
        "[ecql_validation_node] input generated_ecql=%s geometry_column=%s",
        state.get("generated_ecql", ""),
        state.get("geometry_column", ""),
    )
    generated_ecql = state.get("generated_ecql", "")
    if not generated_ecql.strip():
        logger.warning("[ecql_validation_node] Skipping validation: ECQL filter is empty.")
        output = {"validation_error": None}
        logger.debug("[ecql_validation_node] output=%s", _as_json(output))
        return output

    is_valid, error_msg = validate_ecql(
        ecql_string=generated_ecql,
        layer_schema=state.get("layer_schema", {}),
        geometry_column=state.get("geometry_column", ""),
    )

    if is_valid:
        output: dict[str, str | None] = {"validation_error": None}
        logger.debug("[ecql_validation_node] output=%s", _as_json(output))
        return output

    output = {"validation_error": error_msg}
    logger.debug("[ecql_validation_node] output=%s", _as_json(output))
    return output


async def wfs_executor_node(state: AgentState) -> dict[str, Any]:
    selected_layer = state.get("selected_layer", "")
    generated_ecql = state.get("generated_ecql", "")
    logger.debug(
        "[wfs_executor_node] input selected_layer=%s generated_ecql=%s",
        selected_layer,
        generated_ecql,
    )
    if not selected_layer :
        output = {
            "validation_error": "Missing selected_layer for WFS execution",
        }
        logger.debug("[wfs_executor_node] output=%s", _as_json(output))
        return output

    settings = get_settings()
    wfs_url = settings.geoserver_wfs_url
    wfs_http_client = cast(httpx.AsyncClient | None, state.get("wfs_http_client"))
    username = settings.geoserver_wfs_username
    password = settings.geoserver_wfs_password
    srs_name = settings.geoserver_wfs_srs_name
    count = 1000
    try:
        # Construct the WFS request URL for traceability
        import urllib.parse
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": selected_layer,
            "srsName": srs_name,
            "outputFormat": "application/json",
            "cql_filter": generated_ecql,
            "count": count,
        }
        wfs_request_url = wfs_url + "?" + urllib.parse.urlencode(params)

        payload = await execute_wfs_query(
            wfs_url=wfs_url,
            type_name=selected_layer,
            cql_filter=generated_ecql,
            count=count,
            srs_name=srs_name,
            http_client=wfs_http_client,
            username=username,
            password=password,
        )
    except Exception as exc:
        output = {"validation_error": f"WFS execution failed: {exc}"}
        logger.debug("[wfs_executor_node] output=%s", _as_json(output))
        return output

    output = {
        "wfs_result": payload,
        "wfs_request_url": wfs_request_url,
        "validation_error": None,
    }
    logger.debug("[wfs_executor_node] output=%s", _as_json(output))
    return output


async def fallback_node(state: AgentState) -> dict[str, FinalResponsePayload]:
    error = state.get("validation_error") or "Unknown processing error"
    logger.debug("[fallback_node] input validation_error=%s", error)
    if "Low-confidence layer retrieval" in str(error):
        output = {
            "final_response": _build_final_response(
                summary=(
                    "I could not confidently determine which map layer matches your request. "
                    "Please mention a clearer feature type (for example, hospitals, roads, or schools)."
                )
            )
        }
        logger.debug("[fallback_node] output=%s", _as_json(output))
        return output

    output = {
        "final_response": _build_final_response(
            summary=(
                "I could not complete the spatial query after retries. "
                f"Last error: {error}"
            )
        )
    }
    logger.debug("[fallback_node] output=%s", _as_json(output))
    return output


async def synthesizer_node(state: AgentState) -> dict[str, FinalResponsePayload]:
    wfs_result = state.get("wfs_result", {})
    features = wfs_result.get("features") if isinstance(wfs_result, dict) else None
    # geojson is now communicated in the wfs_executor_node message
    logger.debug(
        "[synthesizer_node] input user_query=%s features_count=%s",
        state.get("user_query", ""),
        len(features) if isinstance(features, list) else 0,
    )

    if not isinstance(features, list):
        output = {
            "final_response": _build_final_response(
                summary="No valid WFS result was available to summarize."
            )
        }
        logger.debug("[synthesizer_node] output=%s", _as_json(output))
        return output

    feature_count = len(features)
    sample_properties = [
        feature.get("properties", {})
        for feature in features[:5]
        if isinstance(feature, dict)
    ]

    messages = [
        {
            "role": "system",
            "content": "Summarize geospatial query results in concise, user-friendly language. in maximum 2 sentences. Focus on the most relevant information for the user's query.",
        },
        {
            "role": "user",
            "content": (
                f"User query: {state.get('user_query', '')}\n"
                f"Feature count: {feature_count}\n"
                f"Sample properties:\n{_as_json(sample_properties)}"
            ),
        },
    ]

    summary = await ainvoke_llm(messages=messages, agent_state=state)
    # Log aggregate usage before final response
    logger.debug("[synthesizer_node] aggregate_usage=%s", json.dumps(state.get("aggregate_usage", {})))
    output = {
        "final_response": _build_final_response(
            summary=str(summary)
        )
    }
    logger.debug("[synthesizer_node] output=%s", _as_json(output))
    return output
