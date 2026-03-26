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


def _normalize_bbox_list(value: Any) -> list[list[float]]:
    if isinstance(value, list) and len(value) == 4:
        normalized = _normalize_bbox(value)
        return [normalized] if normalized is not None else []
    if isinstance(value, list):
        items: list[list[float]] = []
        for entry in value:
            normalized = _normalize_bbox(entry)
            if normalized is not None:
                items.append(normalized)
        return items
    return []


def _normalize_coordinate_pair(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) == 2:
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return None
    return None


def _normalize_coordinate_list(value: Any) -> list[list[float]]:
    if isinstance(value, list) and len(value) == 2:
        normalized = _normalize_coordinate_pair(value)
        return [normalized] if normalized is not None else []
    if isinstance(value, list):
        items: list[list[float]] = []
        for entry in value:
            normalized = _normalize_coordinate_pair(entry)
            if normalized is not None:
                items.append(normalized)
        return items
    return []


def _normalize_spatial_references(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(entry) for entry in value if isinstance(entry, str) and entry.strip()]
    return []


def _normalize_spatial_filters(value: Any) -> list[dict[str, Any] | None]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [entry if isinstance(entry, dict) else None for entry in value]
    return []


def _align_spatial_filters(
    filters: list[dict[str, Any] | None],
    count: int,
) -> list[dict[str, Any] | None]:
    if count <= 0:
        return []
    if not filters:
        return [None for _ in range(count)]
    if len(filters) >= count:
        return filters[:count]
    return filters + [None for _ in range(count - len(filters))]


def _extract_srid(crs: str) -> int | None:
    normalized = str(crs).strip().upper()
    if normalized.startswith("EPSG:"):
        code = normalized.split(":", 1)[1].strip()
        if code.isdigit():
            return int(code)
    return None


def _to_ewkt(wkt: str, crs: str) -> str:
    srid = _extract_srid(crs)
    if srid is None:
        return wkt
    return f"SRID={srid};{wkt}"



async def unified_router_analyzer_node(state: AgentState) -> dict[str, Any]:
    user_query = state.get("user_query", "")
    logger.debug("[unified_router_analyzer_node] input user_query=%s", user_query)
    settings = get_settings()
    routing_model = settings.routing_model.strip() or settings.current_model
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
        await ainvoke_llm(
            messages=messages,
            output_schema=AnalyzedIntent,
            agent_state=state,
            model_name=routing_model,
        ),
    )
    final_payload: FinalResponsePayload | None = None
    if result.intent == "irrelevant":
        summary = (result.general_response or "").strip()
        if not summary:
            summary = "I can help with spatial queries against available map layers."
        final_payload = _build_final_response(summary=summary)

    spatial_filters_payload = None
    if result.spatial_filters:
        spatial_filters_payload = [
            spatial_filter.model_dump(exclude_none=True)
            for spatial_filter in result.spatial_filters
            if spatial_filter is not None
        ]

    output = {
        "intent": result.intent,
        "final_response": final_payload,
        "spatial_reference": result.spatial_reference,
        "explicit_coordinates": result.explicit_coordinates,
        "explicit_bbox": _normalize_bbox(result.explicit_bbox),
        "spatial_filters": spatial_filters_payload,
        "layer_subject": result.layer_subject,
        "attribute_hints": result.attribute_hints or [],
    }
    logger.debug("[unified_router_analyzer_node] output=%s", _as_json(output))
    return output


async def geocoder_context_node(state: AgentState) -> dict[str, Any]:
    spatial_references = _normalize_spatial_references(state.get("spatial_reference"))
    explicit_bboxes = _normalize_bbox_list(state.get("explicit_bbox"))
    explicit_coordinates = _normalize_coordinate_list(state.get("explicit_coordinates"))
    spatial_filters = _normalize_spatial_filters(state.get("spatial_filters"))

    logger.debug(
        "[geocoder_context_node] input references=%s explicit_bboxes=%s explicit_coordinates=%s",
        spatial_references,
        len(explicit_bboxes),
        len(explicit_coordinates),
    )

    # Check for 'Germany' reference (case-insensitive)
    germany_bbox = [5.86632, 47.27011, 15.04193, 55.09916]
    germany_ref = None
    for ref in spatial_references:
        if ref.strip().lower() == "germany":
            germany_ref = ref
            break

    # If explicit geometry exists, ignore textual 'Germany' references.
    if explicit_bboxes or explicit_coordinates:
        spatial_references = [r for r in spatial_references if r.strip().lower() != "germany"]

    # Build explicit geometry requests first.
    requests: list[tuple[str, Any]] = []
    for bbox in explicit_bboxes:
        requests.append(("bbox", bbox))
    for coord in explicit_coordinates:
        requests.append(("point", coord))

    # Only use 'Germany' bbox as fallback if no explicit bbox/coordinates
    if not requests and germany_ref:
        requests.append(("bbox", germany_bbox))
        spatial_references = [r for r in spatial_references if r.strip().lower() != "germany"]

    # Add remaining spatial references (excluding 'Germany' if fallback used)
    for reference in spatial_references:
        requests.append(("reference", reference))

    if not requests:
        output = {"spatial_contexts": []}
        logger.debug("[geocoder_context_node] output=%s", _as_json(output))
        return output

    aligned_filters = _align_spatial_filters(spatial_filters, len(requests))
    contexts: list[dict[str, Any]] = []

    geocoder_http_client = cast(httpx.AsyncClient | None, state.get("geocoder_http_client"))
    geocoder = GeocoderClient(http_client=geocoder_http_client)

    for (kind, value), spatial_filter in zip(requests, aligned_filters):
        if kind == "bbox":
            min_x, min_y, max_x, max_y = value
            crs = "EPSG:4326"
            wkt_polygon = (
                f"POLYGON(({min_x} {min_y}, {min_x} {max_y}, "
                f"{max_x} {max_y}, {max_x} {min_y}, {min_x} {min_y}))"
            )
            contexts.append(
                {
                    "source": "explicit_bbox" if value != germany_bbox else "fallback_germany_bbox",
                    "crs": crs,
                    "bbox": [min_x, min_y, max_x, max_y],
                    "geometry_wkt": _to_ewkt(wkt_polygon, crs),
                    "geometry_type": "Polygon",
                }
            )
            continue

        if kind == "point":
            lon, lat = value
            crs = "EPSG:4326"
            bbox = _build_bbox_for_point(lon, lat, spatial_filter)
            contexts.append(
                {
                    "source": "explicit_point",
                    "crs": crs,
                    "bbox": bbox,
                    "geometry_wkt": _to_ewkt(f"POINT({lon} {lat})", crs),
                    "geometry_type": "Point",
                }
            )
            continue

        if kind == "reference":
            context = await _resolve_reference_context(str(value), spatial_filter, geocoder)
            if context is None:
                output = {"spatial_contexts": []}
                logger.debug("[geocoder_context_node] output=%s", _as_json(output))
                return output
            contexts.append(context)

    output: dict[str, Any] = {"spatial_contexts": contexts}
    logger.debug("[geocoder_context_node] output=%s", _as_json(output))
    return output


def _build_bbox_for_point(
    lon: float,
    lat: float,
    spatial_filter: dict[str, Any] | None,
) -> list[float]:
    buffer_m = None
    if isinstance(spatial_filter, dict):
        predicate = str(spatial_filter.get("predicate") or "").upper()
        distance = spatial_filter.get("distance")
        units = spatial_filter.get("units")
        if predicate in {"DWITHIN", "BEYOND"} and isinstance(distance, int | float) and isinstance(units, str):
            buffer_m = _distance_to_meters(float(distance), units)

    if buffer_m is None:
        epsilon = 0.0001
        return [lon - epsilon, lat - epsilon, lon + epsilon, lat + epsilon]

    transformer_to_3857 = Transformer.from_crs(4326, 3857, always_xy=True)
    transformer_to_4326 = Transformer.from_crs(3857, 4326, always_xy=True)
    x, y = transformer_to_3857.transform(lon, lat)
    minx = x - buffer_m
    maxx = x + buffer_m
    miny = y - buffer_m
    maxy = y + buffer_m
    llon, llat = transformer_to_4326.transform(min(minx, maxx), min(miny, maxy))
    ulon, ulat = transformer_to_4326.transform(max(minx, maxx), max(miny, maxy))
    return [llon, llat, ulon, ulat]


async def _resolve_reference_context(
    query: str,
    spatial_filter: dict[str, Any] | None,
    geocoder: GeocoderClient,
) -> dict[str, Any] | None:
    if not query:
        return None

    suggest_payload = await geocoder.suggest(query=query, max_results=1)
    suggest_results = suggest_payload.get("SuggestResult") or []
    location_type = None
    if suggest_results:
        location_type = suggest_results[0].get("locationType")

    payload = await geocoder.forward_fulltext(query=query, max_results=1, epsg=4326)
    results = payload.get("Result") or []
    first_result = results[0] if results else {}

    coord_str = first_result.get("Coordinate")
    if not coord_str:
        return None

    try:
        lat_str, lon_str = coord_str.split(",")
        lat, lon = float(lat_str.strip()), float(lon_str.strip())
    except Exception:
        return None

    buffer_m = None
    if isinstance(spatial_filter, dict):
        predicate = str(spatial_filter.get("predicate") or "").upper()
        distance = spatial_filter.get("distance")
        units = spatial_filter.get("units")
        if predicate in {"DWITHIN", "BEYOND"} and isinstance(distance, int | float) and isinstance(units, str):
            buffer_m = _distance_to_meters(float(distance), units)

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

    if buffer_m is None:
        buffer_m = 5000
    logger.info("[geocoder_context_node] buffer_m=%s", buffer_m)

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

    if location_type in {"POI", "Address", "Unknown"}:
        geometry_wkt = Point(lon, lat).wkt
        geometry_type = "Point"
    else:
        polygon = box(*bbox_4326)
        geometry_wkt = polygon.wkt
        geometry_type = "Polygon"

    crs = "EPSG:4326"

    return {
        "source": "reference",
        "crs": crs,
        "bbox": bbox_4326,
        "geometry_wkt": _to_ewkt(geometry_wkt, crs),
        "geometry_type": geometry_type,
    }


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
        await ainvoke_llm(
            messages=messages,
            response_format=LayerSelection,
            agent_state=state,
            enable_prompt_cache=True,
        ),
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
    spatial_contexts = state.get("spatial_contexts")
    spatial_filters = state.get("spatial_filters")
    spatial_ecql = build_spatial_ecql(
        geom_column=state.get("geometry_column", ""),
        spatial_contexts=cast(list[dict[str, Any]] | None, spatial_contexts),
        spatial_filters=cast(list[dict[str, Any] | None] | None, spatial_filters),
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

    settings = get_settings()
    synthesizer_model = settings.synthesizer_model.strip() or settings.current_model
    summary = await ainvoke_llm(messages=messages, agent_state=state, model_name=synthesizer_model)
    # Log aggregate usage before final response
    logger.debug("[synthesizer_node] aggregate_usage=%s", json.dumps(state.get("aggregate_usage", {})))
    output = {
        "final_response": _build_final_response(
            summary=str(summary)
        )
    }
    logger.debug("[synthesizer_node] output=%s", _as_json(output))
    return output
