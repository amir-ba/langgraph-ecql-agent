from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, cast

import httpx
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from pyproj import Transformer
from shapely import wkt as shapely_wkt
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


@lru_cache(maxsize=8)
def _get_transformer(from_epsg: int, to_epsg: int) -> Transformer:
    return Transformer.from_crs(from_epsg, to_epsg, always_xy=True)


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


def _normalize_spatial_targets(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict)]


def _normalize_spatial_predicates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict)]


def _distance_filter_for_target(
    target_id: str,
    spatial_predicates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for predicate in spatial_predicates:
        predicate_name = str(predicate.get("predicate") or "").upper()
        target_ids = predicate.get("target_ids")
        distance = predicate.get("distance")
        units = predicate.get("units")
        if (
            isinstance(target_ids, list)
            and target_id in target_ids
            and predicate_name in {"DWITHIN", "BEYOND"}
            and isinstance(distance, int | float)
            and isinstance(units, str)
        ):
            candidates.append({
                "predicate": predicate_name,
                "distance": float(distance),
                "units": units,
            })
    if not candidates:
        return None
    # Use the largest distance to size the pre-filter bbox conservatively.
    return max(
        candidates,
        key=lambda c: _distance_to_meters(c["distance"], c["units"]) or 0.0,
    )


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



async def unified_router_analyzer_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
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
                "target features (layer subject), id-bound spatial_targets, "
                "id-bound spatial_predicates, explicit WKT geometries "
                "(POINT/LINESTRING/POLYGON), and attributes. "
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

    spatial_targets_payload = [
        target.model_dump(exclude_none=True)
        for target in (result.spatial_targets or [])
    ]
    spatial_predicates_payload = [
        predicate.model_dump(exclude_none=True)
        for predicate in (result.spatial_predicates or [])
    ]

    output = {
        "intent": result.intent,
        "final_response": final_payload,
        "spatial_targets": spatial_targets_payload,
        "spatial_predicates": spatial_predicates_payload,
        "layer_subject": result.layer_subject,
        "attribute_hints": result.attribute_hints or [],
    }
    logger.debug("[unified_router_analyzer_node] output=%s", _as_json(output))
    return output


async def geocoder_context_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    spatial_targets = _normalize_spatial_targets(state.get("spatial_targets"))
    spatial_predicates = _normalize_spatial_predicates(state.get("spatial_predicates"))

    logger.debug(
        "[geocoder_context_node] input targets=%s predicates=%s",
        len(spatial_targets),
        len(spatial_predicates),
    )

    if not spatial_targets:
        output = {"spatial_contexts": []}
        logger.debug("[geocoder_context_node] output=%s", _as_json(output))
        return output

    contexts: list[dict[str, Any]] = []
    unresolved_target_ids: list[str] = []

    configurable = (config or {}).get("configurable") or {}
    geocoder_http_client = cast(httpx.AsyncClient | None, configurable.get("geocoder_http_client"))
    geocoder = GeocoderClient(http_client=geocoder_http_client)

    for target in spatial_targets:
        target_id = str(target.get("id") or "").strip()
        target_kind = str(target.get("kind") or "").strip().lower()
        target_value = str(target.get("value") or "").strip()
        required = bool(target.get("required", True))

        if not target_id or not target_kind or not target_value:
            if required and target_id:
                unresolved_target_ids.append(target_id)
            continue

        distance_filter = _distance_filter_for_target(target_id, spatial_predicates)

        context: dict[str, Any] | None = None
        if target_kind == "explicit_geometry":
            context = _build_explicit_geometry_context(target_id, target_value, distance_filter)
        elif target_kind == "spatial_reference":
            context = await _resolve_reference_context(target_id, target_value, distance_filter, geocoder)

        if context is None:
            if required:
                unresolved_target_ids.append(target_id)
            continue
        contexts.append(context)

    resolved_target_ids = {str(context.get("target_id") or "") for context in contexts}
    filtered_predicates: list[dict[str, Any]] = []
    for predicate in spatial_predicates:
        target_ids = predicate.get("target_ids")
        if not isinstance(target_ids, list):
            continue
        resolved_ids = [
            str(target_id)
            for target_id in target_ids
            if isinstance(target_id, str) and target_id in resolved_target_ids
        ]
        if not resolved_ids:
            if bool(predicate.get("required", True)):
                unresolved_target_ids.extend(
                    [str(target_id) for target_id in target_ids if isinstance(target_id, str)]
                )
            continue
        next_predicate = dict(predicate)
        next_predicate["target_ids"] = resolved_ids
        filtered_predicates.append(next_predicate)

    if unresolved_target_ids:
        unresolved_unique = sorted({target_id for target_id in unresolved_target_ids if target_id})
        output: dict[str, Any] = {
            "spatial_contexts": contexts,
            "spatial_predicates": filtered_predicates,
            "unresolved_target_ids": unresolved_unique,
            "validation_error": "Required spatial targets could not be resolved: " + ", ".join(unresolved_unique),
        }
        logger.debug("[geocoder_context_node] output=%s", _as_json(output))
        return output

    output = {
        "spatial_contexts": contexts,
        "spatial_predicates": filtered_predicates,
        "unresolved_target_ids": [],
        "validation_error": None,
    }
    logger.debug("[geocoder_context_node] output=%s", _as_json(output))
    return output


def _build_explicit_geometry_context(
    target_id: str,
    geometry_wkt: str,
    spatial_filter: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not geometry_wkt:
        return None

    try:
        geom = shapely_wkt.loads(geometry_wkt)
    except Exception:
        return None

    crs = "EPSG:4326"
    geometry_type = str(getattr(geom, "geom_type", "") or "")
    if not geometry_type:
        return None

    if geometry_type == "Point":
        coords = list(geom.coords)
        if not coords:
            return None
        lon, lat = float(coords[0][0]), float(coords[0][1])
        bbox = _build_bbox_for_point(lon, lat, spatial_filter)
    else:
        min_x, min_y, max_x, max_y = geom.bounds
        bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]

    return {
        "target_id": target_id,
        "source": "explicit_geometry",
        "crs": crs,
        "bbox": bbox,
        "geometry_wkt": _to_ewkt(geom.wkt, crs),
        "geometry_type": geometry_type,
    }


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

    transformer_to_3857 = _get_transformer(4326, 3857)
    transformer_to_4326 = _get_transformer(3857, 4326)
    x, y = transformer_to_3857.transform(lon, lat)
    minx = x - buffer_m
    maxx = x + buffer_m
    miny = y - buffer_m
    maxy = y + buffer_m
    llon, llat = transformer_to_4326.transform(min(minx, maxx), min(miny, maxy))
    ulon, ulat = transformer_to_4326.transform(max(minx, maxx), max(miny, maxy))
    return [llon, llat, ulon, ulat]


async def _resolve_reference_context(
    target_id: str,
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

    transformer_to_3857 = _get_transformer(4326, 3857)
    transformer_to_4326 = _get_transformer(3857, 4326)
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
        "target_id": target_id,
        "source": "reference",
        "crs": crs,
        "bbox": bbox_4326,
        "geometry_wkt": _to_ewkt(geometry_wkt, crs),
        "geometry_type": geometry_type,
    }


async def wfs_discovery_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, list[dict[str, str]] | str | None]:
    logger.debug("[wfs_discovery_node] input received")
    settings = get_settings()
    configurable = (config or {}).get("configurable") or {}
    wfs_http_client = cast(httpx.AsyncClient | None, configurable.get("wfs_http_client"))
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


async def layer_discoverer_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
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


async def schema_extractor_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
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
    configurable = (config or {}).get("configurable") or {}
    wfs_http_client = cast(httpx.AsyncClient | None, configurable.get("wfs_http_client"))
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


async def ecql_generator_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, str | int]:
    logger.debug(
        "[ecql_generator_node] input user_query=%s retry_count=%s",
        state.get("user_query", ""),
        state.get("retry_count", 0),
    )
    spatial_contexts = state.get("spatial_contexts")
    spatial_predicates = state.get("spatial_predicates")
    spatial_ecql = build_spatial_ecql(
        geom_column=state.get("geometry_column", ""),
        spatial_contexts=cast(list[dict[str, Any]] | None, spatial_contexts),
        spatial_predicates=cast(list[dict[str, Any]] | None, spatial_predicates),
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


async def ecql_validation_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, str | None]:
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


async def wfs_executor_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
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
    configurable = (config or {}).get("configurable") or {}
    wfs_http_client = cast(httpx.AsyncClient | None, configurable.get("wfs_http_client"))
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


async def fallback_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, FinalResponsePayload]:
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


async def synthesizer_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, FinalResponsePayload]:
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
