from __future__ import annotations

import json
import logging
from typing import Any, cast

from pydantic import BaseModel, Field

from app.core.llm import invoke_llm
from app.core.schemas import AnalyzedIntent, ECQLGeneration
from app.core.settings import get_settings
from app.graph.state import AgentState, FinalResponsePayload
from app.tools.ecql_validator import validate_ecql
from app.tools.geocoder import GeocoderClient
from app.tools.wfs_client import (
    discover_layers,
    execute_wfs_query,
    filter_layers_by_subject,
    get_layer_schema,
)


logger = logging.getLogger(__name__)


class LayerSelection(BaseModel):
    layer_name: str = Field(description="The selected WFS layer name")


def _as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, default=str)


def _build_final_response(summary: str, geojson: dict[str, Any] | None = None) -> FinalResponsePayload:
    return {
        "summary": summary,
        "geojson": geojson,
    }


def _normalize_geojson_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    return None


def unified_router_analyzer_node(state: AgentState) -> dict[str, Any]:
    user_query = state.get("user_query", "")
    logger.debug("[unified_router_analyzer_node] input user_query=%s", user_query)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a geospatial AI assistant. Determine if the user is asking "
                "for spatial data or just chatting. If spatial, extract the location, "
                "target features (layer subject), and attributes. "
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
        invoke_llm(messages=messages, output_schema=AnalyzedIntent),
    )
    final_payload: FinalResponsePayload | None = None
    if result.intent in {"general_chat", "irrelevant"}:
        summary = (result.general_response or "").strip()
        if not summary:
            summary = "I can help with spatial queries against available map layers."
        final_payload = _build_final_response(summary=summary, geojson=None)

    output = {
        "intent": result.intent,
        "final_response": final_payload,
        "spatial_reference": result.spatial_reference,
        "spatial_relationship": result.spatial_relationship,
        "layer_subject": result.layer_subject,
        "attribute_hints": result.attribute_hints or [],
    }
    logger.debug("[unified_router_analyzer_node] output=%s", _as_json(output))
    return output


def geocoder_context_node(state: AgentState) -> dict[str, Any]:
    query = state.get("spatial_reference") or ""
    logger.debug("[geocoder_context_node] input query=%s", query)
    if not query:
        output = {"spatial_context_bbox": None}
        logger.debug("[geocoder_context_node] output=%s", _as_json(output))
        return output

    geocoder = GeocoderClient()
    payload = geocoder.forward_fulltext(query=query, max_results=1, epsg=4326)
    results = payload.get("Result") or []
    first_result = results[0] if results else {}

    output = {
        "spatial_context_bbox": first_result.get("BoundingBox") or first_result.get("bbox"),
    }
    logger.debug("[geocoder_context_node] output=%s", _as_json(output))
    return output


def wfs_discovery_node(_: AgentState) -> dict[str, list[dict[str, str]] | str | None]:
    logger.debug("[wfs_discovery_node] input received")
    settings = get_settings()
    layers = discover_layers(
        wfs_url=settings.geoserver_wfs_url,
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

    output = {
        "available_layers": layers,
        "validation_error": None,
    }
    logger.debug("[wfs_discovery_node] output=%s", _as_json(output))
    return output


def layer_discoverer_node(state: AgentState) -> dict[str, str | None]:
    available_layers = state.get("available_layers", [])
    layer_subject = state.get("layer_subject")
    logger.debug(
        "[layer_discoverer_node] input user_query=%s layer_subject=%s available_layers_count=%s",
        state.get("user_query", ""),
        layer_subject,
        len(available_layers) if isinstance(available_layers, list) else 0,
    )
    if not available_layers:
        output: dict[str, str | None] = {
            "selected_layer": "",
            "validation_error": "No WFS layers available for selection",
        }
        logger.debug("[layer_discoverer_node] output=%s", _as_json(output))
        return output

    candidate_layers = filter_layers_by_subject(available_layers, layer_subject)
    if layer_subject and len(candidate_layers) == 1:
        output = {
            "selected_layer": str(candidate_layers[0].get("name", "")),
            "validation_error": None,
        }
        logger.debug("[layer_discoverer_node] output=%s", _as_json(output))
        return output

    if not candidate_layers:
        candidate_layers = available_layers

    messages = [
        {
            "role": "system",
            "content": (
                "Pick the best layer for the query from the provided WFS layers. "
                "Return exactly one layer_name from the list."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query:\n{state.get('user_query', '')}\n\n"
                f"Layer subject:\n{layer_subject or ''}\n\n"
                f"Available layers:\n{_as_json(candidate_layers)}"
            ),
        },
    ]

    result = cast(
        LayerSelection,
        invoke_llm(messages=messages, response_format=LayerSelection),
    )
    output = {
        "selected_layer": result.layer_name,
        "validation_error": None,
    }
    logger.debug("[layer_discoverer_node] output=%s", _as_json(output))
    return output


def schema_extractor_node(state: AgentState) -> dict[str, Any]:
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
    attributes, geometry_column = get_layer_schema(
        wfs_url=settings.geoserver_wfs_url,
        type_name=selected_layer,
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


def ecql_generator_node(state: AgentState) -> dict[str, str | int]:
    logger.debug(
        "[ecql_generator_node] input user_query=%s geometry_column=%s retry_count=%s",
        state.get("user_query", ""),
        state.get("geometry_column", ""),
        state.get("retry_count", 0),
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Generate a valid GeoServer ECQL expression. "
                "Use only provided schema attributes and the exact geometry column."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query:\n{state.get('user_query', '')}\n\n"
                f"Layer schema:\n{_as_json(state.get('layer_schema', {}))}\n\n"
                f"Geometry column:\n{state.get('geometry_column', '')}\n\n"
                f"Spatial context bbox:\n{_as_json(state.get('spatial_context_bbox'))}\n\n"
                f"Attribute hints:\n{_as_json(state.get('attribute_hints', []))}\n\n"
                f"Previous validation error:\n{state.get('validation_error')}"
            ),
        },
    ]

    result = cast(
        ECQLGeneration,
        invoke_llm(messages=messages, response_format=ECQLGeneration),
    )
    output = {
        "generated_ecql": result.ecql_string,
        "retry_count": int(state.get("retry_count", 0)) + 1,
    }
    logger.debug("[ecql_generator_node] output=%s", _as_json(output))
    return output


def ecql_validation_node(state: AgentState) -> dict[str, str | None]:
    logger.debug(
        "[ecql_validation_node] input generated_ecql=%s geometry_column=%s",
        state.get("generated_ecql", ""),
        state.get("geometry_column", ""),
    )
    is_valid, error_msg = validate_ecql(
        ecql_string=state.get("generated_ecql", ""),
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


def wfs_executor_node(state: AgentState) -> dict[str, Any]:
    selected_layer = state.get("selected_layer", "")
    generated_ecql = state.get("generated_ecql", "")
    logger.debug(
        "[wfs_executor_node] input selected_layer=%s generated_ecql=%s",
        selected_layer,
        generated_ecql,
    )
    if not selected_layer or not generated_ecql:
        output = {
            "validation_error": "Missing selected_layer or generated_ecql for WFS execution",
        }
        logger.debug("[wfs_executor_node] output=%s", _as_json(output))
        return output

    settings = get_settings()
    wfs_url = settings.geoserver_wfs_url
    username = settings.geoserver_wfs_username
    password = settings.geoserver_wfs_password
    count = 1000
    try:
        # Construct the WFS request URL for traceability
        import urllib.parse
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": selected_layer,
            "outputFormat": "application/json",
            "cql_filter": generated_ecql,
            "count": count,
        }
        wfs_request_url = wfs_url + "?" + urllib.parse.urlencode(params)

        payload = execute_wfs_query(
            wfs_url=wfs_url,
            type_name=selected_layer,
            cql_filter=generated_ecql,
            count=count,
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


def fallback_node(state: AgentState) -> dict[str, FinalResponsePayload]:
    error = state.get("validation_error") or "Unknown processing error"
    logger.debug("[fallback_node] input validation_error=%s", error)
    output = {
        "final_response": _build_final_response(
            summary=(
                "I could not complete the spatial query after retries. "
                f"Last error: {error}"
            ),
            geojson=None,
        )
    }
    logger.debug("[fallback_node] output=%s", _as_json(output))
    return output


def synthesizer_node(state: AgentState) -> dict[str, FinalResponsePayload]:
    wfs_result = state.get("wfs_result", {})
    features = wfs_result.get("features") if isinstance(wfs_result, dict) else None
    geojson = _normalize_geojson_payload(wfs_result)
    logger.debug(
        "[synthesizer_node] input user_query=%s features_count=%s",
        state.get("user_query", ""),
        len(features) if isinstance(features, list) else 0,
    )

    if not isinstance(features, list):
        output = {
            "final_response": _build_final_response(
                summary="No valid WFS result was available to summarize.",
                geojson=geojson,
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
            "content": "Summarize geospatial query results in concise, user-friendly language.",
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

    summary = invoke_llm(messages=messages)
    output = {
        "final_response": _build_final_response(
            summary=str(summary),
            geojson=geojson,
        )
    }
    logger.debug("[synthesizer_node] output=%s", _as_json(output))
    return output
