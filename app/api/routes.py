from __future__ import annotations
import logging

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.http_clients import HttpClientPool, get_http_client_pool
from app.graph.builder import graph
from app.graph.state import build_initial_state
from app.graph.nodes import layer_discoverer_node, wfs_discovery_node

router = APIRouter(prefix="/api", tags=["agent"])
logger = logging.getLogger(__name__)

class SpatialChatRequest(BaseModel):
    query: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)


def _format_sse_event(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"


def _normalize_final_response(value: Any, fallback_summary: str) -> dict[str, str]:
    if isinstance(value, dict):
        summary = value.get("summary")
        result = {"summary": str(summary) if summary is not None else fallback_summary}

        return result

    if isinstance(value, str):
        return {"summary": value}

    return {"summary": fallback_summary}


def _sanitize_update_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "available_layers" and isinstance(item, list):
                sanitized["available_layers_count"] = len(item)
                continue
            if key == "layer_catalog_markdown":
                continue
            if key == "layer_catalog_rows":
                continue
            sanitized[key] = _sanitize_update_payload(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_update_payload(item) for item in value]
    return value


@router.post("/spatial-chat")
async def spatial_chat(
    payload: SpatialChatRequest,
    http_client_pool: HttpClientPool = Depends(get_http_client_pool),
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        inputs = build_initial_state(payload.query)
        latest_state: dict[str, Any] = dict(inputs)
        logger.debug("[spatial_chat] received query=%s thread_id=%s", payload.query, payload.thread_id)
        config = {
            "configurable": {
                "thread_id": payload.thread_id,
                "geocoder_http_client": http_client_pool.geocoder,
                "wfs_http_client": http_client_pool.wfs,
            }
        }
        yield _format_sse_event("status", {"thread_id": payload.thread_id, "status": "starting"})

        try:
            async for update in graph.astream(inputs, stream_mode="updates", config=config):
                if isinstance(update, dict):
                    for node_update in update.values():
                        if isinstance(node_update, dict):
                            latest_state.update(node_update)
                    # Suppress synthesizer step updates to avoid duplicating the final SSE payload.
                    if "synthesizer" in update:
                        continue
                event = {
                    "thread_id": payload.thread_id,
                    "update": _sanitize_update_payload(update),
                }
                yield _format_sse_event("update", event)

            fallback_summary = "Request processed."
            final_payload = _normalize_final_response(
                latest_state.get("final_response"),
                fallback_summary=fallback_summary,
            )
            yield _format_sse_event(
                "final",
                {
                    "thread_id": payload.thread_id,
                    "final_response": final_payload,
                },
            )
            yield _format_sse_event("done", {"thread_id": payload.thread_id, "status": "completed"})
        except Exception as exc:  # pragma: no cover - defensive stream envelope
            error_event = {
                "thread_id": payload.thread_id,
                "status": "failed",
                "message": str(exc),
            }
            yield _format_sse_event("error", error_event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class LayerDiscoveryRequest(BaseModel):
    query: str = Field(min_length=1)


class LayerDiscoveryResult(BaseModel):
    layer_name: str
    validation_error: str | None = None
    retrieval_mode: str | None = None
    retrieval_top_score: float | None = None
    retrieval_score_gap: float | None = None
    retrieval_reason: str | None = None


@router.post("/layer-discovery")
async def layer_discovery(
    payload: LayerDiscoveryRequest,
    http_client_pool: HttpClientPool = Depends(get_http_client_pool),
) -> LayerDiscoveryResult:
    state: dict[str, Any] = {
        "user_query": payload.query,
        "layer_subject": payload.query,
    }
    config = {
        "configurable": {
            "wfs_http_client": http_client_pool.wfs,
        }
    }

    discovery_updates = await wfs_discovery_node(state, config=config)
    state.update(discovery_updates)

    selector_updates = await layer_discoverer_node(state, config=config)

    return LayerDiscoveryResult(
        layer_name=str(selector_updates.get("selected_layer", "")),
        validation_error=selector_updates.get("validation_error"),
        retrieval_mode=selector_updates.get("retrieval_mode"),
        retrieval_top_score=selector_updates.get("retrieval_top_score"),
        retrieval_score_gap=selector_updates.get("retrieval_score_gap"),
        retrieval_reason=selector_updates.get("retrieval_reason"),
    )