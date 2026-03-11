from __future__ import annotations
import logging

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.graph.builder import graph
from app.graph.state import build_initial_state

router = APIRouter(prefix="/api", tags=["agent"])
logger = logging.getLogger(__name__)

class SpatialChatRequest(BaseModel):
    query: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)


def _format_sse_event(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"


def _normalize_final_response(value: Any, fallback_summary: str) -> dict[str, object]:
    if isinstance(value, dict):
        summary = value.get("summary")
        geojson = value.get("geojson")
        return {
            "summary": str(summary) if summary is not None else fallback_summary,
            "geojson": geojson if isinstance(geojson, dict) or geojson is None else None,
        }

    if isinstance(value, str):
        return {
            "summary": value,
            "geojson": None,
        }

    return {
        "summary": fallback_summary,
        "geojson": None,
    }


@router.post("/spatial-chat")
async def spatial_chat(payload: SpatialChatRequest) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        inputs = build_initial_state(payload.query)
        latest_state: dict[str, Any] = dict(inputs)
        logger.debug("[spatial_chat] received query=%s thread_id=%s", payload.query, payload.thread_id)
        config = {"configurable": {"thread_id": payload.thread_id}}
        yield _format_sse_event("status", {"thread_id": payload.thread_id, "status": "starting"})

        try:
            async for update in graph.astream(inputs, stream_mode="updates", config=config):
                if isinstance(update, dict):
                    for node_update in update.values():
                        if isinstance(node_update, dict):
                            latest_state.update(node_update)
                event = {"thread_id": payload.thread_id, "update": update}
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