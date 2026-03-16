from typing import Any, NotRequired, TypedDict

import httpx


class FinalResponsePayload(TypedDict, total=False):
    summary: str


class AgentState(TypedDict):
    user_query: str

    intent: NotRequired[str]
    final_response: NotRequired[FinalResponsePayload | None]

    spatial_reference: NotRequired[str | None]
    spatial_filter: NotRequired[dict[str, Any] | None]
    layer_subject: NotRequired[str | None]
    attribute_hints: NotRequired[list[str]]
    spatial_context: NotRequired[dict[str, Any] | None]
    explicit_coordinates: NotRequired[list[float] | None]
    explicit_bbox: NotRequired[list[float] | None]

    available_layers: NotRequired[list[dict[str, str]]]
    layer_catalog_markdown: NotRequired[str]
    retrieval_mode: NotRequired[str]
    retrieval_top_score: NotRequired[float | None]
    retrieval_score_gap: NotRequired[float | None]
    candidate_layers_for_llm_count: NotRequired[int]
    selected_layer: NotRequired[str]
    layer_schema: NotRequired[dict[str, str]]
    geometry_column: NotRequired[str]

    generated_ecql: NotRequired[str]
    validation_error: NotRequired[str | None]
    retry_count: NotRequired[int]

    wfs_request_url: NotRequired[str]
    wfs_result: NotRequired[dict[str, Any]]

    geocoder_http_client: NotRequired[httpx.AsyncClient]
    wfs_http_client: NotRequired[httpx.AsyncClient]

    aggregate_usage: NotRequired[dict[str, int]]  # keys: prompt_tokens, completion_tokens, total_tokens, request_count


def build_initial_state(
    user_query: str,
    geocoder_http_client: httpx.AsyncClient | None = None,
    wfs_http_client: httpx.AsyncClient | None = None,
) -> AgentState:
    state = AgentState(
        user_query=user_query,
        intent="irrelevant",
        final_response=None,
        spatial_reference=None,
        spatial_filter=None,
        layer_subject=None,
        attribute_hints=[],
        spatial_context=None,
        retry_count=0,
        validation_error=None,
        aggregate_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
        },
    )
    if geocoder_http_client is not None:
        state["geocoder_http_client"] = geocoder_http_client
    if wfs_http_client is not None:
        state["wfs_http_client"] = wfs_http_client
    return state
