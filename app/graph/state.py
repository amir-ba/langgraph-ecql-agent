from typing import Any, NotRequired, TypedDict


class FinalResponsePayload(TypedDict):
    summary: str
    geojson: dict[str, Any] | None


class AgentState(TypedDict):
    user_query: str

    intent: NotRequired[str]
    final_response: NotRequired[FinalResponsePayload | None]

    spatial_reference: NotRequired[str | None]
    spatial_relationship: NotRequired[str | None]
    layer_subject: NotRequired[str | None]
    attribute_hints: NotRequired[list[str]]
    spatial_context_bbox: NotRequired[list[float] | None]

    available_layers: NotRequired[list[dict[str, str]]]
    selected_layer: NotRequired[str]
    layer_schema: NotRequired[dict[str, str]]
    geometry_column: NotRequired[str]

    generated_ecql: NotRequired[str]
    validation_error: NotRequired[str | None]
    retry_count: NotRequired[int]

    wfs_request_url: NotRequired[str]
    wfs_result: NotRequired[dict[str, Any]]


def build_initial_state(user_query: str) -> AgentState:
    return AgentState(
        user_query=user_query,
        intent="irrelevant",
        final_response=None,
        spatial_reference=None,
        spatial_relationship=None,
        layer_subject=None,
        attribute_hints=[],
        spatial_context_bbox=None,
        retry_count=0,
        validation_error=None,
    )
