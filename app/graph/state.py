from typing import Any, NotRequired, TypedDict


class FinalResponsePayload(TypedDict, total=False):
    summary: str


class AgentState(TypedDict):
    user_query: str

    intent: NotRequired[str]
    final_response: NotRequired[FinalResponsePayload | None]

    spatial_targets: NotRequired[list[dict[str, Any]] | None]
    spatial_predicates: NotRequired[list[dict[str, Any]] | None]
    layer_subject: NotRequired[str | None]
    attribute_hints: NotRequired[list[str]]
    spatial_contexts: NotRequired[list[dict[str, Any]] | None]
    unresolved_target_ids: NotRequired[list[str]]
    available_layers: NotRequired[list[dict[str, str]]]
    layer_catalog_markdown: NotRequired[str]
    layer_catalog_rows: NotRequired[list[dict[str, Any]]]
    retrieval_mode: NotRequired[str]
    retrieval_top_score: NotRequired[float | None]
    retrieval_score_gap: NotRequired[float | None]
    retrieval_reason: NotRequired[str]
    candidate_layers_for_llm_count: NotRequired[int]
    selected_layer: NotRequired[str]
    layer_schema: NotRequired[dict[str, str]]
    geometry_column: NotRequired[str]

    generated_ecql: NotRequired[str]
    validation_error: NotRequired[str | None]
    retry_count: NotRequired[int]

    wfs_request_url: NotRequired[str]
    wfs_result: NotRequired[dict[str, Any]]

    aggregate_usage: NotRequired[dict[str, Any]]  # keys: prompt_tokens, completion_tokens, total_tokens, request_count, by_node


def build_initial_state(user_query: str) -> AgentState:
    return AgentState(
        user_query=user_query,
        intent="irrelevant",
        final_response=None,
        spatial_targets=None,
        spatial_predicates=None,
        layer_subject=None,
        attribute_hints=[],
        spatial_contexts=[],
        unresolved_target_ids=[],
        retry_count=0,
        validation_error=None,
        aggregate_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
        },
    )
