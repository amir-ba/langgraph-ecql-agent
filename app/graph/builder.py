from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
import inspect
import logging
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    ecql_generator_node,
    ecql_validation_node,
    fallback_node,
    geocoder_context_node,
    layer_discoverer_node,
    schema_extractor_node,
    synthesizer_node,
    unified_router_analyzer_node,
    wfs_discovery_node,
    wfs_executor_node,
)
from app.graph.state import AgentState

NodeFn = Callable[[AgentState, RunnableConfig], Awaitable[dict[str, Any]]]
logger = logging.getLogger(__name__)


def _with_node_logging(node_name: str, node_fn: NodeFn) -> NodeFn:
    _accepts_config = len(inspect.signature(node_fn).parameters) >= 2

    async def _wrapped(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        logger.debug(
            "[graph] node_triggered=%s state_keys=%s",
            node_name,
            sorted(state.keys()),
        )
        if _accepts_config:
            output = await node_fn(state, config)  # type: ignore[arg-type]
        else:
            output = await node_fn(state)  # type: ignore[call-arg]
        logger.debug(
            "[graph] node_completed=%s output_keys=%s",
            node_name,
            sorted(output.keys()),
        )
        return output

    return _wrapped


def route_after_generator(state: AgentState) -> Literal["validator", "fallback"]:
    ecql = str(state.get("generated_ecql") or "").strip()
    if not ecql or ecql.upper() in {"NONE", "NULL"}:
        return "fallback"
    return "validator"


def route_after_analysis(state: AgentState) -> Literal["geocoder", "end"]:
    if state.get("intent") == "spatial_query" and state.get("layer_subject"):
        return "geocoder"
    return "end"


def route_after_geocoder(state: AgentState) -> Literal["discoverer", "fallback"]:
    if state.get("validation_error"):
        return "fallback"
    return "discoverer"


def validator_router(state: AgentState) -> Literal["generator", "fallback", "executor"]:
    retry_count = int(state.get("retry_count", 0))
    if state.get("validation_error") and retry_count < 3:
        return "generator"
    if state.get("validation_error"):
        return "fallback"
    return "executor"


def layer_selector_router(state: AgentState) -> Literal["schema", "fallback"]:
    if state.get("validation_error") and not state.get("selected_layer"):
        return "fallback"
    return "schema"


def executor_router(state: AgentState) -> Literal["generator", "fallback", "synthesizer"]:
    retry_count = int(state.get("retry_count", 0))
    if state.get("validation_error") and retry_count < 3:
        return "generator"
    if state.get("validation_error"):
        return "fallback"
    return "synthesizer"


def build_graph(node_overrides: Mapping[str, NodeFn] | None = None):
    node_map: dict[str, NodeFn] = {
        "router_analyzer": unified_router_analyzer_node,
        "geocoder": geocoder_context_node,
        "discoverer": wfs_discovery_node,
        "layer_selector": layer_discoverer_node,
        "schema": schema_extractor_node,
        "generator": ecql_generator_node,
        "validator": ecql_validation_node,
        "executor": wfs_executor_node,
        "synthesizer": synthesizer_node,
        "fallback": fallback_node,
    }
    if node_overrides:
        node_map.update(node_overrides)

    builder = StateGraph(AgentState)
    for name, node in node_map.items():
        builder.add_node(name, cast(Any, _with_node_logging(name, node)))

    builder.add_edge(START, "router_analyzer")
    builder.add_conditional_edges(
        "router_analyzer",
        route_after_analysis,
        {
            "geocoder": "geocoder",
            "end": END,
        },
    )

    builder.add_conditional_edges(
        "geocoder",
        route_after_geocoder,
        {
            "discoverer": "discoverer",
            "fallback": "fallback",
        },
    )
    builder.add_edge("discoverer", "layer_selector")
    builder.add_conditional_edges(
        "layer_selector",
        layer_selector_router,
        {
            "schema": "schema",
            "fallback": "fallback",
        },
    )
    builder.add_edge("schema", "generator")
    builder.add_conditional_edges(
        "generator",
        route_after_generator,
        {
            "validator": "validator",
            "fallback": "fallback",
        },
    )

    builder.add_conditional_edges("validator", validator_router)
    builder.add_conditional_edges("executor", executor_router)

    builder.add_edge("synthesizer", END)
    builder.add_edge("fallback", END)

    return builder.compile()


graph = build_graph()
