from collections import OrderedDict
import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from litellm import acompletion, completion
from pydantic import BaseModel

from app.core.settings import get_settings
if TYPE_CHECKING:
    from app.graph.state import AgentState


logger = logging.getLogger(__name__)


_PROMPT_CACHE: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()


def _make_prompt_cache_key(
    *,
    messages: list[dict[str, str]],
    model_name: str,
    base_url: str,
    response_format: type[BaseModel] | None,
) -> str:
    payload = {
        "model": model_name,
        "base_url": base_url,
        "response_format": response_format.__name__ if response_format is not None else None,
        "messages": messages,
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cache_get(key: str) -> Any | None:
    now = time.time()
    entry = _PROMPT_CACHE.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if expires_at < now:
        _PROMPT_CACHE.pop(key, None)
        return None
    _PROMPT_CACHE.move_to_end(key)
    return value


def _cache_set(key: str, value: Any, ttl_seconds: int, max_entries: int) -> None:
    now = time.time()
    expires_at = now + max(1, ttl_seconds)
    _PROMPT_CACHE[key] = (expires_at, value)
    _PROMPT_CACHE.move_to_end(key)
    while len(_PROMPT_CACHE) > max(1, max_entries):
        _PROMPT_CACHE.popitem(last=False)


def _normalize_model_name(model: str) -> str:
    # LiteLLM expects Gemini models in provider/model format.
    if model.startswith("gemini") and "/" not in model:
        return f"gemini/{model}"
    return model


def _resolve_provider_api_key(settings: Any, model_name: str) -> str:
    provider = model_name.split("/", 1)[0].lower() if "/" in model_name else ""

    if provider == "gemini" or model_name.lower().startswith("gemini"):
        return settings.gemini_api_key.strip()
    if provider == "anthropic" or model_name.lower().startswith(("claude", "anthropic")):
        return settings.anthropic_api_key.strip()
    return settings.openai_api_key.strip()


def _extract_message_content(response: Any) -> Any:
    logger.debug("[_extract_message_content] extracting content from response type=%s", type(response).__name__)
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")

    if not choices:
        raise ValueError("LLM response did not contain choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None and isinstance(first_choice, dict):
        message = first_choice.get("message")

    if message is None:
        raise ValueError("LLM response did not contain a message")

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if content is None:
        raise ValueError("LLM response did not contain message content")

    return content


def invoke_llm(
    messages: list[dict[str, str]],
    response_format: type[BaseModel] | None = None,
    output_schema: type[BaseModel] | None = None,
    model_name: str | None = None,
    enable_prompt_cache: bool = False,
) -> str | BaseModel:
    if output_schema is not None:
        if response_format is not None and response_format is not output_schema:
            raise ValueError("Provide only one schema or ensure response_format matches output_schema")
        response_format = output_schema

    settings = get_settings()
    selected_model = model_name or settings.current_model
    model_name = _normalize_model_name(selected_model)
    llm_base_url = settings.llm_base_url.strip()
    logger.debug(
        "[invoke_llm] input model=%s base_url_set=%s message_count=%s response_format=%s",
        model_name,
        bool(llm_base_url),
        len(messages),
        response_format.__name__ if response_format is not None else None,
    )
    logger.debug("[invoke_llm] messages=%s", messages)

    cache_key: str | None = None
    if settings.llm_prompt_cache_enabled and enable_prompt_cache:
        cache_key = _make_prompt_cache_key(
            messages=messages,
            model_name=model_name,
            base_url=llm_base_url,
            response_format=response_format,
        )
        cached = _cache_get(cache_key)
        if cached is not None:
            logger.debug("[invoke_llm] prompt cache hit key=%s", cache_key)
            return cached

    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if llm_base_url:
        kwargs["base_url"] = llm_base_url

    api_key = settings.llm_api_key.strip() if llm_base_url else _resolve_provider_api_key(settings, model_name)
    if api_key:
        kwargs["api_key"] = api_key

    if response_format is not None:
        kwargs["response_format"] = response_format

    logger.debug("[invoke_llm] calling completion")
    response = completion(**kwargs)
    logger.debug("[invoke_llm] completion returned response type=%s", type(response).__name__)
    content = _extract_message_content(response)
    logger.debug("[invoke_llm] extracted content type=%s", type(content).__name__)

    if response_format is None:
        output = str(content)
        logger.debug("[invoke_llm] output(raw)=%s", output)
        if cache_key is not None:
            _cache_set(
                cache_key,
                output,
                settings.llm_prompt_cache_ttl_seconds,
                settings.llm_prompt_cache_max_entries,
            )
        return output

    if isinstance(content, str):
        output = response_format.model_validate_json(content)
        logger.debug("[invoke_llm] output(parsed-json)=%s", output.model_dump())
        if cache_key is not None:
            _cache_set(
                cache_key,
                output,
                settings.llm_prompt_cache_ttl_seconds,
                settings.llm_prompt_cache_max_entries,
            )
        return output

    output = response_format.model_validate(content)
    logger.debug("[invoke_llm] output(parsed-object)=%s", output.model_dump())
    if cache_key is not None:
        _cache_set(
            cache_key,
            output,
            settings.llm_prompt_cache_ttl_seconds,
            settings.llm_prompt_cache_max_entries,
        )
    return output


async def ainvoke_llm(
    messages: list[dict[str, str]],
    response_format: type[BaseModel] | None = None,
    output_schema: type[BaseModel] | None = None,
    agent_state: "AgentState | None" = None,
    model_name: str | None = None,
    enable_prompt_cache: bool = False,
    node_name: str | None = None,
) -> str | BaseModel:
    if output_schema is not None:
        if response_format is not None and response_format is not output_schema:
            raise ValueError("Provide only one schema or ensure response_format matches output_schema")
        response_format = output_schema

    settings = get_settings()
    selected_model = model_name or settings.current_model
    model_name = _normalize_model_name(selected_model)
    llm_base_url = settings.llm_base_url.strip()
    logger.debug(
        "[ainvoke_llm] input model=%s base_url_set=%s message_count=%s response_format=%s",
        model_name,
        bool(llm_base_url),
        len(messages),
        response_format.__name__ if response_format is not None else None,
    )
    logger.debug("[ainvoke_llm] messages=%s", messages)

    cache_key: str | None = None
    if settings.llm_prompt_cache_enabled and enable_prompt_cache:
        cache_key = _make_prompt_cache_key(
            messages=messages,
            model_name=model_name,
            base_url=llm_base_url,
            response_format=response_format,
        )
        cached = _cache_get(cache_key)
        if cached is not None:
            logger.debug("[ainvoke_llm] prompt cache hit key=%s", cache_key)
            return cached

    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if llm_base_url:
        kwargs["base_url"] = llm_base_url

    api_key = settings.llm_api_key.strip() if llm_base_url else _resolve_provider_api_key(settings, model_name)
    if api_key:
        kwargs["api_key"] = api_key

    if response_format is not None:
        kwargs["response_format"] = response_format

    logger.debug("[ainvoke_llm] calling acompletion")

    response = await acompletion(**kwargs)
    logger.debug("[ainvoke_llm] acompletion returned response type=%s", type(response).__name__)
    # Aggregate usage if agent_state is provided
    usage = getattr(response, "usage", None)
    if usage and agent_state is not None:
        agg = agent_state.setdefault("aggregate_usage", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
        })
        agg["prompt_tokens"] += usage.get("prompt_tokens", 0)
        agg["completion_tokens"] += usage.get("completion_tokens", 0)
        agg["total_tokens"] += usage.get("total_tokens", 0)
        agg["request_count"] += 1
        if node_name:
            by_node = agg.setdefault("by_node", {})
            node_agg = by_node.setdefault(node_name, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "request_count": 0})
            node_agg["prompt_tokens"] += usage.get("prompt_tokens", 0)
            node_agg["completion_tokens"] += usage.get("completion_tokens", 0)
            node_agg["total_tokens"] += usage.get("total_tokens", 0)
            node_agg["request_count"] += 1

    content = _extract_message_content(response)
    logger.debug("[ainvoke_llm] extracted content type=%s", type(content).__name__)

    if response_format is None:
        output = str(content)
        logger.debug("[ainvoke_llm] output(raw)=%s", output)
        if cache_key is not None:
            _cache_set(
                cache_key,
                output,
                settings.llm_prompt_cache_ttl_seconds,
                settings.llm_prompt_cache_max_entries,
            )
        return output

    if isinstance(content, str):
        output = response_format.model_validate_json(content)
        logger.debug("[ainvoke_llm] output(parsed-json)=%s", output.model_dump())
        if cache_key is not None:
            _cache_set(
                cache_key,
                output,
                settings.llm_prompt_cache_ttl_seconds,
                settings.llm_prompt_cache_max_entries,
            )
        return output

    output = response_format.model_validate(content)
    logger.debug("[ainvoke_llm] output(parsed-object)=%s", output.model_dump())
    if cache_key is not None:
        _cache_set(
            cache_key,
            output,
            settings.llm_prompt_cache_ttl_seconds,
            settings.llm_prompt_cache_max_entries,
        )
    return output
