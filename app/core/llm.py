from typing import Any
import logging

from litellm import completion
from pydantic import BaseModel

from app.core.settings import get_settings


logger = logging.getLogger(__name__)


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
) -> str | BaseModel:
    if output_schema is not None:
        if response_format is not None and response_format is not output_schema:
            raise ValueError("Provide only one schema or ensure response_format matches output_schema")
        response_format = output_schema

    settings = get_settings()
    model_name = _normalize_model_name(settings.current_model)
    llm_base_url = settings.llm_base_url.strip()
    logger.debug(
        "[invoke_llm] input model=%s base_url_set=%s message_count=%s response_format=%s",
        model_name,
        bool(llm_base_url),
        len(messages),
        response_format.__name__ if response_format is not None else None,
    )
    logger.debug("[invoke_llm] messages=%s", messages)

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
        return output

    if isinstance(content, str):
        output = response_format.model_validate_json(content)
        logger.debug("[invoke_llm] output(parsed-json)=%s", output.model_dump())
        return output

    output = response_format.model_validate(content)
    logger.debug("[invoke_llm] output(parsed-object)=%s", output.model_dump())
    return output
