from pydantic import BaseModel

from app.core.llm import invoke_llm
from app.core.settings import Settings


class _DummyResponse:
    def __init__(self, content: str) -> None:
        message = type("Message", (), {"content": content})
        choice = type("Choice", (), {"message": message()})
        self.choices = [choice()]


def test_invoke_llm_uses_current_model_from_settings(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return _DummyResponse("plain text")

    monkeypatch.setattr(
        "app.core.llm.get_settings",
        lambda: Settings(
            current_model="claude-3-5-sonnet",
            llm_base_url="",
            anthropic_api_key="anthropic-key",
        ),
    )
    monkeypatch.setattr("app.core.llm.completion", fake_completion)

    result = invoke_llm(messages=[{"role": "user", "content": "hello"}])

    assert seen["model"] == "claude-3-5-sonnet"
    assert seen["messages"] == [{"role": "user", "content": "hello"}]
    assert seen["api_key"] == "anthropic-key"
    assert "base_url" not in seen
    assert result == "plain text"


def test_invoke_llm_returns_validated_model_when_response_format_is_provided(monkeypatch) -> None:
    class ExampleSchema(BaseModel):
        value: str

    def fake_completion(**kwargs):
        assert kwargs["response_format"] is ExampleSchema
        return _DummyResponse('{"value":"ok"}')

    monkeypatch.setattr("app.core.llm.get_settings", lambda: Settings(current_model="gpt-4o"))
    monkeypatch.setattr("app.core.llm.completion", fake_completion)

    result = invoke_llm(
        messages=[{"role": "user", "content": "return JSON"}],
        response_format=ExampleSchema,
    )

    assert isinstance(result, ExampleSchema)
    assert result.value == "ok"


def test_invoke_llm_returns_validated_model_when_output_schema_is_provided(monkeypatch) -> None:
    class ExampleSchema(BaseModel):
        value: str

    def fake_completion(**kwargs):
        assert kwargs["response_format"] is ExampleSchema
        return _DummyResponse('{"value":"ok"}')

    monkeypatch.setattr("app.core.llm.get_settings", lambda: Settings(current_model="gpt-4o"))
    monkeypatch.setattr("app.core.llm.completion", fake_completion)

    result = invoke_llm(
        messages=[{"role": "user", "content": "return JSON"}],
        output_schema=ExampleSchema,
    )

    assert isinstance(result, ExampleSchema)
    assert result.value == "ok"


def test_invoke_llm_normalizes_short_gemini_model_name(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return _DummyResponse("plain text")

    monkeypatch.setattr("app.core.llm.get_settings", lambda: Settings(current_model="gemini-2.0-flash"))
    monkeypatch.setattr("app.core.llm.completion", fake_completion)

    result = invoke_llm(messages=[{"role": "user", "content": "hello"}])

    assert seen["model"] == "gemini/gemini-2.0-flash"
    assert result == "plain text"


def test_invoke_llm_uses_base_url_from_settings(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return _DummyResponse("plain text")

    monkeypatch.setattr(
        "app.core.llm.get_settings",
        lambda: Settings(
            current_model="gpt-4o",
            llm_base_url="https://llm.example.com/v1",
            llm_api_key="generic-key",
            openai_api_key="openai-key",
        ),
    )
    monkeypatch.setattr("app.core.llm.completion", fake_completion)

    result = invoke_llm(messages=[{"role": "user", "content": "hello"}])

    assert seen["base_url"] == "https://llm.example.com/v1"
    assert seen["api_key"] == "generic-key"
    assert result == "plain text"


def test_invoke_llm_omits_base_url_when_empty_after_trim(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return _DummyResponse("plain text")

    monkeypatch.setattr(
        "app.core.llm.get_settings",
        lambda: Settings(current_model="gpt-4o", llm_base_url="   ", openai_api_key="openai-key"),
    )
    monkeypatch.setattr("app.core.llm.completion", fake_completion)

    result = invoke_llm(messages=[{"role": "user", "content": "hello"}])

    assert "base_url" not in seen
    assert seen["api_key"] == "openai-key"
    assert result == "plain text"


def test_invoke_llm_uses_gemini_key_when_model_is_gemini(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return _DummyResponse("plain text")

    monkeypatch.setattr(
        "app.core.llm.get_settings",
        lambda: Settings(current_model="gemini-2.0-flash", llm_base_url="", gemini_api_key="gemini-key"),
    )
    monkeypatch.setattr("app.core.llm.completion", fake_completion)

    result = invoke_llm(messages=[{"role": "user", "content": "hello"}])

    assert seen["api_key"] == "gemini-key"
    assert result == "plain text"
