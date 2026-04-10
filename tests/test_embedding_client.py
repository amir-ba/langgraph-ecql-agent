import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.tools.embedding_client import get_embeddings


def _make_embedding_response(embeddings: list[list[float]]) -> dict:
    return {
        "data": [{"embedding": emb, "index": i} for i, emb in enumerate(embeddings)],
        "model": "jina-embeddings-v2-base-de",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }


def _fake_request() -> httpx.Request:
    return httpx.Request("POST", "https://example.com/embeddings")


def test_get_embeddings_returns_vectors(monkeypatch) -> None:
    fake_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    async def fake_post(self, url, **kwargs):
        body = kwargs.get("json", {})
        assert body["model"] == "jina-embeddings-v2-base-de"
        assert body["input"] == ["hello world", "flood zones"]
        return httpx.Response(
            200,
            json=_make_embedding_response(fake_vectors),
            request=_fake_request(),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setenv("LLM_BASE_URL", "https://my-llm-provider.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key-123")
    monkeypatch.setenv("EMBEDDING_MODEL", "jina-embeddings-v2-base-de")

    # Clear settings cache
    from app.core.settings import get_settings
    get_settings.cache_clear()

    result = asyncio.run(get_embeddings(["hello world", "flood zones"]))

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]

    get_settings.cache_clear()


def test_get_embeddings_empty_input() -> None:
    result = asyncio.run(get_embeddings([]))
    assert result == []


def test_get_embeddings_requires_llm_base_url(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_API_KEY", "test-key-123")
    monkeypatch.setenv("EMBEDDING_MODEL", "jina-embeddings-v2-base-de")

    from app.core.settings import get_settings
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        asyncio.run(get_embeddings(["test"]))

    get_settings.cache_clear()


def test_get_embeddings_raises_on_http_error(monkeypatch) -> None:
    async def fake_post(self, url, **kwargs):
        return httpx.Response(500, json={"error": "server down"}, request=_fake_request())

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setenv("LLM_BASE_URL", "https://my-llm-provider.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key-123")
    monkeypatch.setenv("EMBEDDING_MODEL", "jina-embeddings-v2-base-de")

    from app.core.settings import get_settings
    get_settings.cache_clear()

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(get_embeddings(["test"]))

    get_settings.cache_clear()


def test_get_embeddings_uses_custom_http_client(monkeypatch) -> None:
    fake_vectors = [[0.9, 0.8, 0.7]]
    call_log: list[str] = []

    async def fake_post(self, url, **kwargs):
        call_log.append(url)
        return httpx.Response(200, json=_make_embedding_response(fake_vectors), request=_fake_request())

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setenv("LLM_BASE_URL", "https://my-provider.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("EMBEDDING_MODEL", "jina-embeddings-v2-base-de")

    from app.core.settings import get_settings
    get_settings.cache_clear()

    client = httpx.AsyncClient()
    result = asyncio.run(get_embeddings(["test"], http_client=client))
    assert result == [[0.9, 0.8, 0.7]]
    assert any("embeddings" in url for url in call_log)

    get_settings.cache_clear()
