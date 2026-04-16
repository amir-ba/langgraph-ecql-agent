import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import app.tools.embedding_client as ec
from app.tools.embedding_client import get_embeddings


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the embedding cache at a temp file so tests don't interfere."""
    monkeypatch.setattr(ec, "_EMBEDDING_CACHE_FILE", tmp_path / ".embedding_cache.json")


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


def test_get_embeddings_retries_on_429(monkeypatch) -> None:
    """429 responses should be retried with backoff until success."""
    fake_vectors = [[0.1, 0.2]]
    call_count = 0

    async def fake_post(self, url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return httpx.Response(429, json={"error": "rate limited"}, request=_fake_request())
        return httpx.Response(200, json=_make_embedding_response(fake_vectors), request=_fake_request())

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setenv("LLM_BASE_URL", "https://my-llm-provider.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key-123")
    monkeypatch.setenv("EMBEDDING_MODEL", "jina-embeddings-v2-base-de")

    from app.core.settings import get_settings
    get_settings.cache_clear()

    # Patch sleep to avoid real delays in tests
    import app.tools.embedding_client as ec
    monkeypatch.setattr(ec.asyncio, "sleep", AsyncMock())

    result = asyncio.run(get_embeddings(["test"]))
    assert result == [[0.1, 0.2]]
    assert call_count == 3  # 2 retries + 1 success

    get_settings.cache_clear()


def test_get_embeddings_respects_retry_after_header(monkeypatch) -> None:
    """Should use Retry-After header value when present."""
    fake_vectors = [[0.5]]
    call_count = 0
    sleep_values: list[float] = []

    async def fake_post(self, url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                429,
                json={"error": "rate limited"},
                headers={"retry-after": "2"},
                request=_fake_request(),
            )
        return httpx.Response(200, json=_make_embedding_response(fake_vectors), request=_fake_request())

    async def fake_sleep(seconds):
        sleep_values.append(seconds)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setenv("LLM_BASE_URL", "https://my-llm-provider.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key-123")
    monkeypatch.setenv("EMBEDDING_MODEL", "jina-embeddings-v2-base-de")

    from app.core.settings import get_settings
    get_settings.cache_clear()

    import app.tools.embedding_client as ec
    monkeypatch.setattr(ec.asyncio, "sleep", fake_sleep)

    result = asyncio.run(get_embeddings(["test"]))
    assert result == [[0.5]]
    assert sleep_values == [2.0]

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


def test_get_embeddings_uses_disk_cache(monkeypatch, tmp_path) -> None:
    """Second call with same texts should use cache and not call the API."""
    fake_vectors = [[0.1, 0.2]]
    call_count = 0

    async def fake_post(self, url, **kwargs):
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json=_make_embedding_response(fake_vectors), request=_fake_request())

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setenv("LLM_BASE_URL", "https://my-llm-provider.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "jina-embeddings-v2-base-de")

    from app.core.settings import get_settings
    get_settings.cache_clear()

    # First call: should hit the API
    result1 = asyncio.run(get_embeddings(["hello"]))
    assert result1 == [[0.1, 0.2]]
    assert call_count == 1

    # Second call: same text, should use cache
    result2 = asyncio.run(get_embeddings(["hello"]))
    assert result2 == [[0.1, 0.2]]
    assert call_count == 1  # no additional API call

    get_settings.cache_clear()


def test_get_embeddings_partial_cache_hit(monkeypatch, tmp_path) -> None:
    """When some texts are cached and some are not, only uncached ones are fetched."""
    call_inputs: list[list[str]] = []

    async def fake_post(self, url, **kwargs):
        body = kwargs.get("json", {})
        call_inputs.append(body["input"])
        vecs = [[float(i)] for i in range(len(body["input"]))]
        return httpx.Response(200, json=_make_embedding_response(vecs), request=_fake_request())

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setenv("LLM_BASE_URL", "https://my-llm-provider.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "jina-embeddings-v2-base-de")

    from app.core.settings import get_settings
    get_settings.cache_clear()

    # First call: cache "hello"
    asyncio.run(get_embeddings(["hello"]))
    assert len(call_inputs) == 1

    # Second call: "hello" is cached, only "world" should be fetched
    asyncio.run(get_embeddings(["hello", "world"]))
    assert len(call_inputs) == 2
    assert call_inputs[1] == ["world"]

    get_settings.cache_clear()
