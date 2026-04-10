from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.settings import get_settings

logger = logging.getLogger(__name__)


async def _embed_batch(
    texts: list[str],
    *,
    url: str,
    model: str,
    headers: dict[str, str],
    client: httpx.AsyncClient,
) -> list[list[float]]:
    payload: dict[str, Any] = {"model": model, "input": texts}
    response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    body = response.json()
    data = sorted(body["data"], key=lambda d: d["index"])
    return [item["embedding"] for item in data]


async def get_embeddings(
    texts: list[str],
    *,
    http_client: httpx.AsyncClient | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    settings = get_settings()
    base_url = settings.llm_base_url.strip().rstrip("/")
    api_key = settings.llm_api_key.strip()
    model = settings.embedding_model
    batch_size = max(1, settings.embedding_batch_size)

    if not base_url:
        raise ValueError("LLM_BASE_URL must be set for embedding requests")
    if not model.strip():
        raise ValueError("EMBEDDING_MODEL must be set for embedding requests")

    url = f"{base_url}/embeddings"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    client = http_client or httpx.AsyncClient(timeout=60.0)
    owns_client = http_client is None
    try:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug("Embedding batch %d-%d of %d", i + 1, i + len(batch), len(texts))
            batch_embeddings = await _embed_batch(
                batch, url=url, model=model, headers=headers, client=client
            )
            all_embeddings.extend(batch_embeddings)
        return all_embeddings
    finally:
        if owns_client:
            await client.aclose()
