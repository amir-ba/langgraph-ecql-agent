from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import httpx

from app.core.settings import get_settings

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_INITIAL_BACKOFF = 10.0  # seconds

_RATELIMIT_HEADERS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
)

_EMBEDDING_CACHE_FILE = Path(".embedding_cache.json")


# ---------------------------------------------------------------------------
# Disk-based embedding cache keyed by content hash
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning("Embedding cache corrupted, rebuilding: %s", exc)
    return {}


def _save_cache(path: Path, cache: dict[str, list[float]]) -> None:
    try:
        path.write_text(json.dumps(cache), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write embedding cache: %s", exc)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for multilingual text."""
    return max(1, len(text) // 4)


def _log_ratelimit_headers(response: httpx.Response, *, context: str = "") -> None:
    parts = [f"{h}: {response.headers[h]}" for h in _RATELIMIT_HEADERS if h in response.headers]
    if parts:
        logger.info("Embedding rate-limit headers%s — %s", f" ({context})" if context else "", " | ".join(parts))


async def _embed_batch(
    texts: list[str],
    *,
    url: str,
    model: str,
    headers: dict[str, str],
    client: httpx.AsyncClient,
) -> list[list[float]]:
    payload: dict[str, Any] = {"model": model, "input": texts}
    backoff = _INITIAL_BACKOFF
    for attempt in range(_MAX_RETRIES):
        response = await client.post(url, json=payload, headers=headers)
        _log_ratelimit_headers(response, context=f"attempt {attempt + 1}")
        if response.status_code == 429:
            logger.warning(
                "Embedding 429 response headers: %s",
                dict(response.headers),
            )
            try:
                logger.warning("Embedding 429 response body: %s", response.text)
            except Exception:
                pass
            retry_after = response.headers.get("retry-after")
            wait = float(retry_after) if retry_after else backoff
            logger.warning(
                "Embedding 429 rate-limited, retrying in %.1fs (attempt %d/%d)",
                wait, attempt + 1, _MAX_RETRIES,
            )
            await asyncio.sleep(wait)
            backoff *= 2
            continue
        response.raise_for_status()
        body = response.json()
        data = sorted(body["data"], key=lambda d: d["index"])
        return [item["embedding"] for item in data]
    # Final attempt — let the error propagate
    response = await client.post(url, json=payload, headers=headers)
    _log_ratelimit_headers(response, context="final attempt")
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

    # --- Resolve cached vs uncached texts ----------------------------------
    cache = _load_cache(_EMBEDDING_CACHE_FILE)
    text_hashes = [_text_hash(t) for t in texts]
    results: list[list[float] | None] = [cache.get(h) for h in text_hashes]

    uncached_indices = [i for i, emb in enumerate(results) if emb is None]
    uncached_texts = [texts[i] for i in uncached_indices]

    cached_count = len(texts) - len(uncached_texts)
    if cached_count:
        logger.info(
            "Embedding cache: %d/%d texts cached, %d to fetch",
            cached_count, len(texts), len(uncached_texts),
        )

    if not uncached_texts:
        return [r for r in results if r is not None]  # type: ignore[misc]

    # --- Fetch uncached embeddings with TPM pacing -------------------------
    rpm = max(1, settings.embedding_rpm)
    tpm = max(1, settings.embedding_tpm)
    min_interval = 60.0 / rpm

    client = http_client or httpx.AsyncClient(timeout=60.0)
    owns_client = http_client is None
    try:
        new_embeddings: list[list[float]] = []
        last_request_time = 0.0
        tokens_this_minute = 0
        minute_start = asyncio.get_event_loop().time()

        for i in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[i : i + batch_size]
            batch_tokens = sum(_estimate_tokens(t) for t in batch)

            now = asyncio.get_event_loop().time()

            # Reset token counter after 60s window
            if now - minute_start >= 60.0:
                tokens_this_minute = 0
                minute_start = now

            # TPM pacing: if this batch would exceed limit, wait for window reset
            if tokens_this_minute + batch_tokens > tpm:
                wait = 60.0 - (now - minute_start) + 1.0
                if wait > 0:
                    logger.info(
                        "TPM pacing: ~%d tokens used, batch needs ~%d, waiting %.1fs for window reset",
                        tokens_this_minute, batch_tokens, wait,
                    )
                    await asyncio.sleep(wait)
                    tokens_this_minute = 0
                    minute_start = asyncio.get_event_loop().time()

            # RPM pacing
            now = asyncio.get_event_loop().time()
            elapsed = now - last_request_time
            if last_request_time and elapsed < min_interval:
                wait = min_interval - elapsed
                logger.debug("Rate-limit pause %.1fs to stay under %d RPM", wait, rpm)
                await asyncio.sleep(wait)

            logger.debug(
                "Embedding batch %d-%d of %d (~%d tokens)",
                i + 1, i + len(batch), len(uncached_texts), batch_tokens,
            )
            last_request_time = asyncio.get_event_loop().time()
            batch_embeddings = await _embed_batch(
                batch, url=url, model=model, headers=headers, client=client
            )
            tokens_this_minute += batch_tokens
            new_embeddings.extend(batch_embeddings)
    finally:
        if owns_client:
            await client.aclose()

    # --- Merge results and update cache ------------------------------------
    for idx, emb in zip(uncached_indices, new_embeddings):
        results[idx] = emb
        cache[text_hashes[idx]] = emb

    _save_cache(_EMBEDDING_CACHE_FILE, cache)
    logger.info("Embedding cache updated: %d total entries", len(cache))

    return results  # type: ignore[return-value]
