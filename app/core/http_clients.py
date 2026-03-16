from __future__ import annotations

from dataclasses import dataclass

import httpx
from fastapi import Request


@dataclass
class HttpClientPool:
    geocoder: httpx.AsyncClient
    wfs: httpx.AsyncClient

    async def aclose(self) -> None:
        await self.geocoder.aclose()
        await self.wfs.aclose()


def create_http_client_pool() -> HttpClientPool:
    geocoder_client = httpx.AsyncClient(timeout=15.0)
    wfs_client = httpx.AsyncClient(timeout=20.0)
    return HttpClientPool(geocoder=geocoder_client, wfs=wfs_client)


def get_http_client_pool(request: Request) -> HttpClientPool:
    pool = getattr(request.app.state, "http_client_pool", None)
    if pool is None:
        raise RuntimeError("HTTP client pool is not initialized")
    return pool