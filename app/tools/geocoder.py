from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.core.settings import Settings, get_settings


@dataclass
class OAuthToken:
    value: str
    expires_at: datetime


class OAuthClientCredentialsProvider:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._http_client = http_client
        self._token: OAuthToken | None = None

    async def get_access_token(self) -> str:
        now = datetime.now(timezone.utc)
        if self._token and now < self._token.expires_at:
            return self._token.value

        token_url = self._settings.geocoder_token_url.strip()
        if not token_url:
            raise ValueError("Missing GEOCODER_TOKEN_URL for OAuth client credentials flow")

        response = await self._http_client.post(
            token_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": self._settings.geocoder_client_id,
                "client_secret": self._settings.geocoder_client_secret,
                **({"scope": self._settings.geocoder_scope} if self._settings.geocoder_scope else {}),
            },
        )
        response.raise_for_status()

        payload = response.json()
        access_token = payload.get("access_token")
        if not access_token:
            raise ValueError("OAuth token response missing access_token")

        expires_in = int(payload.get("expires_in", 3600))
        # Refresh slightly before expiry to avoid race conditions during long operations.
        expires_at = now + timedelta(seconds=max(30, expires_in - 30))

        self._token = OAuthToken(value=access_token, expires_at=expires_at)
        return access_token


class GeocoderClient:
    def __init__(self, settings: Settings | None = None, http_client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings or get_settings()
        self._http_client = http_client or httpx.AsyncClient(timeout=15.0)
        self._oauth = OAuthClientCredentialsProvider(settings=self._settings, http_client=self._http_client)

    async def forward(
        self,
        zip_code: str | None = None,
        city: str | None = None,
        street: str | None = None,
        house_number: str | None = None,
        max_results: int = 1,
        epsg: int = 4326,
    ) -> dict[str, Any]:
        params = {
            "zip": zip_code,
            "city": city,
            "street": street,
            "hsnr": house_number,
            "maxResults": max_results,
            "epsg": epsg,
        }
        return await self._get("/geocoder/v1/forward", params=params)

    async def forward_fulltext(self, query: str, max_results: int = 5, epsg: int = 4326) -> dict[str, Any]:
        return await self._get(
            "/geocoder/v1/forward/fulltext",
            params={"query": query, "maxResults": max_results, "epsg": epsg},
        )

    async def reverse(self, coord: str, epsg: int | None = None, max_dist: float = 0.05) -> dict[str, Any]:
        params: dict[str, Any] = {"coord": coord, "maxDist": max_dist}
        if epsg is not None:
            params["epsg"] = epsg
        return await self._get("/geocoder/v1/reverse", params=params)

    async def suggest(self, query: str, max_results: int = 5) -> dict[str, Any]:
        return await self._get("/geocoder/v1/suggest", params={"query": query, "maxResults": max_results})

    async def select(self, address_id: str, epsg: int = 4326) -> dict[str, Any]:
        return await self._get(f"/geocoder/v1/select/{address_id}", params={"epsg": epsg})

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        access_token = await self._oauth.get_access_token()
        base_url = self._settings.geocoder_api_url.rstrip("/")
        response = await self._http_client.get(
            f"{base_url}{path}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={k: v for k, v in params.items() if v is not None},
        )
        response.raise_for_status()
        payload = response.json()
        self._ensure_ok(payload)
        return payload

    @staticmethod
    def _ensure_ok(payload: dict[str, Any]) -> None:
        status = payload.get("responseHeader", {}).get("status")
        if status is None:
            return
        if int(status) != 0:
            raise ValueError(f"Geocoder responseHeader.status={status}: {payload.get('error')}")
