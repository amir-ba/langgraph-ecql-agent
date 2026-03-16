import asyncio
import json

import httpx
import pytest

from app.core.settings import Settings
from app.tools.geocoder import GeocoderClient


def test_geocoder_uses_oauth_token_for_fulltext_forwarding() -> None:
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)

        if request.url.path == "/oauth/token":
            body = dict(item.split("=") for item in request.content.decode().split("&"))
            assert body["grant_type"] == "client_credentials"
            assert body["client_id"] == "client-id"
            assert body["client_secret"] == "client-secret"
            return httpx.Response(
                200,
                json={
                    "access_token": "abc123",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )

        if request.url.path == "/geo/geocoder/v1/forward/fulltext":
            assert request.headers["Authorization"] == "Bearer abc123"
            assert request.url.params["query"] == "bonn friedrich ebert allee 140"
            return httpx.Response(
                200,
                json={
                    "responseHeader": {"status": 0},
                    "Result": [
                        {
                            "Coordinate": "50.7192,7.1220",
                            "Address": {"Label": "53113 Bonn, Friedrich-Ebert-Allee 140"},
                        }
                    ],
                },
            )

        raise AssertionError(f"Unexpected request URL: {request.url}")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)

    settings = Settings(
        geocoder_api_url="https://example.test/geo",
        geocoder_token_url="https://example.test/oauth/token",
        geocoder_client_id="client-id",
        geocoder_client_secret="client-secret",
    )

    geocoder = GeocoderClient(settings=settings, http_client=client)
    payload = asyncio.run(geocoder.forward_fulltext(query="bonn friedrich ebert allee 140"))

    assert payload["Result"][0]["Address"]["Label"].startswith("53113 Bonn")
    assert len(requests_seen) == 2


def test_geocoder_raises_for_non_zero_response_header_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth/token":
            return httpx.Response(200, json={"access_token": "abc123", "expires_in": 3600})
        return httpx.Response(200, json={"responseHeader": {"status": 4}, "error": {"msg": "bad query"}})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)

    settings = Settings(
        geocoder_api_url="https://example.test/geo",
        geocoder_token_url="https://example.test/oauth/token",
        geocoder_client_id="client-id",
        geocoder_client_secret="client-secret",
    )

    geocoder = GeocoderClient(settings=settings, http_client=client)

    with pytest.raises(ValueError, match="status=4"):
        asyncio.run(geocoder.reverse(coord="50.7,7.1,4326"))
