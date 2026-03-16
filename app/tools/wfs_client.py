from __future__ import annotations

import asyncio
import time
from typing import Any
from xml.etree import ElementTree as ET

import httpx
from owslib.wfs import WebFeatureService


_CAPABILITIES_CACHE_TTL_SECONDS = 12 * 60 * 60
_CapabilitiesCacheKey = tuple[str, float, str | None, str | None]
_capabilities_cache: dict[_CapabilitiesCacheKey, tuple[float, list[dict[str, str]]]] = {}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _build_auth(username: str | None, password: str | None) -> tuple[str, str] | None:
    if not username or not password:
        return None
    return (username, password)


def _discovery_cache_key(
    wfs_url: str,
    timeout: float,
    username: str | None,
    password: str | None,
) -> _CapabilitiesCacheKey:
    return (wfs_url, timeout, username, password)


def _cache_copy(layers: list[dict[str, str]]) -> list[dict[str, str]]:
    return [dict(layer) for layer in layers]


def clear_discovery_cache() -> None:
    _capabilities_cache.clear()


def _discover_layers_from_xml(xml_text: str) -> list[dict[str, str]]:
    root = ET.fromstring(xml_text)
    layers: list[dict[str, str]] = []

    for node in root.iter():
        if _local_name(node.tag) != "FeatureType":
            continue

        name = ""
        title = ""
        abstract = ""

        for child in list(node):
            child_name = _local_name(child.tag)
            text = (child.text or "").strip()
            if child_name == "Name":
                name = text
            elif child_name == "Title":
                title = text
            elif child_name == "Abstract":
                abstract = text

        if name:
            layers.append({"name": name, "title": title, "abstract": abstract})

    return layers


def _describe_feature_type_from_xml(xml_text: str) -> dict[str, Any]:
    root = ET.fromstring(xml_text)
    attributes: dict[str, str] = {}
    geometry_column = ""

    for node in root.iter():
        if _local_name(node.tag) != "element":
            continue

        name = node.attrib.get("name", "")
        if not name:
            continue

        attr_type = node.attrib.get("type", "")
        attributes[name] = attr_type

        lowered_type = attr_type.lower()
        lowered_name = name.lower()
        if not geometry_column and (
            "gml:" in lowered_type
            or "geometry" in lowered_type
            or lowered_name in {"geom", "the_geom", "shape"}
        ):
            geometry_column = name

    return {"attributes": attributes, "geometry_column": geometry_column}


def _discover_layers_with_owslib(
    wfs_url: str,
    timeout: float,
    username: str | None,
    password: str | None,
) -> list[dict[str, str]]:
    service = WebFeatureService(
        url=wfs_url,
        version="2.0.0",
        timeout=int(timeout),
        username=username,
        password=password,
    )
    layers: list[dict[str, str]] = []
    for layer_name, metadata in service.contents.items():
        title = str(getattr(metadata, "title", "") or "")
        abstract = str(getattr(metadata, "abstract", "") or "")
        layers.append({"name": str(layer_name), "title": title, "abstract": abstract})
    return layers


def _describe_feature_type_with_owslib(
    wfs_url: str,
    type_name: str,
    timeout: float,
    username: str | None,
    password: str | None,
) -> dict[str, Any]:
    service = WebFeatureService(
        url=wfs_url,
        version="2.0.0",
        timeout=int(timeout),
        username=username,
        password=password,
    )
    schema = service.get_schema(type_name)

    properties = schema.get("properties") or {}
    attributes: dict[str, str] = {
        str(name): str(attr_type)
        for name, attr_type in properties.items()
    }

    geometry_column = str(schema.get("geometry_column") or "")
    if not geometry_column:
        for name, attr_type in attributes.items():
            lowered_type = attr_type.lower()
            lowered_name = name.lower()
            if (
                "gml:" in lowered_type
                or "geometry" in lowered_type
                or lowered_name in {"geom", "the_geom", "shape"}
            ):
                geometry_column = name
                break

    return {"attributes": attributes, "geometry_column": geometry_column}


async def discover_layers(
    wfs_url: str,
    http_client: httpx.AsyncClient | None = None,
    timeout: float = 15.0,
    username: str | None = None,
    password: str | None = None,
) -> list[dict[str, str]]:
    if http_client is None:
        cache_key = _discovery_cache_key(
            wfs_url=wfs_url,
            timeout=timeout,
            username=username,
            password=password,
        )
        now = time.time()
        cached = _capabilities_cache.get(cache_key)
        if cached and now < cached[0]:
            return _cache_copy(cached[1])

    if http_client is None:
        try:
            layers = await asyncio.to_thread(
                _discover_layers_with_owslib,
                wfs_url,
                timeout,
                username,
                password,
            )
            _capabilities_cache[cache_key] = (now + _CAPABILITIES_CACHE_TTL_SECONDS, _cache_copy(layers))
            return layers
        except Exception:
            pass

    if http_client is None:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                wfs_url,
                params={"service": "WFS", "request": "GetCapabilities"},
                auth=_build_auth(username, password),
            )
    else:
        response = await http_client.get(
            wfs_url,
            params={"service": "WFS", "request": "GetCapabilities"},
            auth=_build_auth(username, password),
        )
    response.raise_for_status()
    layers = _discover_layers_from_xml(response.text)
    if http_client is None:
        _capabilities_cache[cache_key] = (now + _CAPABILITIES_CACHE_TTL_SECONDS, _cache_copy(layers))
    return layers


def filter_layers_by_subject(
    layers: list[dict[str, str]],
    layer_subject: str | None,
) -> list[dict[str, str]]:
    if not layer_subject:
        return layers

    subject = layer_subject.strip().lower()
    if not subject:
        return layers

    filtered: list[dict[str, str]] = []
    for layer in layers:
        combined = " ".join(
            [
                str(layer.get("name", "")),
                str(layer.get("title", "")),
                str(layer.get("abstract", "")),
            ]
        ).lower()
        if subject in combined:
            filtered.append(layer)

    return filtered


async def describe_feature_type(
    wfs_url: str,
    type_name: str,
    http_client: httpx.AsyncClient | None = None,
    timeout: float = 15.0,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Any]:
    if http_client is None:
        try:
            return await asyncio.to_thread(
                _describe_feature_type_with_owslib,
                wfs_url,
                type_name,
                timeout,
                username,
                password,
            )
        except Exception:
            pass

    if http_client is None:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                wfs_url,
                params={
                    "service": "WFS",
                    "version": "2.0.0",
                    "request": "DescribeFeatureType",
                    "typeNames": type_name,
                },
                auth=_build_auth(username, password),
            )
    else:
        response = await http_client.get(
            wfs_url,
            params={
                "service": "WFS",
                "version": "2.0.0",
                "request": "DescribeFeatureType",
                "typeNames": type_name,
            },
            auth=_build_auth(username, password),
        )
    response.raise_for_status()
    return _describe_feature_type_from_xml(response.text)


async def get_layer_schema(
    wfs_url: str,
    type_name: str,
    http_client: httpx.AsyncClient | None = None,
    timeout: float = 15.0,
    username: str | None = None,
    password: str | None = None,
) -> tuple[dict[str, str], str]:
    schema = await describe_feature_type(
        wfs_url=wfs_url,
        type_name=type_name,
        http_client=http_client,
        timeout=timeout,
        username=username,
        password=password,
    )
    return schema["attributes"], schema["geometry_column"]


async def execute_wfs_query(
    wfs_url: str,
    type_name: str,
    cql_filter: str,
    count: int = 1000,
    srs_name: str = "EPSG:3857",
    output_format: str = "application/json",
    http_client: httpx.AsyncClient | None = None,
    timeout: float = 20.0,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Any]:
    safe_count = min(max(1, count), 1000)

    if http_client is None:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                wfs_url,
                params={
                    "service": "WFS",
                    "version": "2.0.0",
                    "request": "GetFeature",
                    "typeNames": type_name,
                    "srsName": srs_name,
                    "outputFormat": output_format,
                    "cql_filter": cql_filter,
                    "count": safe_count,
                },
                auth=_build_auth(username, password),
            )
    else:
        response = await http_client.get(
            wfs_url,
            params={
                "service": "WFS",
                "version": "2.0.0",
                "request": "GetFeature",
                "typeNames": type_name,
                "srsName": srs_name,
                "outputFormat": output_format,
                "cql_filter": cql_filter,
                "count": safe_count,
            },
            auth=_build_auth(username, password),
        )
    response.raise_for_status()
    return response.json()
