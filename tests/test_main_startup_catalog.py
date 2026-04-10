import asyncio
from pathlib import Path
from typing import Any, cast

import main


class _DummyPool:
    def __init__(self) -> None:
        self.wfs = object()
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _DummyState:
    pass


class _DummyApp:
    def __init__(self) -> None:
        self.state = _DummyState()


def test_lifespan_initializes_markdown_catalog(monkeypatch, tmp_path: Path) -> None:
    dummy_pool = _DummyPool()
    called: dict[str, object] = {}

    monkeypatch.setattr(main, "create_http_client_pool", lambda: dummy_pool)
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "geoserver_wfs_url": "https://wfs",
                "geoserver_wfs_username": "demo-user",
                "geoserver_wfs_password": "demo-pass",
                "layer_catalog_markdown_path": str(tmp_path / "layer_catalog.md"),
                "layer_catalog_stale_after_hours": 8,
            },
        )(),
    )

    async def fake_discover_layers(*, wfs_url: str, http_client=None, username: str, password: str):
        assert wfs_url == "https://wfs"
        assert username == "demo-user"
        assert password == "demo-pass"
        assert http_client is dummy_pool.wfs
        return [{"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare"}]

    async def fake_ensure_markdown_layer_catalog(*, layers, catalog_path: str, stale_after_hours: int):
        called["layers"] = layers
        called["catalog_path"] = catalog_path
        called["stale_after_hours"] = stale_after_hours
        return "# GeoServer Layer Catalog", []

    monkeypatch.setattr(main, "discover_layers", fake_discover_layers)
    monkeypatch.setattr(main, "ensure_markdown_layer_catalog", fake_ensure_markdown_layer_catalog)

    app = _DummyApp()

    async def _run() -> None:
        async with main.lifespan(cast(Any, app)):
            assert hasattr(app.state, "http_client_pool")

    asyncio.run(_run())

    assert called["layers"] == [{"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare"}]
    assert called["stale_after_hours"] == 8
    assert str(called["catalog_path"]).endswith("layer_catalog.md")
    assert dummy_pool.closed is True


def test_lifespan_continues_when_catalog_initialization_fails(monkeypatch) -> None:
    dummy_pool = _DummyPool()

    monkeypatch.setattr(main, "create_http_client_pool", lambda: dummy_pool)
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "geoserver_wfs_url": "https://wfs",
                "geoserver_wfs_username": "demo-user",
                "geoserver_wfs_password": "demo-pass",
                "layer_catalog_markdown_path": "layer_catalog.md",
                "layer_catalog_stale_after_hours": 8,
            },
        )(),
    )

    async def fake_discover_layers(*, wfs_url: str, http_client=None, username: str, password: str):
        raise RuntimeError("wfs unavailable")

    monkeypatch.setattr(main, "discover_layers", fake_discover_layers)

    app = _DummyApp()

    async def _run() -> None:
        async with main.lifespan(cast(Any, app)):
            assert hasattr(app.state, "http_client_pool")

    asyncio.run(_run())

    assert dummy_pool.closed is True


def test_lifespan_indexes_vector_store_on_startup(monkeypatch, tmp_path: Path) -> None:
    dummy_pool = _DummyPool()
    vector_indexed: dict[str, Any] = {}

    monkeypatch.setattr(main, "create_http_client_pool", lambda: dummy_pool)
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "geoserver_wfs_url": "https://wfs",
                "geoserver_wfs_username": "",
                "geoserver_wfs_password": "",
                "layer_catalog_markdown_path": str(tmp_path / "layer_catalog.md"),
                "layer_catalog_stale_after_hours": 8,
            },
        )(),
    )

    fake_layers = [
        {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare"},
        {"name": "city:schools", "title": "Schools", "abstract": "Education"},
    ]
    fake_rows = [
        {"name": "city:hospitals", "de_title": "Krankenhäuser", "en_title": "Hospitals", "de_abstract": "Gesundheit", "en_abstract": "Healthcare", "aliases": []},
        {"name": "city:schools", "de_title": "Schulen", "en_title": "Schools", "de_abstract": "Bildung", "en_abstract": "Education", "aliases": []},
    ]

    async def fake_discover_layers(*, wfs_url, http_client=None, username, password):
        return fake_layers

    async def fake_ensure_markdown_layer_catalog(*, layers, catalog_path, stale_after_hours):
        return "# Catalog", fake_rows

    async def fake_index_layers(self, layers, catalog_rows, embed_fn):
        vector_indexed["layers"] = layers
        vector_indexed["catalog_rows"] = catalog_rows

    monkeypatch.setattr(main, "discover_layers", fake_discover_layers)
    monkeypatch.setattr(main, "ensure_markdown_layer_catalog", fake_ensure_markdown_layer_catalog)

    from app.tools.vector_store import LayerVectorStore
    monkeypatch.setattr(LayerVectorStore, "index_layers", fake_index_layers)

    app = _DummyApp()

    async def _run() -> None:
        async with main.lifespan(cast(Any, app)):
            pass

    asyncio.run(_run())

    assert vector_indexed["layers"] == fake_layers
    assert vector_indexed["catalog_rows"] == fake_rows
    assert dummy_pool.closed is True
