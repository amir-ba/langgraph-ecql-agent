import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.tools.layer_catalog import (
    LayerTranslationBatch,
    LayerTranslationRow,
    ensure_markdown_layer_catalog,
    parse_markdown_layer_catalog,
    translate_layers_with_llm,
)


async def _fake_translator(layers: list[dict[str, str]]) -> list[dict[str, str | list[str]]]:
    rows: list[dict[str, str | list[str]]] = []
    for layer in layers:
        rows.append(
            {
                "name": layer["name"],
                "de_title": layer["title"],
                "en_title": f"EN {layer['title']}",
                "de_abstract": layer["abstract"],
                "en_abstract": f"EN {layer['abstract']}",
                "aliases": [layer["title"].lower()],
            }
        )
    return rows


def test_ensure_markdown_layer_catalog_creates_file_when_missing(tmp_path: Path) -> None:
    catalog_path = tmp_path / "layer_catalog.md"
    layers = [{"name": "city:roads", "title": "Strassen", "abstract": "Strassennetz"}]

    markdown = asyncio.run(
        ensure_markdown_layer_catalog(
            layers=layers,
            catalog_path=str(catalog_path),
            stale_after_hours=8,
            translator=_fake_translator,
        )
    )

    assert catalog_path.exists()
    assert "# GeoServer Layer Catalog" in markdown
    assert "city:roads" in markdown
    assert "EN Strassen" in markdown


def test_ensure_markdown_layer_catalog_reuses_fresh_file(tmp_path: Path) -> None:
    catalog_path = tmp_path / "layer_catalog.md"
    catalog_path.write_text("# GeoServer Layer Catalog\n\n- **Layer ID:** `city:roads`\n", encoding="utf-8")

    markdown = asyncio.run(
        ensure_markdown_layer_catalog(
            layers=[{"name": "city:lakes", "title": "Seen", "abstract": "Wasser"}],
            catalog_path=str(catalog_path),
            stale_after_hours=8,
            translator=_fake_translator,
        )
    )

    assert "city:roads" in markdown
    assert "city:lakes" not in markdown


def test_ensure_markdown_layer_catalog_regenerates_stale_file(tmp_path: Path) -> None:
    catalog_path = tmp_path / "layer_catalog.md"
    catalog_path.write_text("# GeoServer Layer Catalog\n\n- **Layer ID:** `city:old`\n", encoding="utf-8")

    stale_time = datetime.now(tz=timezone.utc) - timedelta(hours=9)
    timestamp = stale_time.timestamp()
    catalog_path.touch()
    import os

    os.utime(catalog_path, (timestamp, timestamp))

    markdown = asyncio.run(
        ensure_markdown_layer_catalog(
            layers=[{"name": "city:new", "title": "Neu", "abstract": "Beschreibung"}],
            catalog_path=str(catalog_path),
            stale_after_hours=8,
            translator=_fake_translator,
        )
    )

    assert "city:new" in markdown
    assert "city:old" not in markdown


def test_parse_markdown_layer_catalog_extracts_layer_fields() -> None:
    markdown = """# GeoServer Layer Catalog

- **Layer ID:** `basis:fca_kgs2_bundesland`
  - **DE Title:** Bundeslander
  - **EN Translation:** Federal States
  - **DE Abstract:** Ubersicht uber Bundeslander.
  - **EN Abstract:** Overview of federal states.
  - **Aliases:** bundeslander, federal states, provinces
"""

    layers = parse_markdown_layer_catalog(markdown)

    assert len(layers) == 1
    assert layers[0]["name"] == "basis:fca_kgs2_bundesland"
    assert layers[0]["title"] == "Federal States"
    assert "federal states" in layers[0]["abstract"].lower()


def test_translate_layers_with_llm_keeps_input_cardinality_on_partial_llm_result(monkeypatch) -> None:
    async def fake_ainvoke_llm(*, messages, output_schema):
        _ = messages
        _ = output_schema
        return LayerTranslationBatch(
            layers=[
                LayerTranslationRow(
                    name="city:roads",
                    de_title="Strassen",
                    en_title="Roads",
                    de_abstract="Strassennetz",
                    en_abstract="Road network",
                    aliases=["roads"],
                )
            ]
        )

    monkeypatch.setattr("app.tools.layer_catalog.ainvoke_llm", fake_ainvoke_llm)

    layers = [
        {"name": "city:roads", "title": "Strassen", "abstract": "Strassennetz"},
        {"name": "city:lakes", "title": "Seen", "abstract": "Wasser"},
    ]

    translated = asyncio.run(translate_layers_with_llm(layers))

    assert len(translated) == len(layers)
    assert translated[0]["name"] == "city:roads"
    assert translated[0]["en_title"] == "Roads"
    assert translated[1]["name"] == "city:lakes"
    assert translated[1]["en_title"] == "Seen"
