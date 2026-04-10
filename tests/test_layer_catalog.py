import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.tools.layer_catalog import (
    LayerTranslationBatch,
    LayerTranslationRow,
    _fallback_translation_rows,
    _has_translation_occurred,
    _parse_full_rows_from_markdown,
    ensure_markdown_layer_catalog,
    parse_markdown_layer_catalog,
    render_catalog_rows_as_markdown,
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

    markdown, rows = asyncio.run(
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
    assert len(rows) == 1
    assert rows[0]["name"] == "city:roads"


def test_ensure_markdown_layer_catalog_reuses_fresh_file(tmp_path: Path) -> None:
    catalog_path = tmp_path / "layer_catalog.md"
    catalog_path.write_text(
        "# GeoServer Layer Catalog\n\n- **Layer ID:** `city:roads`\n  - **DE Title:** Strassen\n  - **EN Translation:** Roads\n  - **DE Abstract:** Strassennetz\n  - **EN Abstract:** Road network\n",
        encoding="utf-8",
    )

    markdown, rows = asyncio.run(
        ensure_markdown_layer_catalog(
            layers=[{"name": "city:lakes", "title": "Seen", "abstract": "Wasser"}],
            catalog_path=str(catalog_path),
            stale_after_hours=8,
            translator=_fake_translator,
        )
    )

    assert "city:roads" in markdown
    assert "city:lakes" not in markdown
    assert len(rows) == 1
    assert rows[0]["name"] == "city:roads"
    assert rows[0]["en_title"] == "Roads"


def test_ensure_markdown_layer_catalog_regenerates_stale_file(tmp_path: Path) -> None:
    catalog_path = tmp_path / "layer_catalog.md"
    catalog_path.write_text("# GeoServer Layer Catalog\n\n- **Layer ID:** `city:old`\n", encoding="utf-8")

    stale_time = datetime.now(tz=timezone.utc) - timedelta(hours=9)
    timestamp = stale_time.timestamp()
    catalog_path.touch()
    import os

    os.utime(catalog_path, (timestamp, timestamp))

    markdown, rows = asyncio.run(
        ensure_markdown_layer_catalog(
            layers=[{"name": "city:new", "title": "Neu", "abstract": "Beschreibung"}],
            catalog_path=str(catalog_path),
            stale_after_hours=8,
            translator=_fake_translator,
        )
    )

    assert "city:new" in markdown
    assert "city:old" not in markdown
    assert len(rows) == 1
    assert rows[0]["name"] == "city:new"


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


# ---------------------------------------------------------------------------
# Phase 1 — Translation Hardening
# ---------------------------------------------------------------------------


def test_has_translation_occurred_returns_false_when_en_and_de_are_identical() -> None:
    row = {
        "name": "city:roads",
        "de_title": "Strassen",
        "en_title": "Strassen",
        "de_abstract": "Strassennetz",
        "en_abstract": "Strassennetz",
    }
    assert _has_translation_occurred(row) is False


def test_has_translation_occurred_returns_true_when_titles_differ() -> None:
    row = {
        "name": "city:roads",
        "de_title": "Strassen",
        "en_title": "Roads",
        "de_abstract": "Strassennetz",
        "en_abstract": "Strassennetz",
    }
    assert _has_translation_occurred(row) is True


def test_has_translation_occurred_returns_true_when_abstracts_differ() -> None:
    row = {
        "name": "city:roads",
        "de_title": "Strassen",
        "en_title": "Strassen",
        "de_abstract": "Strassennetz",
        "en_abstract": "Road network",
    }
    assert _has_translation_occurred(row) is True


def test_fallback_translation_rows_marks_translation_verified_false() -> None:
    layers = [{"name": "city:roads", "title": "Strassen", "abstract": "Strassennetz"}]
    rows = _fallback_translation_rows(layers)
    assert len(rows) == 1
    assert rows[0]["translation_verified"] is False


def test_translate_layers_with_llm_fires_repair_pass_when_en_matches_de(monkeypatch) -> None:
    """When the first LLM call returns identical EN/DE, a second repair pass should fix them."""
    call_count = 0

    async def fake_ainvoke_llm(*, messages, output_schema):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: LLM returns identical EN/DE (no real translation)
            return LayerTranslationBatch(
                layers=[
                    LayerTranslationRow(
                        name="city:roads",
                        de_title="Strassen",
                        en_title="Strassen",
                        de_abstract="Strassennetz",
                        en_abstract="Strassennetz",
                        aliases=["strassen"],
                    )
                ]
            )
        # Second call (repair): returns fixed translations
        return LayerTranslationBatch(
            layers=[
                LayerTranslationRow(
                    name="city:roads",
                    de_title="Strassen",
                    en_title="Roads",
                    de_abstract="Strassennetz",
                    en_abstract="Road network",
                    aliases=["strassen", "roads"],
                )
            ]
        )

    monkeypatch.setattr("app.tools.layer_catalog.ainvoke_llm", fake_ainvoke_llm)

    layers = [{"name": "city:roads", "title": "Strassen", "abstract": "Strassennetz"}]
    translated = asyncio.run(translate_layers_with_llm(layers))

    assert call_count == 2, "A repair pass should have been triggered"
    assert translated[0]["en_title"] == "Roads"
    assert translated[0]["de_title"] == "Strassen"
    assert translated[0]["en_abstract"] == "Road network"
    assert translated[0]["translation_verified"] is True


def test_translate_layers_with_llm_skips_repair_when_translation_ok(monkeypatch) -> None:
    """When the first LLM call returns differentiated EN/DE, no repair pass fires."""
    call_count = 0

    async def fake_ainvoke_llm(*, messages, output_schema):
        nonlocal call_count
        call_count += 1
        return LayerTranslationBatch(
            layers=[
                LayerTranslationRow(
                    name="city:roads",
                    de_title="Strassen",
                    en_title="Roads",
                    de_abstract="Strassennetz",
                    en_abstract="Road network",
                    aliases=["strassen", "roads"],
                )
            ]
        )

    monkeypatch.setattr("app.tools.layer_catalog.ainvoke_llm", fake_ainvoke_llm)

    layers = [{"name": "city:roads", "title": "Strassen", "abstract": "Strassennetz"}]
    translated = asyncio.run(translate_layers_with_llm(layers))

    assert call_count == 1, "No repair should fire when translations already differ"
    assert translated[0]["en_title"] == "Roads"
    assert translated[0]["translation_verified"] is True


def test_parse_full_rows_from_markdown_extracts_bilingual_fields() -> None:
    markdown = """# GeoServer Layer Catalog

- **Layer ID:** `city:hospitals`
  - **DE Title:** Krankenhäuser
  - **EN Translation:** Hospitals
  - **DE Abstract:** Gesundheitseinrichtungen
  - **EN Abstract:** Healthcare facilities
  - **Aliases:** krankenhäuser, hospitals

- **Layer ID:** `city:roads`
  - **DE Title:** Strassen
  - **EN Translation:** Roads
  - **DE Abstract:** Strassennetz
  - **EN Abstract:** Road network
"""

    rows = _parse_full_rows_from_markdown(markdown)

    assert len(rows) == 2
    assert rows[0]["name"] == "city:hospitals"
    assert rows[0]["de_title"] == "Krankenhäuser"
    assert rows[0]["en_title"] == "Hospitals"
    assert rows[0]["de_abstract"] == "Gesundheitseinrichtungen"
    assert rows[0]["en_abstract"] == "Healthcare facilities"
    assert "krankenhäuser" in rows[0]["aliases"]
    assert rows[1]["name"] == "city:roads"
    assert rows[1]["en_title"] == "Roads"


def test_parse_full_rows_from_markdown_returns_empty_for_blank_input() -> None:
    assert _parse_full_rows_from_markdown("") == []
    assert _parse_full_rows_from_markdown("   ") == []


def test_render_catalog_rows_as_markdown_produces_labeled_output() -> None:
    rows = [
        {
            "name": "city:hospitals",
            "de_title": "Krankenhäuser",
            "en_title": "Hospitals",
            "de_abstract": "Gesundheitseinrichtungen",
            "en_abstract": "Healthcare facilities",
            "aliases": ["hospitals", "krankenhäuser"],
        }
    ]

    markdown = render_catalog_rows_as_markdown(rows)

    assert "city:hospitals" in markdown
    assert "Hospitals" in markdown
    assert "Healthcare facilities" in markdown
    assert "hospitals" in markdown


def test_ensure_markdown_layer_catalog_returns_rows_on_cache_hit(tmp_path: Path) -> None:
    catalog_path = tmp_path / "layer_catalog.md"
    catalog_path.write_text(
        "# GeoServer Layer Catalog\n\n"
        "- **Layer ID:** `city:roads`\n"
        "  - **DE Title:** Strassen\n"
        "  - **EN Translation:** Roads\n"
        "  - **DE Abstract:** Strassennetz\n"
        "  - **EN Abstract:** Road network\n"
        "  - **Aliases:** strassen, roads\n",
        encoding="utf-8",
    )

    markdown, rows = asyncio.run(
        ensure_markdown_layer_catalog(
            layers=[],
            catalog_path=str(catalog_path),
            stale_after_hours=8,
            translator=_fake_translator,
        )
    )

    assert "city:roads" in markdown
    assert len(rows) == 1
    assert rows[0]["name"] == "city:roads"
    assert rows[0]["en_title"] == "Roads"
    assert "roads" in rows[0]["aliases"]
