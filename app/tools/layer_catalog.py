from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, Field

from app.core.llm import ainvoke_llm

logger = logging.getLogger(__name__)

_TRANSLATION_BATCH_SIZE = 20
_TRANSLATION_MAX_CONCURRENCY = 3


LAYER_NAMING_CONVENTION = """\
Naming conventions for GeoServer feature classes:
- fc*  = feature class (prefix for all feature classes)
  - fca_* = feature class area (polygon), e.g. fca_admin_units
  - fcl_* = feature class line, e.g. fcl_fahrbahnbegrenzungslinie
  - fcp_* = feature class point, e.g. fcp_grenzpunkt
- fv*  = feature view (database view with geographic features)
  - fva_* = feature view area (polygon), e.g. fva_kvz_nahbereich
  - fvl_* = feature view line, e.g. fvl_trenches
  - fvp_* = feature view point, e.g. fvp_pictures_label
- mfv* = materialized feature view
  - mfva_* = materialized view area (polygon)
  - mfcl_* = materialized view line
  - mfcp_* = materialized view point
"""

_PREFIX_TO_GEOMETRY_TYPE: dict[str, str] = {
    "mfva_": "polygon",
    "mfcl_": "line",
    "mfcp_": "point",
    "fva_": "polygon",
    "fvl_": "line",
    "fvp_": "point",
    "fca_": "polygon",
    "fcl_": "line",
    "fcp_": "point",
}


def parse_geometry_type_from_name(layer_name: str) -> str | None:
    """Derive geometry type (polygon/line/point) from the layer name prefix convention."""
    # Strip optional workspace prefix (e.g. "city:fca_admin_units" -> "fca_admin_units")
    bare = layer_name.split(":")[-1] if ":" in layer_name else layer_name
    bare_lower = bare.lower()
    for prefix, geom_type in _PREFIX_TO_GEOMETRY_TYPE.items():
        if bare_lower.startswith(prefix):
            return geom_type
    return None


LayerTranslator = Callable[[list[dict[str, str]]], Awaitable[list[dict[str, str | list[str]]]]]


class LayerTranslationRow(BaseModel):
    name: str
    de_title: str = ""
    en_title: str = ""
    de_abstract: str = ""
    en_abstract: str = ""
    aliases: list[str] = Field(default_factory=list)


class LayerTranslationBatch(BaseModel):
    layers: list[LayerTranslationRow] = Field(default_factory=list)


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _has_translation_occurred(row: dict[str, str | list[str]]) -> bool:
    de_title = _normalize_text(str(row.get("de_title", "")))
    en_title = _normalize_text(str(row.get("en_title", "")))
    de_abstract = _normalize_text(str(row.get("de_abstract", "")))
    en_abstract = _normalize_text(str(row.get("en_abstract", "")))
    return de_title != en_title or de_abstract != en_abstract


def _fallback_translation_rows(layers: list[dict[str, str]]) -> list[dict[str, str | list[str]]]:
    rows: list[dict[str, str | list[str]]] = []
    for layer in layers:
        title = _normalize_text(str(layer.get("title", "")))
        abstract = _normalize_text(str(layer.get("abstract", "")))
        name = _normalize_text(str(layer.get("name", "")))
        rows.append(
            {
                "name": name,
                "de_title": title,
                "en_title": title,
                "de_abstract": abstract,
                "en_abstract": abstract,
                "aliases": [title.lower()] if title else [],
                "translation_verified": False,
            }
        )
    return rows


_TRANSLATION_SYSTEM_PROMPT = (
    "You enrich GeoServer layer metadata. For each input layer, return bilingual metadata with "
    "German and English text and useful aliases. Keep layer names unchanged. Return JSON only.\n\n"
    "IMPORTANT: German (de_title/de_abstract) and English (en_title/en_title) MUST differ. "
    "If the source text is German, provide a proper English translation. "
    "If the source text is English, provide a proper German translation.\n\n"
    "Example:\n"
    '  Input: {"name": "city:schulen", "title": "Schulen", "abstract": "Alle Schulen im Stadtgebiet"}\n'
    '  Output: {"name": "city:schulen", "de_title": "Schulen", "en_title": "Schools", '
    '"de_abstract": "Alle Schulen im Stadtgebiet", "en_abstract": "All schools in the city area", '
    '"aliases": ["schulen", "schools"]}\n\n'
    "Example:\n"
    '  Input: {"name": "city:hospitals", "title": "Hospitals", "abstract": "Healthcare facilities"}\n'
    '  Output: {"name": "city:hospitals", "de_title": "Krankenhäuser", "en_title": "Hospitals", '
    '"de_abstract": "Gesundheitseinrichtungen", "en_abstract": "Healthcare facilities", '
    '"aliases": ["krankenhäuser", "hospitals"]}'
)


async def _translate_batch(
    batch_layers: list[dict[str, str]],
    semaphore: asyncio.Semaphore,
) -> list[dict[str, str | list[str]]]:
    """Translate a single batch of layers behind a concurrency semaphore."""
    rows = [
        {
            "name": _normalize_text(str(layer.get("name", ""))),
            "title": _normalize_text(str(layer.get("title", ""))),
            "abstract": _normalize_text(str(layer.get("abstract", ""))),
        }
        for layer in batch_layers
    ]

    messages = [
        {"role": "system", "content": _TRANSLATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Generate translations for each layer.\n"
                "Output fields per layer: name, de_title, en_title, de_abstract, en_abstract, aliases.\n"
                f"Input layers:\n{rows}"
            ),
        },
    ]

    fallback_rows = _fallback_translation_rows(batch_layers)

    try:
        async with semaphore:
            result = await ainvoke_llm(messages=messages, output_schema=LayerTranslationBatch)
        merged_rows = _merge_translation_results(result, fallback_rows)
        if merged_rows:
            for row in merged_rows:
                row["translation_verified"] = _has_translation_occurred(row)
            return merged_rows
    except Exception:
        logger.warning("Translation batch failed for %d layers, using fallback", len(batch_layers))

    return fallback_rows


async def translate_layers_with_llm(layers: list[dict[str, str]]) -> list[dict[str, str | list[str]]]:
    if not layers:
        return []

    # Split into small batches for manageable LLM calls
    batches = [
        layers[i : i + _TRANSLATION_BATCH_SIZE]
        for i in range(0, len(layers), _TRANSLATION_BATCH_SIZE)
    ]
    logger.info(
        "Translating %d layers in %d batches (batch_size=%d, concurrency=%d)",
        len(layers), len(batches), _TRANSLATION_BATCH_SIZE, _TRANSLATION_MAX_CONCURRENCY,
    )

    semaphore = asyncio.Semaphore(_TRANSLATION_MAX_CONCURRENCY)
    batch_results = await asyncio.gather(
        *(_translate_batch(batch, semaphore) for batch in batches)
    )

    # Flatten batch results
    all_rows: list[dict[str, str | list[str]]] = []
    for result in batch_results:
        all_rows.extend(result)

    # Repair pass for untranslated rows (single batch, best-effort)
    untranslated_indices = [
        i for i, row in enumerate(all_rows) if not _has_translation_occurred(row)
    ]
    if untranslated_indices and len(untranslated_indices) <= _TRANSLATION_BATCH_SIZE:
        repair_input = [all_rows[i] for i in untranslated_indices]
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You are a translation repair assistant. The following layer rows have identical "
                    "German and English fields. Translate each en_title from German to English → en_title, "
                    "and each en_abstract from German to English → en_abstract. "
                    "If a field is already English, translate it to German for the de_ field. "
                    "Return only the corrected rows. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Fix translations for these layers. "
                    "Output fields per layer: name, de_title, en_title, de_abstract, en_abstract, aliases.\n"
                    f"Input layers:\n{repair_input}"
                ),
            },
        ]
        try:
            repair_result = await ainvoke_llm(messages=repair_messages, output_schema=LayerTranslationBatch)
            repair_parsed = _parse_translation_batch(repair_result)
            repair_by_name = {
                _normalize_text(str(item.get("name", ""))): item
                for item in repair_parsed
                if _normalize_text(str(item.get("name", "")))
            }
            for orig_idx in untranslated_indices:
                row_name = _normalize_text(str(all_rows[orig_idx].get("name", "")))
                repaired = repair_by_name.get(row_name)
                if repaired and _has_translation_occurred(repaired):
                    all_rows[orig_idx] = repaired
        except Exception:
            pass  # Repair pass is best-effort

    for row in all_rows:
        row["translation_verified"] = _has_translation_occurred(row)

    return all_rows


def _parse_translation_batch(result: LayerTranslationBatch) -> list[dict[str, str | list[str]]]:
    parsed = []
    for item in result.layers:
        parsed.append(
            {
                "name": _normalize_text(item.name),
                "de_title": _normalize_text(item.de_title),
                "en_title": _normalize_text(item.en_title),
                "de_abstract": _normalize_text(item.de_abstract),
                "en_abstract": _normalize_text(item.en_abstract),
                "aliases": [a.strip() for a in item.aliases if a and a.strip()],
            }
        )
    return parsed


def _merge_translation_results(
    result: LayerTranslationBatch,
    fallback_rows: list[dict[str, str | list[str]]],
) -> list[dict[str, str | list[str]]]:
    parsed = _parse_translation_batch(result)
    if not parsed:
        return []

    parsed_by_name = {
        _normalize_text(str(item.get("name", ""))): item for item in parsed if _normalize_text(str(item.get("name", "")))
    }

    merged_rows: list[dict[str, str | list[str]]] = []
    for index, fallback in enumerate(fallback_rows):
        fallback_name = _normalize_text(str(fallback.get("name", "")))
        candidate = parsed_by_name.get(fallback_name)
        if candidate is None and index < len(parsed):
            candidate = parsed[index]

        if candidate is None:
            merged_rows.append(fallback)
            continue

        de_title = _normalize_text(str(candidate.get("de_title", ""))) or _normalize_text(str(fallback.get("de_title", "")))
        en_title = _normalize_text(str(candidate.get("en_title", ""))) or _normalize_text(str(fallback.get("en_title", "")))
        de_abstract = _normalize_text(str(candidate.get("de_abstract", ""))) or _normalize_text(
            str(fallback.get("de_abstract", ""))
        )
        en_abstract = _normalize_text(str(candidate.get("en_abstract", ""))) or _normalize_text(
            str(fallback.get("en_abstract", ""))
        )
        aliases = [
            _normalize_text(str(alias))
            for alias in (candidate.get("aliases") or [])
            if _normalize_text(str(alias))
        ]
        if not aliases:
            aliases = [
                _normalize_text(str(alias))
                for alias in (fallback.get("aliases") or [])
                if _normalize_text(str(alias))
            ]

        merged_rows.append(
            {
                "name": fallback_name or _normalize_text(str(candidate.get("name", ""))),
                "de_title": de_title,
                "en_title": en_title,
                "de_abstract": de_abstract,
                "en_abstract": en_abstract,
                "aliases": aliases,
            }
        )

    return merged_rows


def _render_markdown_from_rows(rows: list[dict[str, str | list[str]]]) -> str:
    lines: list[str] = ["# GeoServer Layer Catalog", ""]
    generated_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    lines.append(f"Generated at (UTC): {generated_at}")
    lines.append("")

    for row in rows:
        layer_name = _normalize_text(str(row.get("name", "")))
        if not layer_name:
            continue

        de_title = _normalize_text(str(row.get("de_title", "")))
        en_title = _normalize_text(str(row.get("en_title", "")))
        de_abstract = _normalize_text(str(row.get("de_abstract", "")))
        en_abstract = _normalize_text(str(row.get("en_abstract", "")))
        aliases = [
            _normalize_text(str(alias))
            for alias in (row.get("aliases") or [])
            if _normalize_text(str(alias))
        ]

        lines.append(f"- **Layer ID:** `{layer_name}`")
        lines.append(f"  - **DE Title:** {de_title}")
        lines.append(f"  - **EN Translation:** {en_title}")
        lines.append(f"  - **DE Abstract:** {de_abstract}")
        lines.append(f"  - **EN Abstract:** {en_abstract}")
        if aliases:
            lines.append(f"  - **Aliases:** {', '.join(aliases)}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_basic_markdown_catalog(layers: list[dict[str, str]]) -> str:
    return _render_markdown_from_rows(_fallback_translation_rows(layers))


def render_catalog_rows_as_markdown(rows: list[dict[str, Any]]) -> str:
    """Render a pre-built list of translated catalog rows to markdown."""
    return _render_markdown_from_rows(rows)


def _parse_full_rows_from_markdown(markdown: str) -> list[dict[str, Any]]:
    """Parse bilingual rows back from a rendered markdown catalog (cache-hit path)."""
    rows: list[dict[str, Any]] = []
    if not markdown.strip():
        return rows
    blocks = re.split(r"\n(?=- \*\*Layer ID:\*\*)", markdown)
    for block in blocks:
        id_match = re.search(r"- \*\*Layer ID:\*\*\s*`([^`]+)`", block)
        if not id_match:
            continue
        name = _normalize_text(id_match.group(1))
        if not name:
            continue

        def _extract(pattern: str) -> str:
            m = re.search(pattern, block)
            return _normalize_text(m.group(1)) if m else ""

        de_title = _extract(r"\*\*DE Title:\*\*\s*(.+)")
        en_title = _extract(r"\*\*EN Translation:\*\*\s*(.+)")
        de_abstract = _extract(r"\*\*DE Abstract:\*\*\s*(.+)")
        en_abstract = _extract(r"\*\*EN Abstract:\*\*\s*(.+)")
        aliases_raw = _extract(r"\*\*Aliases:\*\*\s*(.+)")
        aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()] if aliases_raw else []

        rows.append({
            "name": name,
            "de_title": de_title,
            "en_title": en_title,
            "de_abstract": de_abstract,
            "en_abstract": en_abstract,
            "aliases": aliases,
        })
    return rows


async def generate_markdown_layer_catalog(
    layers: list[dict[str, str]],
    translator: LayerTranslator | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    translate = translator or translate_layers_with_llm
    rows = await translate(layers)
    return _render_markdown_from_rows(rows), rows


def is_catalog_stale(catalog_path: str, stale_after_hours: int = 8) -> bool:
    path = Path(catalog_path)
    if not path.exists():
        return True

    modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    max_age = timedelta(hours=max(1, stale_after_hours))
    return (datetime.now(tz=timezone.utc) - modified_at) > max_age


async def ensure_markdown_layer_catalog(
    *,
    layers: list[dict[str, str]],
    catalog_path: str,
    stale_after_hours: int = 8,
    translator: LayerTranslator | None = None,
    force_refresh: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    path = Path(catalog_path)
    if (not force_refresh) and path.exists() and not is_catalog_stale(catalog_path, stale_after_hours):
        markdown = path.read_text(encoding="utf-8")
        rows = _parse_full_rows_from_markdown(markdown)
        return markdown, rows

    markdown, rows = await generate_markdown_layer_catalog(layers=layers, translator=translator)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return markdown, rows


def parse_markdown_layer_catalog(markdown: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    if not markdown.strip():
        return entries

    blocks = re.split(r"\n(?=- \*\*Layer ID:\*\*)", markdown)
    for block in blocks:
        id_match = re.search(r"- \*\*Layer ID:\*\*\s*`([^`]+)`", block)
        if not id_match:
            continue

        name = _normalize_text(id_match.group(1))
        if not name:
            continue

        de_title = _normalize_text(re.search(r"\*\*DE Title:\*\*\s*(.+)", block).group(1) if re.search(r"\*\*DE Title:\*\*\s*(.+)", block) else "")
        en_title = _normalize_text(re.search(r"\*\*EN Translation:\*\*\s*(.+)", block).group(1) if re.search(r"\*\*EN Translation:\*\*\s*(.+)", block) else "")
        de_abstract = _normalize_text(re.search(r"\*\*DE Abstract:\*\*\s*(.+)", block).group(1) if re.search(r"\*\*DE Abstract:\*\*\s*(.+)", block) else "")
        en_abstract = _normalize_text(re.search(r"\*\*EN Abstract:\*\*\s*(.+)", block).group(1) if re.search(r"\*\*EN Abstract:\*\*\s*(.+)", block) else "")
        aliases = _normalize_text(re.search(r"\*\*Aliases:\*\*\s*(.+)", block).group(1) if re.search(r"\*\*Aliases:\*\*\s*(.+)", block) else "")

        title = en_title or de_title or name
        abstract_parts = [part for part in [en_abstract, de_abstract, aliases] if part]
        abstract = " | ".join(abstract_parts) if abstract_parts else title

        entries.append({"name": name, "title": title, "abstract": abstract})

    return entries
