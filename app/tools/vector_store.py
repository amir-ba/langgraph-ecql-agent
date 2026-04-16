from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import hashlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import chromadb

from app.tools.layer_catalog import parse_geometry_type_from_name

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


def _build_document(layer: dict[str, str], catalog_row: dict[str, Any] | None) -> str:
    """Build embedding document from layer metadata and optional catalog row."""
    name = layer.get("name", "")
    if catalog_row:
        parts = [
            f"Layer: {name}",
            f"Title DE: {catalog_row.get('de_title', '')}",
            f"Title EN: {catalog_row.get('en_title', '')}",
            f"Abstract DE: {catalog_row.get('de_abstract', '')}",
            f"Abstract EN: {catalog_row.get('en_abstract', '')}",
        ]
        aliases = catalog_row.get("aliases") or []
        if aliases:
            parts.append(f"Aliases: {', '.join(aliases)}")
    else:
        parts = [
            f"Layer: {name}",
            f"Title: {layer.get('title', '')}",
            f"Abstract: {layer.get('abstract', '')}",
        ]
    geom_type = parse_geometry_type_from_name(name)
    if geom_type:
        parts.append(f"Geometry type: {geom_type}")
    return " | ".join(parts)


def _compute_content_hash(documents: list[str]) -> str:
    content = "\n".join(sorted(documents))
    return hashlib.sha256(content.encode()).hexdigest()


class LayerVectorStore:
    _instance_counter: int = 0

    def __init__(self) -> None:
        LayerVectorStore._instance_counter += 1
        self._collection_name = f"layers_{LayerVectorStore._instance_counter}"
        self._client = chromadb.Client()
        self._collection: chromadb.Collection | None = None
        self._indexed = False
        self._last_indexed_at: datetime | None = None
        self._last_content_hash: str | None = None
        self._index_lock = asyncio.Lock()

    async def index_layers(
        self,
        layers: list[dict[str, str]],
        catalog_rows: list[dict[str, Any]],
        embed_fn: EmbedFn,
    ) -> None:
        async with self._index_lock:
            row_by_name: dict[str, dict[str, Any]] = {
                str(r.get("name", "")): r for r in catalog_rows
            }

            documents: list[str] = []
            ids: list[str] = []
            metadatas: list[dict[str, str]] = []

            for layer in layers:
                name = str(layer.get("name", ""))
                if not name:
                    continue
                catalog_row = row_by_name.get(name)
                doc = _build_document(layer, catalog_row)
                documents.append(doc)
                ids.append(name)
                metadatas.append({
                    "layer_name": name,
                    "title": str(layer.get("title", "")),
                    "abstract": str(layer.get("abstract", "")),
                })

            if not documents:
                self._indexed = True
                self._last_indexed_at = datetime.now(UTC)
                return

            content_hash = _compute_content_hash(documents)
            if self._indexed and self._last_content_hash == content_hash:
                logger.info("Catalog unchanged (hash %s), skipping re-index", content_hash[:12])
                return

            # Start fresh: delete then create
            try:
                self._client.delete_collection(self._collection_name)
            except Exception:
                pass
            self._collection = self._client.create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            embeddings = await embed_fn(documents)

            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            self._indexed = True
            self._last_indexed_at = datetime.now(UTC)
            self._last_content_hash = content_hash
            logger.info("Indexed %d layers into vector store", len(ids))

    def search(self, query_embedding: list[float], top_k: int = 8) -> list[dict[str, Any]]:
        if not self._collection or not self._indexed:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.layer_count()) if self.layer_count() > 0 else top_k,
        )

        items: list[dict[str, Any]] = []
        if not results or not results.get("ids"):
            return items

        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)

        for layer_id, distance, metadata in zip(ids, distances, metadatas):
            # ChromaDB cosine distance = 1 - cosine_similarity
            score = 1.0 - distance
            items.append({
                "layer_name": metadata.get("layer_name", layer_id),
                "score": score,
                "title": metadata.get("title", ""),
                "abstract": metadata.get("abstract", ""),
            })

        items.sort(key=lambda x: x["score"], reverse=True)
        return items

    def is_indexed(self) -> bool:
        return self._indexed

    def should_reindex(self, *, max_age_hours: int) -> bool:
        if not self._indexed:
            return True
        if self._last_indexed_at is None:
            return True
        age = datetime.now(UTC) - self._last_indexed_at
        return age >= timedelta(hours=max(1, max_age_hours))

    def layer_count(self) -> int:
        if not self._collection:
            return 0
        return self._collection.count()

    def clear(self) -> None:
        if self._collection is not None:
            try:
                self._client.delete_collection(self._collection_name)
            except Exception:
                pass
            self._collection = None
        self._indexed = False
        self._last_indexed_at = None
        self._last_content_hash = None


_store: LayerVectorStore | None = None


def get_layer_vector_store() -> LayerVectorStore:
    global _store
    if _store is None:
        _store = LayerVectorStore()
    return _store


def reset_layer_vector_store() -> None:
    global _store
    if _store is not None:
        _store.clear()
    _store = None
