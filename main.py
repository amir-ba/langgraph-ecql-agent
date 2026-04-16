import logging
from contextlib import asynccontextmanager


def configure_logging() -> None:
    # Respect server/runtime log-level configuration instead of hardcoding DEBUG.
    uvicorn_error = logging.getLogger("uvicorn.error")

    app_logger = logging.getLogger("app")
    # Mirror uvicorn's effective level so --log-level also controls app.* loggers.
    app_logger.setLevel(uvicorn_error.getEffectiveLevel())
    if uvicorn_error.handlers:
        app_logger.handlers = uvicorn_error.handlers
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        app_logger.handlers = [handler]
    app_logger.propagate = False

    logging.getLogger("litellm").setLevel(uvicorn_error.getEffectiveLevel())
    logging.getLogger("LiteLLM").setLevel(uvicorn_error.getEffectiveLevel())


configure_logging()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.http_clients import create_http_client_pool
from app.core.settings import get_settings
from app.tools.layer_catalog import ensure_markdown_layer_catalog
from app.tools.wfs_client import discover_layers
from app.tools.embedding_client import get_embeddings
from app.tools.vector_store import get_layer_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client_pool = create_http_client_pool()
    try:
        settings = get_settings()
        try:
            layers = await discover_layers(
                wfs_url=settings.geoserver_wfs_url,
                http_client=app.state.http_client_pool.wfs,
                username=settings.geoserver_wfs_username,
                password=settings.geoserver_wfs_password,
            )
            _, catalog_rows = await ensure_markdown_layer_catalog(
                layers=layers,
                catalog_path=settings.layer_catalog_markdown_path,
                stale_after_hours=settings.layer_catalog_stale_after_hours,
            )
            logging.getLogger("app.main").info(
                "Layer markdown catalog ready with %s layers", len(layers)
            )

            if settings.layer_discovery_mode.strip().lower() == "semantic":
                store = get_layer_vector_store()
                await store.index_layers(layers, catalog_rows, get_embeddings)
                logging.getLogger("app.main").info(
                    "Vector store indexed with %s layers", store.layer_count()
                )
            else:
                logging.getLogger("app.main").info(
                    "Layer discovery mode is '%s', skipping vector store indexing",
                    settings.layer_discovery_mode,
                )
        except Exception as exc:
            logging.getLogger("app.main").warning(
                "Layer catalog / vector store initialization failed: %s", exc
            )
        yield
    finally:
        await app.state.http_client_pool.aclose()


app = FastAPI(title="GeoServer ECQL Agent", version="0.1.0", lifespan=lifespan)

# Allow only Telekom Pages hosted frontends.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https://([a-zA-Z0-9-]+\.)*pages\.devops\.telekom\.de$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(api_router)
