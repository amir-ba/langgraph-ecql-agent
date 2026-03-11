import logging


def configure_logging() -> None:
    # Keep third-party libraries quiet while allowing project debug logs.
    logging.getLogger().setLevel(logging.INFO)
    uvicorn_error = logging.getLogger("uvicorn.error")

    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.DEBUG)
    if uvicorn_error.handlers:
        app_logger.handlers = uvicorn_error.handlers
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        app_logger.handlers = [handler]
    app_logger.propagate = False

    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)


configure_logging()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router


app = FastAPI(title="GeoServer ECQL Agent", version="0.1.0")

# Phase 1 default: permissive CORS for local UI development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(api_router)
