from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    current_model: str = "gpt-4.1"
    routing_model: str = "gpt-4.1"
    synthesizer_model: str = "gpt-4o-mini"
    layer_selector_model: str = "gpt-4.1"
    llm_base_url: str = ""
    llm_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    geoserver_wfs_url: str = "http://localhost:8080/geoserver/wfs"
    geoserver_wfs_username: str = ""
    geoserver_wfs_password: str = ""
    geoserver_wfs_srs_name: str = "EPSG:3857"

    layer_catalog_markdown_path: str = "layer_catalog.md"
    layer_catalog_stale_after_hours: int = 8

    geocoder_api_url: str = "https://stargate-cetus.prod.tardis.telekom.de/geo"
    geocoder_token_url: str = ""
    geocoder_client_id: str = ""
    geocoder_client_secret: str = ""
    geocoder_scope: str = ""

    llm_prompt_cache_enabled: bool = False
    llm_prompt_cache_ttl_seconds: int = 3600
    llm_prompt_cache_max_entries: int = 512

    layer_discovery_mode: str = "fuzzy"  # "fuzzy" | "semantic"

    embedding_model: str = "jina-embeddings-v2-base-de"
    embedding_batch_size: int = 8
    embedding_rpm: int = 20
    embedding_tpm: int = 30000
    vector_store_top_k: int = 10
    vector_reindex_hours: int = 24

    min_retrieval_score: float = 0.15
    max_llm_candidates: int = 15

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
