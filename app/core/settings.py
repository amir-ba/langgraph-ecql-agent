from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    current_model: str = "gpt-4o"
    llm_base_url: str = ""
    llm_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    geoserver_wfs_url: str = "http://localhost:8080/geoserver/wfs"
    geoserver_wfs_username: str = ""
    geoserver_wfs_password: str = ""

    geocoder_api_url: str = "https://stargate-cetus.prod.tardis.telekom.de/geo"
    geocoder_token_url: str = ""
    geocoder_client_id: str = ""
    geocoder_client_secret: str = ""
    geocoder_scope: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
