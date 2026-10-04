from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ------------------------------------------------------------
    # Generation provider selection
    # ------------------------------------------------------------
    # "openrouter" | "ollama" | "opencode"
    generation_provider: Literal[
        "openrouter", "ollama", "opencode"
    ] = "openrouter"

    # ------------------------------------------------------------
    # Generation provider (OpenRouter)
    # ------------------------------------------------------------
    # The API key is optional at import time so that modules can be
    # imported (tests, tooling) without a .env file present.
    # Requests fail fast with a clear message when it is missing.
    openrouter_api_key: str = ""
    openrouter_model: str = "qwen/qwen3.8-27b:free"
    openrouter_timeout: float = 120.0
    openrouter_max_retries: int = 2

    # ------------------------------------------------------------
    # Generation provider (OpenCode Zen)
    # ------------------------------------------------------------
    # OpenAI-compatible gateway, billed per request, so it avoids
    # free-tier rate limits. Its catalogue is text-only: image
    # context is dropped by the provider (logged, not fatal).
    opencode_api_key: str = ""
    opencode_base_url: str = "https://opencode.ai/zen/v1"
    opencode_model: str = "gpt-5"
    opencode_timeout: float = 120.0
    opencode_max_retries: int = 2

    # ------------------------------------------------------------
    # Generation provider (Ollama)
    # ------------------------------------------------------------
    # Local by default (http://localhost:11434, no auth required).
    # For Ollama Cloud models set:
    #   OLLAMA_BASE_URL=https://ollama.com
    #   OLLAMA_API_KEY=<key from ollama.com settings>
    #   OLLAMA_MODEL=<a cloud model, e.g. gpt-oss:120b-cloud>
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str = ""
    ollama_model: str = "qwen3-vl:4b"
    ollama_timeout: float = 300.0

    # ------------------------------------------------------------
    # API layer
    # ------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Comma-separated list of allowed origins for CORS.
    # Use "*" to allow any origin (development only).
    api_cors_origins: str = "http://localhost:5173,http://localhost:4173"
    # Serve retrieved images through the API instead of exposing
    # server filesystem paths to clients.
    api_media_url_prefix: str = "/media"
    api_max_upload_size_mb: int = 25
    # Number of ranked sources returned to clients. Retrieval results
    # arrive best-first (RRF primary results, then related chunks),
    # so this keeps the strongest matches only.
    api_max_sources: int = 1
    # Max size of an image attached to a chat message. It travels
    # base64-encoded inside the JSON body (~33% larger) and is sent
    # to the vision model, so keep it modest.
    api_max_query_image_size_mb: int = 5

    # ------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------
    # Visually similar document images sent to the model alongside
    # an image the user attached to their question.
    retrieval_visual_top_k: int = 2

    # ------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------
    vector_store_directory: str = "data/vector_store"
    # Root of all processed media served under api_media_url_prefix
    # (covers both data/processed/images and data/processed/normalized).
    processed_images_directory: str = "data/processed"
    uploads_directory: str = "data/uploads"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.api_cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.api_max_upload_size_mb * 1024 * 1024

    @property
    def max_query_image_bytes(self) -> int:
        return self.api_max_query_image_size_mb * 1024 * 1024


settings = Settings()