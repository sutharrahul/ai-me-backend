"""Centralized application configuration loaded from environment variables.

Every tunable value (API keys, model names, chunking sizes, Qdrant
credentials, etc.) lives here in one `Settings` class instead of being
scattered across files as magic strings. Values are loaded from
`backend/.env`, falling back to the defaults below when a variable isn't
set.

To add a new setting: add a field below with a sensible default, then set
the matching UPPER_SNAKE_CASE variable in `.env` to override it (pydantic
automatically maps `my_setting` → `MY_SETTING`).
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Backend root (.../backend), used to build absolute paths to data/ regardless
# of the working directory the app is started from.
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE if ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ------------------------------------------------------------------ General
    app_name: str = "Portfolio Chatbot"
    # Defaults to "production" (not "development") so that if a deploy
    # platform ever forgets to set ENVIRONMENT, the app fails safe toward
    # Gemini rather than silently trying to reach a local Ollama that
    # doesn't exist there. Local dev's `.env` explicitly sets
    # ENVIRONMENT=development to opt into the Ollama default instead.
    environment: str = "production"
    api_prefix: str = "/api"

    # ------------------------------------------------------------------ CORS
    # Which frontend origin(s) may call this API from the browser.
    cors_origins: list[str] = ["http://localhost:3000"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        """Accept JSON (`["https://a.com"]`) or comma-separated URLs from
        deployment platforms that don't support JSON env values."""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                return value
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        return value

    # ------------------------------------------------------------------ Google Gemini
    # Get a free key at https://aistudio.google.com/apikey and set
    # GOOGLE_API_KEY (or GEMINI_API_KEY) in .env or your platform's env vars.
    google_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices(
            "google_api_key", "GOOGLE_API_KEY", "GEMINI_API_KEY"
        ),
    )
    gemini_embedding_model: str = "models/gemini-embedding-001"
    gemini_chat_model: str = "gemini-2.5-flash"
    # Pin to a fixed dimension so the Qdrant collection vector size is
    # consistent; changing this requires deleting and recreating the collection.
    embedding_dimensions: int = 768

    # ------------------------------------------------------------------ Ollama (local dev)
    # llm_provider defaults to "gemini" in production and "ollama" in every
    # other environment (see `apply_provider_defaults` below) — set it
    # explicitly here (or via LLM_PROVIDER) to override that. embedding_provider
    # always defaults to "gemini" regardless of environment: switching it
    # requires re-ingesting documents, since embeddings from a different model
    # aren't comparable to what's already stored even at the same dimension.
    llm_provider: str | None = None
    embedding_provider: str = "gemini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma3:4b"
    ollama_embedding_model: str = "nomic-embed-text"



    # ------------------------------------------------------------------ DB_URI
    db_uri:SecretStr = SecretStr("")
    # ------------------------------------------------------------------ Qdrant
    # Matches `docker-compose.yml`'s `qdrant` service, so
    # `docker compose up -d qdrant` + these defaults "just work" for local dev.
    qdrant_url: str = "http://localhost:6333"
    # Leave empty for a local/self-hosted Qdrant with no auth; set when
    # pointing at Qdrant Cloud.
    qdrant_api_key: SecretStr = SecretStr("")
    # Name of the Qdrant collection the chunk vectors live in.
    vector_collection_name: str = "documents"

    # ------------------------------------------------------------------ Storage
    data_dir: Path = BASE_DIR / "data"
    # Drop .md/.pdf files here; `scripts/ingest_documents.py` reads every
    # supported file and embeds it into the vector store.
    seed_dir: Path = BASE_DIR / "data" / "seed"
    # Path to the Markdown file holding the assistant's system prompt/persona.
    # Edit this file and restart the server to change the assistant's voice
    # without touching Python code.
    system_prompt_path: Path = BASE_DIR / "data" / "system_prompt.md"

    # ------------------------------------------------------------------ Chunking
    # chunk_size: max characters per chunk.
    # chunk_overlap: trailing characters of one chunk repeated at the start of
    # the next, so context isn't lost at chunk boundaries.
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ------------------------------------------------------------------ Retrieval
    # Default number of chunks retrieved per question if the caller doesn't
    # specify `top_k` explicitly.
    default_top_k: int = 4

    # ------------------------------------------------------------------ Rate limiting
    # Max questions a single IP can ask per day. Protects the paid Gemini API
    # from being drained by bots (no auth is required to use the chat endpoint).
    daily_question_limit: int = 10

    # ------------------------------------------------------------------ Redis query cache
    # When two visitors ask the same question the second one is served from
    # cache, skipping both the embedding call and the LLM call. Purely an
    # optimisation: if Redis is unreachable, caching silently becomes a no-op.
    enable_query_cache: bool = True
    redis_url: str = "redis://localhost:6379/0"
    # How long a cached answer stays valid. Kept under a day by default so
    # stale answers don't persist long if seed content changes. The ingestion
    # script also clears the cache automatically after re-ingesting.
    query_cache_ttl_seconds: int = 3600

    # ------------------------------------------------------------------ Computed helpers

    @property
    def system_prompt(self) -> str:
        """Loads the system prompt from `system_prompt_path` on every access.
        Editing that file and restarting the server is all it takes to change
        the assistant's persona. Falls back to a minimal built-in default if
        the file is missing."""
        try:
            return self.system_prompt_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return (
                "You are a helpful assistant. Answer ONLY using the "
                "provided context. If the answer cannot be found in the "
                "context, say you don't have that information instead of "
                "making something up."
            )

    def resolved_google_api_key(self) -> str:
        """Returns the Gemini API key, falling back to raw process env vars.
        On deploy, platforms inject GOOGLE_API_KEY or GEMINI_API_KEY directly;
        this avoids losing the key when an empty placeholder in .env would
        otherwise override the real env var."""
        key = self.google_api_key.get_secret_value().strip()
        if key:
            return key
        return (
            os.environ.get("GOOGLE_API_KEY", "").strip()
            or os.environ.get("GEMINI_API_KEY", "").strip()
        )

    @model_validator(mode="after")
    def apply_provider_defaults(self) -> Self:
        """Defaults llm_provider to "gemini" in production and "ollama"
        everywhere else, when LLM_PROVIDER isn't set explicitly — so
        production always uses Gemini and local dev/testing uses a local
        Ollama model without needing to remember to flip a setting."""
        if self.llm_provider is None:
            self.llm_provider = "gemini" if self.environment == "production" else "ollama"
        return self

    @model_validator(mode="after")
    def validate_google_api_key(self) -> Self:
        """Fail fast at startup with a clear message when the Gemini API key
        is missing, instead of a cryptic LangChain error on the first request.
        Skipped entirely when both providers are set to "ollama", since no
        Gemini call is ever made in that configuration."""
        uses_gemini = "gemini" in (self.llm_provider, self.embedding_provider)
        if uses_gemini and not self.resolved_google_api_key():
            raise ValueError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) must be set. "
                "Get a free key at https://aistudio.google.com/apikey and "
                "add it to backend/.env or your deployment platform's env vars."
            )
        return self

    def ensure_directories(self) -> None:
        """Creates the local data folders on disk if they don't already exist,
        so the app never crashes on a missing directory on a fresh clone."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seed_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Returns the app-wide `Settings` instance, creating it (and reading
    `.env`) only once thanks to `@lru_cache`. Call this everywhere you need
    config instead of instantiating `Settings()` directly."""
    settings = Settings()
    settings.ensure_directories()
    return settings
