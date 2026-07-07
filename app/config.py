"""Centralized application configuration loaded from environment variables.

Every tunable value in the app (API keys, model names, chunking sizes,
Postgres credentials, etc.) lives here in one `Settings` class instead of
being scattered across files as magic strings/numbers. Values are loaded
from `backend/.env` (see `.env.example` for the full list of variables),
falling back to the defaults below when a variable isn't set.

To add a new setting: add a field below with a sensible default, then set
the matching UPPER_SNAKE_CASE variable in `.env` to override it (pydantic
automatically maps `my_setting` -> `MY_SETTING`).
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root for the backend (.../backend), used to build absolute paths
# to the data/ folder regardless of the current working directory the app
# is started from.
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    # Loads `backend/.env` when present (local dev). On deploy, `.env` is
    # usually omitted from the image and real environment variables from
    # the host/platform take priority. `populate_by_name` lets aliases like
    # GEMINI_API_KEY map onto `google_api_key`.
    model_config = SettingsConfigDict(
        env_file=ENV_FILE if ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # General
    app_name: str = "RAG Backend"
    environment: str = "development"
    api_prefix: str = "/api"

    # CORS - which frontend origin(s) are allowed to call this API from the
    # browser. Add more origins here (or via CORS_ORIGINS in .env) if you
    # deploy the frontend somewhere else.
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

    # Provider selection: currently only "gemini" is supported.
    # Flip EMBEDDING_PROVIDER / LLM_PROVIDER in .env if needed.
    embedding_provider: str = "gemini"
    llm_provider: str = "gemini"


    # Google Gemini (via LangChain) - the default provider. Get a free key
    # at https://aistudio.google.com/apikey and set GOOGLE_API_KEY (or
    # GEMINI_API_KEY) in .env / your deployment platform's env vars.
    google_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices(
            "google_api_key", "GOOGLE_API_KEY", "GEMINI_API_KEY"
        ),
    )
    gemini_embedding_model: str = "models/gemini-embedding-001"
    gemini_chat_model: str = "gemini-2.5-flash"
    # Gemini's embedding model can output vectors of different sizes; we
    # pin it to a fixed dimension so the Postgres `vector` column has a
    # consistent size (required for pgvector indexing).
    embedding_dimensions: int = 768

    # Vector store selection: "postgres" (default, pgvector) or "qdrant".
    # Flip VECTOR_STORE_PROVIDER in .env to switch without touching code -
    # see `get_vector_store` in `services/vector_store.py`.
    vector_store_provider: str = "qdrant"

    # Postgres / pgvector (vector store, via LangChain). These defaults
    # match `docker-compose.yml`'s `postgres` service, so `docker compose
    # up -d postgres` + the defaults below "just work" for local dev.
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_user: str = "raguser"
    postgres_password: str = "ragpassword"
    postgres_db: str = "ragdb"

    # Qdrant (alternative vector store, via `qdrant-client`). Only used
    # when vector_store_provider="qdrant". Matches `docker-compose.yml`'s
    # `qdrant` service, so `docker compose up -d qdrant` + these defaults
    # "just work" for local dev.
    qdrant_url: str = "http://localhost:6333"

    # Leave empty for a local/self-hosted Qdrant with no auth (the Docker
    # setup here); set it if you point at Qdrant Cloud instead.
    qdrant_api_key: SecretStr = SecretStr("")

    # Storage - local filesystem paths used by the app.
    data_dir: Path = BASE_DIR / "data"
    # Drop .txt/.md/.pdf files here; `scripts/ingest_documents.py` reads
    # every file in this folder and embeds it into the vector store.
    seed_dir: Path = BASE_DIR / "data" / "seed"
    # JSON file tracking which documents have been ingested (id, filename,
    # status, chunk count) - see `services/document_store.py`.
    documents_index_path: Path = BASE_DIR / "data" / "documents.json"
    # Name of the pgvector "collection" (logical table) the chunks live in.
    vector_collection_name: str = "documents"

    @property
    def database_url(self) -> str:
        """Builds the SQLAlchemy/psycopg3 connection string for Postgres
        from the individual postgres_* fields above. Computed as a property
        (rather than stored) so changing any of the pieces automatically
        keeps the URL in sync."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Chunking - controls how documents are split before embedding.
    # chunk_size: max characters per chunk. chunk_overlap: how many trailing
    # characters of one chunk are repeated at the start of the next, so
    # context isn't lost right at a chunk boundary. See `text_splitter.py`.
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval / generation
    # Default number of chunks retrieved per question if the API caller
    # doesn't specify `top_k` explicitly.
    default_top_k: int = 4

    # Rate limiting - caps how many questions a single IP address can ask
    # per day, without requiring any user accounts/auth (see
    # `core/rate_limiter.py`). Protects the (paid, per-call) embedding/LLM
    # APIs from being drained by one visitor or a bot, since there's no
    # login to otherwise gate usage per-user.
    daily_question_limit: int = 20

    # Query caching (Redis) - when two people ask the same question (e.g.
    # one of the chat UI's suggestion buttons), the second one is served
    # straight from cache instead of re-running the embedding + LLM calls,
    # saving quota/cost. See `services/cache.py`. Purely an optimization:
    # if Redis is unreachable, caching silently disables itself rather
    # than breaking the app - see `get_query_cache`.
    enable_query_cache: bool = True
    redis_url: str = "redis://localhost:6379/0"
    # How long a cached answer stays valid. Kept well under a day by
    # default so cached answers don't go too stale if `data/seed/` content
    # changes without anyone remembering to clear the cache (ingestion
    # also clears it automatically - see `scripts/ingest_documents.py`).
    query_cache_ttl_seconds: int = 3600
    # Path to the markdown file holding the assistant's system prompt/rule
    # set (see `data/system_prompt.md`). Kept as its own file rather than
    # an inline string so it's easy to read/edit the assistant's
    # persona/rules without touching Python code.
    system_prompt_path: Path = BASE_DIR / "data" / "system_prompt.md"

    @property
    def system_prompt(self) -> str:
        """Loads the system prompt from `system_prompt_path` on every
        access (cheap: it's a small local file), so editing that file and
        restarting the server (or re-running `--reload`) is all it takes
        to change the assistant's persona/rules. Falls back to a minimal
        built-in default if the file is missing, so a fresh clone that
        hasn't created `data/system_prompt.md` yet still boots and
        answers questions instead of crashing."""
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
        """Returns the Gemini API key from settings, falling back to raw
        process env vars. On deploy, platforms inject GOOGLE_API_KEY or
        GEMINI_API_KEY directly; this avoids losing the key when an empty
        placeholder in .env would otherwise override the real env var."""
        key = self.google_api_key.get_secret_value().strip()
        if key:
            return key
        return (
            os.environ.get("GOOGLE_API_KEY", "").strip()
            or os.environ.get("GEMINI_API_KEY", "").strip()
        )

    @model_validator(mode="after")
    def validate_provider_keys(self) -> Self:
        """Fail fast at startup with a clear message when the Gemini API
        key is missing, instead of a cryptic LangChain error on the first
        chat/ingest request."""
        if self.embedding_provider == "gemini" and not self.resolved_google_api_key():
            raise ValueError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is required when "
                "EMBEDDING_PROVIDER=gemini. Set it in your deployment "
                "platform's environment variables."
            )
        if self.llm_provider == "gemini" and not self.resolved_google_api_key():
            raise ValueError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is required when "
                "LLM_PROVIDER=gemini. Set it in your deployment platform's "
                "environment variables."
            )
        return self

    def ensure_directories(self) -> None:
        """Creates the local data folders on disk if they don't already
        exist, so the app never crashes on a missing directory on first run
        (e.g. a fresh git clone with no data/ folder yet)."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seed_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Returns the app-wide `Settings` instance, creating it (and reading
    `.env`) only once thanks to `@lru_cache`. Call this function everywhere
    you need config instead of instantiating `Settings()` directly, so the
    whole app shares one consistent, cached configuration object."""
    settings = Settings()
    settings.ensure_directories()
    return settings
