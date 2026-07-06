# RAG Backend

A FastAPI boilerplate for a Retrieval-Augmented Generation (RAG) system: drop documents into `data/seed/` (or fetch them from a remote API via `POST /api/documents/ingest-url`), chunk + embed them (via **LangChain** + **Google Gemini**) into a **Postgres/pgvector** vector store, and ask questions that are answered using retrieved context.

## Architecture

```
app/
├── main.py                 # FastAPI app, CORS, router registration
├── config.py                # Settings loaded from environment (.env)
├── dependencies.py          # Cached singleton providers (DI)
├── api/routes/
│   ├── health.py             # GET  /api/health
│   ├── chat.py                # POST /api/chat/query
│   └── documents.py           # POST /api/documents/ingest-url
├── core/
│   └── rag_pipeline.py       # Orchestrates ingestion + retrieval + generation
├── services/
│   ├── document_loader.py    # Extracts text from .txt/.md/.pdf
│   ├── text_splitter.py      # Recursive character chunking
│   ├── embeddings.py         # Embedding provider abstraction (Gemini via LangChain, OpenAI fallback)
│   ├── vector_store.py       # Postgres/pgvector-backed similarity search (via `langchain-postgres`)
│   ├── llm.py                  # Chat completion wrapper (Gemini via LangChain, OpenAI fallback)
│   └── document_store.py     # JSON-backed document metadata index (tracks what's been ingested)
└── models/schemas.py        # Pydantic request/response models

scripts/
└── ingest_documents.py      # Reads every file in data/seed/ and ingests it
```

Data flow:

1. **Add documents**: drop `.txt`/`.md`/`.pdf` files into `data/seed/`.
2. **Ingest** (`python scripts/ingest_documents.py`) loads each file's raw text, splits it into overlapping chunks, embeds each chunk with Gemini, and stores the vectors in Postgres (pgvector). Already-ingested files are skipped unless `--force` is passed.
3. **Query** (`POST /api/chat/query`) embeds the question, retrieves the top-k most similar chunks via pgvector cosine similarity, and asks the LLM (Gemini) to answer using only that context.

## Getting started

### 1. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Postgres (pgvector) locally via Docker

```bash
docker compose up -d postgres
```

This starts a `pgvector/pgvector:pg16` container on `localhost:5433` (mapped
from the container's `5432`) using the credentials in `docker-compose.yml`
(`raguser` / `ragpassword` / `ragdb`). Data persists in the `pgdata` Docker
volume across restarts. The `PGVector` LangChain integration automatically
creates the `vector` extension, tables, and collection on first connection —
no manual migration needed.

### 3. Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env` and set `GOOGLE_API_KEY` to a valid Gemini API key (get one
at https://aistudio.google.com/apikey). The Postgres values already match
the `docker-compose.yml` defaults, so you only need to change them if you
changed the container's credentials/port. To use OpenAI instead of Gemini,
set `EMBEDDING_PROVIDER=openai` and/or `LLM_PROVIDER=openai` and fill in
`OPENAI_API_KEY`.

### 4. Add documents and ingest them

Drop any `.txt`, `.md`, or `.pdf` files into `data/seed/` (a starter
`about_rahul.md` is included — edit or replace it with your own bio,
resume, project write-ups, etc.), then run:

```bash
python scripts/ingest_documents.py
```

Re-run any time you add or edit files. Already-ingested files are skipped
unless you pass `--force` (which re-ingests everything found in `data/seed/`).

### 5. Run the development server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

### 6. Run tests

```bash
pytest
```

## API summary

| Method | Path                     | Description                          |
| ------ | ------------------------ | ------------------------------------ |
| GET    | `/api/health`             | Health check                         |
| POST   | `/api/chat/query`         | Ask a question, get a grounded answer (rate limited - see below) |
| POST   | `/api/documents/ingest-url` | Fetch a document from a URL (sending an API key in a header) and ingest it |

## Rate limiting

`POST /api/chat/query` is capped at `DAILY_QUESTION_LIMIT` (default 20) questions per day **per IP address**, via [`slowapi`](https://github.com/laurentS/slowapi) (`app/core/rate_limiter.py`). This protects the paid embedding/LLM APIs from being drained by one visitor or a bot, since there's no auth to otherwise gate usage per-user. Exceeding it returns `429` with a `detail` message.

This is in-memory per backend process: counts reset on restart/redeploy, and won't be shared correctly if you ever run multiple backend replicas (point `Limiter(storage_uri="redis://...")` at a shared Redis instance if you scale beyond one process). It also assumes requests reach this app directly - if you put a reverse proxy/load balancer in front of it, swap `get_remote_address` for a forwarded-header-aware key function, or every visitor will appear to share the proxy's IP.

## Query caching

Identical questions (normalized - case/whitespace-insensitive) are cached in Redis for `QUERY_CACHE_TTL_SECONDS` (default 1 hour), keyed on the question text + `top_k` - see `app/services/cache.py`. A cache hit skips **both** the embedding call and the LLM call in `RagPipeline.answer_query`, so it saves real API quota/cost, especially for common questions every visitor asks (e.g. the chat UI's suggestion buttons).

Run `docker compose up -d redis` to start it locally (or `docker compose up --build` to run everything together). This is a pure optimization, not a hard dependency: if Redis is unreachable or `ENABLE_QUERY_CACHE=false`, every cache lookup/write becomes a no-op and the app behaves exactly as it did before caching existed - it just won't be as cheap for repeat questions. The cache is also cleared automatically at the end of `scripts/ingest_documents.py` whenever it (re-)ingests anything, so visitors don't keep getting answers grounded in stale content after you update `data/seed/`.

## Extending this boilerplate

- **Swap the vector store**: implement the same interface as `VectorStore` (`app/services/vector_store.py`) for Pinecone, Qdrant, Weaviate, etc. (or use LangChain's other vector store integrations directly).
- **Swap the embedding/LLM provider**: implement `EmbeddingProvider` (`app/services/embeddings.py`) or add a new `LLMClient` (`app/services/llm.py`) for Anthropic, Cohere, local models, etc.
- **Persistent metadata**: replace `DocumentStore`'s JSON file with a real database table (e.g. a `documents` table in the same Postgres instance) once you outgrow single-instance deployments.
- **File upload API**: `POST /api/documents/ingest-url` covers fetching from a remote API; for direct file uploads, add a route that saves the upload and calls `RagPipeline.ingest_document` as a background task.
- **Streaming responses**: add a `POST /api/chat/query/stream` endpoint using `StreamingResponse` and Gemini/OpenAI's streaming APIs.
- **Auth**: add an API key or JWT dependency to protect the routes before deploying publicly.

## Docker

Run everything (Postgres + backend):

```bash
docker compose up -d postgres   # from the repo root
docker build -t rag-backend .
docker run --env-file .env -e POSTGRES_HOST=host.docker.internal -p 8000:8000 -v $(pwd)/data:/app/data rag-backend
```

Or use the root `docker-compose.yml` to run Postgres + backend + frontend together (see the top-level README).
