# 🚀 Personal Chatbot — Backend

A FastAPI backend for Rahul Suthar's personal portfolio chatbot: classifies each
message's intent, retrieves grounded context from a **Qdrant** vector store for
portfolio questions, and streams the answer back token-by-token from **Google
Gemini** (via LangChain). Conversations are persisted in **Postgres**, split
into fixed-size storage chunks under the hood while still appearing as a
single item in chat history.

## Architecture

```
app/
├── main.py                   # FastAPI app, CORS, rate-limit middleware, router registration
├── config.py                  # Settings loaded from environment (.env)
├── api/routes/
│   ├── health.py               # GET  /api/health
│   └── chat.py                  # POST /api/chat/query (streaming), GET /api/chat/list,
│                                 #   GET /api/chat/{chat_id}, GET /api/chat/chunk/{chunk_id}
├── graph/
│   ├── node.py                 # classify_intent, route_intent, retrieve_chunks (plain functions,
│   │                            #   called directly from chat.py - not run through the LangGraph
│   │                            #   runtime, since intent routing needs to interleave with streaming)
│   ├── state.py                 # AgentState TypedDict shared by the node functions above
│   └── system_prompt.py        # every system prompt: intent classification, greeting, small talk,
│                                 #   unknown/off-topic, portfolio RAG answer, chat summary, chat title
├── db/
│   ├── db_connection.py        # SQLAlchemy engine/session (Postgres via psycopg)
│   ├── schema_modal.py         # `Chat` (one row per storage chunk) and `User_Session` models
│   └── db_query.py             # conversation/chunk persistence, chunk-splitting, title/summary generation
├── services/
│   ├── llm.py                    # Gemini chat client (invoke + stream), Ollama for local dev
│   ├── embeddings.py            # Gemini/Ollama embedding provider
│   ├── vector_store.py          # Qdrant-backed similarity search
│   ├── document_loader.py       # Extracts text from .md/.pdf
│   ├── text_splitter.py         # Recursive character chunking
│   └── cache.py                  # Optional Redis cache for repeated questions
├── models/schemas.py           # Pydantic request/response schemas
├── core/rate_limiter.py        # Per-IP daily question limit (slowapi)
└── utils/logger.py             # Logging setup

scripts/
├── ingest_documents.py         # Reads every file in data/seed/, embeds it, upserts into Qdrant
└── delete_qdrant_collection.py # Wipes the Qdrant collection (start over from a clean index)
```

### How a chat request is handled

`POST /api/chat/query` (see `app/api/routes/chat.py`):

1. **Classify intent** (`classify_intent`) — one fast LLM call labels the message `GREETING`,
   `SMALL TALK`, `PORTFOLIO`, or `UNKNOWN` (see `INTENT_SYSTEM_PROMPT`).
2. **Retrieve, only if needed** — only a `PORTFOLIO`-classified question triggers
   `retrieve_chunks`, which embeds the question and queries Qdrant for the top-k most similar
   chunks. Greetings, small talk, and off-topic/unrelated requests (e.g. "write me some code")
   never touch the vector store — they're answered directly from a fixed system prompt.
3. **Stream the answer** — the model's response streams back as newline-delimited JSON
   (`TokenEvent` → ... → `DoneEvent`/`ErrorEvent`, see `app/models/schemas.py`), so the frontend
   can render it token-by-token instead of waiting for the full answer.
4. **Persist** — once the stream ends, the full exchange is saved via `store_exchange`
   (`app/db/db_query.py`).

### Conversations vs. storage chunks

A "conversation" (what the frontend/API calls `chat_id`) is a stable id created once, on
"New Chat," and never changes. Internally, `db_query.py` splits a long conversation into
multiple `Chat` rows ("chunks") of `CHUNK_SIZE` messages each (currently 20 messages / 10
exchanges), linked via `previous_chunk_id`, so no single row grows unbounded. This is entirely
an implementation detail:

- `GET /api/chat/list` groups by conversation, so a 200-message conversation still lists as
  **one** history item.
- `GET /api/chat/{chat_id}` returns only the latest chunk (fast to load).
- `GET /api/chat/chunk/{chunk_id}` fetches an older chunk on demand, for the frontend's
  infinite-scroll-up pagination.
- The chat's title is generated once (from the first exchange) and copied forward onto every
  later chunk of the same conversation, so the sidebar always shows one consistent title.

## Getting started

### 1. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Postgres, Qdrant, and Redis locally via Docker

From the repo root:

```bash
docker compose up -d postgres qdrant redis
```

Postgres stores chat/session history; Qdrant stores document embeddings; Redis is an optional
query cache (see below). The `Chat`/`User_Session` tables aren't created by a migration tool —
they're expected to already exist (created once via SQLAlchemy or applied manually); if you're
starting from a fresh database, create them from `app/db/schema_modal.py`'s model definitions.

### 3. Configure environment variables

Create a `.env` file in `backend/` with at least:

```dotenv
GOOGLE_API_KEY=your-gemini-api-key       # https://aistudio.google.com/apikey
DB_URI=postgresql+psycopg://user:pass@localhost:5433/ragdb
QDRANT_URL=http://localhost:6333
CORS_ORIGINS=["http://localhost:3000"]
DAILY_QUESTION_LIMIT=10                  # per-IP daily cap on POST /api/chat/query
```

See `app/config.py` for every other setting (embedding/chat model names, chunk size, Redis
cache TTL, local Ollama overrides for offline dev, etc.) and its default.

### 4. Add documents and ingest them

Drop `.md`/`.pdf` files into `data/seed/` (a starter `about_rahul.md` is included — edit or
replace it with your own bio, resume, project write-ups, etc.), then run:

```bash
python scripts/ingest_documents.py
```

Re-run any time you add or edit files; pass `--force` to re-ingest everything. Run
`python scripts/delete_qdrant_collection.py` to wipe the collection and start over.

### 5. Run the development server

```bash
uvicorn app.main:app --reload --port 8000
```

The API is available at `http://localhost:8000` (or whichever port you pick — make sure the
frontend's `NEXT_PUBLIC_API_URL` points at the same one), with interactive docs at
`http://localhost:8000/docs`.

### 6. Run tests

```bash
pytest
```

## API summary

| Method | Path                        | Description |
| ------ | --------------------------- | ------------ |
| GET    | `/api/health`                | Health check |
| POST   | `/api/chat/query`             | Ask a question; streams the answer as newline-delimited JSON (rate limited - see below) |
| GET    | `/api/chat/list`               | List a session's conversations from the last 7 days, for the sidebar |
| GET    | `/api/chat/{chat_id}`          | Open a conversation — returns its latest chunk |
| GET    | `/api/chat/chunk/{chunk_id}`   | Fetch an older chunk of a conversation (pagination) |

## Rate limiting

`POST /api/chat/query` is capped at `DAILY_QUESTION_LIMIT` (default 10) questions per day
**per IP address**, via [`slowapi`](https://github.com/laurentS/slowapi)
(`app/core/rate_limiter.py`). This protects the paid embedding/LLM APIs from being drained by
one visitor or a bot, since there's no auth to otherwise gate usage per-user. Exceeding it
returns `429` with a `detail` message.

This is in-memory per backend process: counts reset on restart/redeploy, and won't be shared
correctly across multiple backend replicas (point `Limiter(storage_uri="redis://...")` at a
shared Redis instance if you scale beyond one process). It also assumes requests reach this app
directly — if you put a reverse proxy/load balancer in front of it, swap `get_remote_address`
for a forwarded-header-aware key function, or every visitor will appear to share the proxy's IP.

## Query caching (optional)

`app/services/cache.py` can cache identical questions in Redis for `QUERY_CACHE_TTL_SECONDS`
(default 1 hour) to save embedding/LLM calls on repeated questions. It's a pure optimization: if
Redis is unreachable or `ENABLE_QUERY_CACHE=false`, it becomes a no-op and the app behaves the
same, just without the savings.

## Docker

Run everything (Postgres + Qdrant + Redis + backend + frontend) via the root `docker-compose.yml`:

```bash
docker compose up --build
```

Or build/run just this service:

```bash
docker build -t rag-backend .
docker run --env-file .env -p 8000:8000 -v $(pwd)/data:/app/data rag-backend
```

See `DEPLOY.md` for a Render-specific deployment guide.

## Notes for future maintainers

This backend evolved from a more generic Postgres/pgvector RAG boilerplate into the current
Qdrant + conversation-history design. A few things in the repo are leftovers from that earlier
shape and aren't used by the live request path: `app/core/rag_pipeline.py` (superseded by
`graph/node.py` + `db/db_query.py`), the `pgvector`/`langchain-postgres`/`asyncpg`/`openai`
entries in `requirements.txt`, and `config.py`'s `system_prompt_path`/`VECTOR_STORE_PROVIDER`
settings (the real prompts live in `graph/system_prompt.py`; the vector store is Qdrant-only).
`data/documents.json` and `data/seed/README.md` are similarly stale. Worth cleaning up in a
dedicated pass rather than mixing it into an unrelated change.
