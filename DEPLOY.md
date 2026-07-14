# Deployment Guide

## Building Docker Image

```bash
docker build -t ai-me-backend:latest .
docker tag ai-me-backend:latest ai-me-backend:$(date +%Y.%m.%d)
```

## Running Locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

## Deploying to Render

Create the service as a **Docker**-runtime web service pointed at this directory's
`Dockerfile`. Render injects a `PORT` environment variable at runtime and routes traffic to
it — the `Dockerfile`'s `CMD` already reads `$PORT` (falling back to 8000 locally), so no
port configuration is needed on Render's side.

Render's own health check (and zero-downtime deploys) hit `GET /`, which returns a 200 JSON
response with no dependencies - it doesn't require the database or Qdrant to be reachable.

## Environment Variables

`.env` is gitignored and dockerignored — it never reaches the deploy platform.
Every variable below must be set in the Render dashboard (Environment tab).

Required:

| Variable | Value |
| --- | --- |
| `ENVIRONMENT` | `production` |
| `GOOGLE_API_KEY` | Gemini API key from https://aistudio.google.com/apikey |
| `DB_URI` | Postgres connection string (`postgresql+psycopg://…?sslmode=require`) |
| `QDRANT_URL` | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `CORS_ORIGINS` | Deployed frontend origin, e.g. `https://yourapp.vercel.app` — the default only allows `http://localhost:3000`, so the browser blocks the real frontend if this is unset |

**`QDRANT_URL`/`QDRANT_API_KEY` are checked at process startup, not just at query time** —
`app/graph/node.py` constructs a `QdrantVectorStore` at import time, which calls Qdrant to
confirm the collection exists. If Qdrant is unreachable or misconfigured, **the whole app
fails to boot** (every route, not just chat) rather than degrading gracefully. Double-check
these two values specifically if a deploy won't come up.

Optional (sensible defaults exist): `GEMINI_CHAT_MODEL`, `GEMINI_EMBEDDING_MODEL`,
`EMBEDDING_DIMENSIONS`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `DEFAULT_TOP_K`,
`DAILY_QUESTION_LIMIT`.

Set `ENABLE_QUERY_CACHE=false` unless a Redis instance is attached — with no
Redis reachable, caching silently becomes a no-op anyway.

**Do not set `LLM_PROVIDER`, `EMBEDDING_PROVIDER=ollama`, or any `OLLAMA_*`
variable on Render.** Ollama is a local-only dev provider that listens on
`localhost:11434`; there is no such server inside a Render container, so
selecting it makes every chat request fail with `ConnectError: Connection
refused`. The app now refuses to start in that configuration rather than
serving 500s.
