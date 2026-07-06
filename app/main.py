"""FastAPI application entrypoint.

This is the file `uvicorn` points at (`app.main:app`). It wires together
everything the API needs: logging, settings, CORS, and route registration.
Keep this file thin - actual business logic belongs in `core/`, `services/`,
and the route modules under `api/routes/`.
"""

from typing import cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.types import ExceptionHandler


from app.api.routes import chat, documents, health
from app.config import get_settings
from app.core.rate_limiter import limiter, rate_limit_exceeded_handler
from app.utils.logger import setup_logging

# Configure root logging once, before anything else runs, so every module's
# `logger.info(...)`/`logger.error(...)` calls are actually printed.
setup_logging()

# `get_settings()` is cached (see config.py), so this reads `.env` once and
# every other part of the app reuses the same `Settings` instance.
settings = get_settings()

# The FastAPI app instance. `title`/`description`/`version` show up in the
# auto-generated docs at /docs and /redoc.
app = FastAPI(
    title=settings.app_name,
    description="Boilerplate backend for a Retrieval-Augmented Generation system.",
    version="0.1.0",
)

# Per-IP daily rate limiting (see `core/rate_limiter.py`) - protects the
# paid embedding/LLM APIs from being drained by one visitor or a bot,
# since there's no auth to otherwise gate usage per-user. `app.state.limiter`
# is where slowapi's decorator (`@limiter.limit(...)`, used in `chat.py`)
# looks up the shared limiter; the exception handler + middleware turn a
# limit breach into a 429 response with `Retry-After`/`X-RateLimit-*` headers.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, cast(ExceptionHandler, rate_limit_exceeded_handler))
app.add_middleware(SlowAPIMiddleware)

# Allows the frontend (running on a different origin, e.g. localhost:3000)
# to call this API from the browser. Without this, browsers block the
# requests due to the same-origin policy. `cors_origins` is configured in
# .env via CORS_ORIGINS. Added last (-> outermost in the middleware stack)
# so CORS headers are attached to every response, including one generated
# by another middleware (like a 429 from the rate limiter above).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount each route module under the shared `/api` prefix. To add a new group
# of endpoints: create a new file in `api/routes/`, define an `APIRouter`
# there, then `include_router` it here.
app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(chat.router, prefix=settings.api_prefix)
app.include_router(documents.router, prefix=settings.api_prefix)


@app.get("/")
def root() -> dict[str, str]:
    """Simple landing route so hitting the bare backend URL in a browser
    shows something useful instead of a 404, and points you to /docs."""
    return {
        "message": f"{settings.app_name} is running. See /swagger or /docs for the API."
    }


@app.get("/swagger", include_in_schema=False)
def swagger_ui() -> RedirectResponse:
    """Convenience route that sends users to FastAPI's Swagger UI."""
    return RedirectResponse(url="/docs")
