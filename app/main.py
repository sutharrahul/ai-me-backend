"""FastAPI application entrypoint.

This is the file `uvicorn` points at (`app.main:app`). It wires together
logging, settings, CORS, rate limiting, and route registration.
Keep this file thin — actual business logic belongs in `core/` and
`services/`.
"""

from typing import cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.types import ExceptionHandler

from app.api.routes import chat, health
from app.config import get_settings
from app.core.rate_limiter import limiter, rate_limit_exceeded_handler
from app.utils.logger import setup_logging

# Configure root logging once, before anything else runs, so every module's
# logger calls are printed to stdout and the log file.
setup_logging()

# `get_settings()` is cached (see config.py), so .env is read once and every
# other part of the app reuses the same `Settings` instance.
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Personal portfolio chatbot — answers visitors' questions about Rahul Suthar using RAG.",
    version="1.0.0",
)

# Per-IP daily rate limiting (see `core/rate_limiter.py`) — protects the
# paid Gemini API from being drained by a single visitor or a bot, since
# there's no auth to otherwise gate usage per-user.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, cast(ExceptionHandler, rate_limit_exceeded_handler))
app.add_middleware(SlowAPIMiddleware)

# Allows the frontend (running on a different origin, e.g. localhost:3000)
# to call this API from the browser. Without this, browsers block requests
# due to the same-origin policy. Added last (→ outermost in the middleware
# stack) so CORS headers appear on every response, including 429s from the
# rate limiter above.
app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(chat.router, prefix=settings.api_prefix)


@app.get("/")
def root() -> dict[str, str]:
    """Landing route — shows a friendly message and points to /docs."""
    return {
        "message": f"{settings.app_name} is running. See /docs for the API."
    }


@app.get("/swagger", include_in_schema=False)
def swagger_ui() -> RedirectResponse:
    """Convenience redirect to FastAPI's built-in Swagger UI."""
    return RedirectResponse(url="/docs")
