"""Per-IP daily rate limiting for public, unauthenticated endpoints.

There's no login/auth in this app (see `backend/README.md`'s "Extending"
notes), so the only identity signal available for an anonymous caller is
their IP address. This uses `slowapi` (a FastAPI-friendly wrapper around
the `limits` package) to cap how many requests a single IP can make per
day - see `daily_question_limit` in `config.py` and its use on
`POST /api/chat/query` in `api/routes/chat.py`.

Caveats (deliberately not solved here, to keep this boilerplate simple):
- Counts are kept in-memory per backend process. They reset on every
  restart/redeploy, and would NOT be shared correctly across multiple
  backend replicas/workers - each process would track its own separate
  counts. Point `Limiter(storage_uri="redis://...")` at a shared Redis
  instance if you ever scale beyond a single process.
- `get_remote_address` reads the request's direct client IP. If this app
  is ever deployed behind a reverse proxy/load balancer, every request
  will appear to come from the proxy's IP unless you swap in a
  forwarded-header-aware key function - which would turn this per-visitor
  limit into a single limit shared by every visitor.
"""

# from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Shared limiter instance - imported by `main.py` (to register it on the
# app + its exception handler) and by any route that needs a `@limiter.limit(...)`
# decorator (currently just `chat.py`).
limiter = Limiter(key_func=get_remote_address)


def rate_limit_exceeded_handler(_request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Turns a `RateLimitExceeded` into a 429 response shaped like every
    other error in this API (a `detail` string - see `chat.py`'s
    `HTTPException` usage), so the frontend's `handleResponse` (which
    reads `body.detail`) surfaces a meaningful message instead of falling
    back to a generic "Request failed with status 429". `_request` is
    unused but required by FastAPI's exception-handler signature
    (`(request, exc) -> Response`)."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": (
                f"Daily question limit reached ({exc.detail}). "
                "Please try again tomorrow."
            )
        },
    )
