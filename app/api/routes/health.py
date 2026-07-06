"""Health-check endpoint.

Used by monitoring tools, load balancers, Docker healthchecks, or just the
frontend's `checkHealth()` call to know whether the backend is up and
reachable. Deliberately has no dependency on the database or any AI
provider, so it stays fast and reports "ok" even if Postgres/Gemini are
temporarily unavailable - use this to distinguish "server is down" from
"server is up but a downstream dependency is failing".
"""

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Returns a static "ok" status plus basic app info (name, environment).
    Extend this if you want deeper checks later (e.g. pinging Postgres),
    but keep in mind that would make this endpoint slower and able to fail
    for reasons unrelated to the API process itself being alive."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.environment,
    )
