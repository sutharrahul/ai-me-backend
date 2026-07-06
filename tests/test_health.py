"""Tests for the health-check and root routes - these don't touch
Postgres or any AI provider, so they should always pass even without
`.env` configured, confirming the app boots and routes correctly."""


def test_health_check(client):
    """`GET /api/health` should always return 200 with status "ok",
    regardless of downstream dependency (DB/LLM) availability."""
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_root(client):
    """`GET /` (the bare backend URL) should return 200 with a friendly
    message, confirming the FastAPI app is mounted and serving traffic."""
    response = client.get("/")
    assert response.status_code == 200


def test_swagger_redirect(client):
    """`GET /swagger` should redirect to FastAPI's built-in Swagger UI."""
    response = client.get("/swagger", follow_redirects=False)
    assert response.status_code in {301, 302, 307, 308}
    assert response.headers["location"] == "/docs"
