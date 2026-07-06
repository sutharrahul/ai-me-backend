"""Shared pytest fixtures for the backend test suite.

pytest automatically discovers `conftest.py` and makes any fixtures
defined here available to every test file in this directory without an
explicit import.
"""

import sys
from pathlib import Path

# Ensures `from app...` imports resolve when tests are run from a
# different working directory (e.g. `pytest` invoked from the repo root
# instead of `backend/`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Provides a FastAPI `TestClient` wrapping the real `app` instance,
    so tests can make requests (`client.get(...)`, `client.post(...)`)
    against the API in-process, without needing a running `uvicorn`
    server or real network calls."""
    return TestClient(app)
