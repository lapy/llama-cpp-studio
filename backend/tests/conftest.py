"""Pytest configuration and fixtures."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure backend is importable when running from repo root
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


@pytest.fixture
def client():
    """HTTP client against the FastAPI app (runs startup/shutdown lifespan)."""
    from backend.main import app

    return TestClient(app)
