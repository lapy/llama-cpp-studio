"""Pytest configuration and fixtures."""

import os
import socket
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Isolate every test process from the developer's real data directory.
os.environ["STUDIO_DATA_DIR"] = tempfile.mkdtemp(prefix="studio-pytest-")
os.environ.setdefault("STUDIO_ACCESS_MODE", "local")

# Ensure backend is importable when running from repo root
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _host_allowed(address) -> bool:
    if isinstance(address, str):
        return True
    if not isinstance(address, tuple) or not address:
        return True
    host = address[0]
    if not isinstance(host, str):
        return True
    return host in _LOCAL_HOSTS or host.startswith("127.")


@pytest.fixture(autouse=True)
def reset_operation_supervisor():
    """Resource locks are in-process and must not leak between tests."""
    from backend.operations import supervisor as operation_supervisor

    operation_supervisor._supervisor = operation_supervisor.OperationSupervisor()
    yield
    operation_supervisor._supervisor = None


@pytest.fixture(autouse=True)
def block_unexpected_network(monkeypatch, request):
    """Fail unit tests that open a non-local socket. Opt out with @pytest.mark.network."""
    if request.node.get_closest_marker("network"):
        yield
        return
    real_connect = socket.socket.connect

    def connect(self, address):
        if not _host_allowed(address):
            raise RuntimeError(f"unexpected network access to {address[0]}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    yield


@pytest.fixture
def client():
    """HTTP client against the FastAPI app without entering the process lifespan."""
    from backend.main import app

    return TestClient(app)


@pytest.fixture
def lifespan_client():
    """HTTP client that actually runs startup and shutdown."""
    from backend.main import app

    with TestClient(app) as instance:
        yield instance
