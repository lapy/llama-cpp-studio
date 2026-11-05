from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from devcontrol.logs import read_logs


class FakeCompose:
    def __init__(self, data: Path) -> None:
        self.config = SimpleNamespace(data=data, backend_port=18081, api_token="")

    def logs(self, service: str, *, tail: int = 200, since: str = ""):
        return SimpleNamespace(stdout=f"{service} error: boom\n{service} ok\n", stderr="")


def test_build_logs_are_bounded_and_filterable(tmp_path: Path) -> None:
    data = tmp_path / "data"
    logs = data / "logs"
    logs.mkdir(parents=True)
    (logs / "build-one.log").write_text("configure\nbuild error: nvcc failed\n", encoding="utf-8")
    compose = FakeCompose(data)
    result = read_logs(compose, source="build", tail=20, level="error", max_bytes=64_000)
    assert result["source"] == "build"
    assert any("nvcc failed" in line for line in result["lines"])
    assert result["bytes"] > 0


def test_app_logs_filter_error_lines() -> None:
    compose = FakeCompose(Path("/nonexistent"))
    result = read_logs(compose, source="app", service="backend", tail=20, level="error")
    assert result["source"] == "app"
    assert result["lines"] == ["backend error: boom"]
