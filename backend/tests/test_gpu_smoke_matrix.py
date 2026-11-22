"""GPU smoke-matrix runner: help, dry-run validation, and a local fixture run."""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gpu-smoke-matrix.py"


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location("gpu_smoke_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_matrix(tmp_path: Path, **overrides) -> Path:
    matrix = {
        "base_url": "http://127.0.0.1:9",
        "key_env": "TEST_SMOKE_KEY",
        "prompt": "Say hello.",
        "max_tokens": 8,
        "timeout_seconds": 5,
        "models": ["org/model"],
    }
    matrix.update(overrides)
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix), encoding="utf-8")
    return path


class _FixtureHandler(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def _reply(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self.seen.append(
            {"method": "GET", "path": self.path, "auth": self.headers.get("Authorization")}
        )
        self._reply(200, {"object": "list", "data": [{"id": "org/model"}, {"id": "alias-one"}]})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.seen.append(
            {
                "method": "POST",
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "model": body.get("model"),
            }
        )
        if body.get("model") == "org/failing-model":
            self._reply(500, {"error": "boom"})
            return
        self._reply(
            200,
            {
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )

    def log_message(self, *_args) -> None:  # keep test output quiet
        return


@pytest.fixture
def fixture_server():
    _FixtureHandler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_help_documents_the_runner(smoke, capsys):
    with pytest.raises(SystemExit) as caught:
        smoke.main(["--help"])

    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert "gpu-smoke-matrix.py" in output
    assert "--dry-run" in output
    assert "--matrix" in output


def test_dry_run_validates_without_any_network_call(smoke, tmp_path, capsys, monkeypatch):
    matrix = _write_matrix(tmp_path)

    def forbidden_request(*_args, **_kwargs):
        raise AssertionError("dry-run must not make a network call")

    monkeypatch.setattr(smoke, "_request_json", forbidden_request)
    code = smoke.main(["--matrix", str(matrix), "--dry-run"])

    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["request_count"] == 1
    assert plan["requests"][0]["method"] == "POST"
    assert plan["requests"][0]["model"] == "org/model"
    assert plan["key_env"] == "TEST_SMOKE_KEY"


def test_invalid_matrix_exits_2_without_network(smoke, tmp_path, capsys, monkeypatch):
    matrix = _write_matrix(tmp_path, models=[])

    monkeypatch.setattr(
        smoke,
        "_request_json",
        lambda *_args, **_kwargs: pytest.fail("invalid matrix must not make a call"),
    )
    code = smoke.main(["--matrix", str(matrix)])

    assert code == 2
    assert "models must be a non-empty list" in capsys.readouterr().err


def test_normalization_rejects_credentials_in_the_base_url(smoke):
    with pytest.raises(smoke.MatrixError):
        smoke.normalize_matrix(
            {
                "base_url": "http://user:secret@127.0.0.1:2000",
                "models": ["org/model"],
            }
        )
    with pytest.raises(smoke.MatrixError):
        smoke.normalize_matrix({"base_url": "ftp://127.0.0.1:2000", "models": ["org/model"]})


def test_matrix_file_accepts_a_windows_utf8_bom(smoke, tmp_path):
    path = tmp_path / "matrix.json"
    path.write_text(
        json.dumps(
            {
                "base_url": "http://127.0.0.1:2000",
                "models": ["org/model"],
            }
        ),
        encoding="utf-8-sig",
    )

    matrix = smoke.load_matrix(path)

    assert matrix["targets"][0]["model_id"] == "org/model"


def test_matrix_run_records_alias_status_latency_and_usage(
    smoke, fixture_server, tmp_path, monkeypatch, capsys
):
    matrix = _write_matrix(
        tmp_path,
        base_url=fixture_server,
        models=[{"id": "org/model", "aliases": ["alias-one"]}],
    )
    output = tmp_path / "smoke.json"
    monkeypatch.setenv("TEST_SMOKE_KEY", "inference-secret-value")

    code = smoke.main(["--matrix", str(matrix), "--output", str(output)])

    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["models_endpoint"]["status"] == 200
    assert report["models_endpoint"]["model_count"] == 2
    assert [row["requested_model"] for row in report["results"]] == [
        "org/model",
        "alias-one",
    ]
    assert report["results"][0]["alias"] is None
    assert report["results"][1]["alias"] == "alias-one"
    assert all(row["listed"] for row in report["results"])
    assert all(row["status"] == 200 for row in report["results"])
    assert all(row["latency_ms"] >= 0 for row in report["results"])
    assert all(row["usage"]["completion_tokens"] == 1 for row in report["results"])
    assert report["summary"] == {"requests": 2, "succeeded": 2, "failed": 0}
    assert "inference-secret-value" not in output.read_text(encoding="utf-8")
    assert capsys.readouterr().err.startswith("wrote ")
    assert all(row["auth"] == "Bearer inference-secret-value" for row in _FixtureHandler.seen)


def test_failed_chat_call_produces_a_failing_exit_status(
    smoke, fixture_server, tmp_path, monkeypatch, capsys
):
    matrix = _write_matrix(
        tmp_path,
        base_url=fixture_server,
        models=[{"id": "org/failing-model"}],
    )
    output = tmp_path / "smoke.json"

    code = smoke.main(["--matrix", str(matrix), "--output", str(output)])

    assert code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"] == {"requests": 1, "succeeded": 0, "failed": 1}
    assert report["results"][0]["status"] == 500
    assert report["results"][0]["error"] == "http_500"
