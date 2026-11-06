"""1Cat release publication keeps the selected GitHub tag (review P2).

The release recipe selects `release_tag`; publication must put it in the row
and in the frozen snapshot inputs so a retry re-requests that TAG instead of
a version derived from the package version.
"""

from __future__ import annotations

import pytest

from backend.tests.venv_fixtures import ENGINE_ID, OneCatVllmInstaller as OneCatInstaller


@pytest.mark.asyncio
async def test_release_publication_preserves_tag_and_retry_uses_it(tmp_path, monkeypatch):
    installer = OneCatInstaller(
        log_path=str(tmp_path / "1cat.log"),
        base_dir=str(tmp_path / "1cat"),
    )
    ready: dict = {}
    snapshots: dict = {}

    async def fake_run_pip(args, stage, **kwargs):
        return 0

    async def fake_fetch(repository, version=None):
        assert repository == "1CatAI/1Cat-vLLM"
        return {
            "tag_name": "v1.3.0",
            "assets": [
                {
                    "name": "1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl",
                    "browser_download_url": "https://example.invalid/w.whl",
                }
            ],
        }

    installer._run_pip = fake_run_pip
    installer._update_progress_task = lambda *a, **k: _async_none()
    installer._ensure_venv = lambda: None
    installer._detect_installed_version = lambda *a, **k: "1.3.0+cu128"

    import backend.engines.python.release_assets as release_assets

    monkeypatch.setattr(release_assets, "fetch_github_release", fake_fetch)

    def fake_ready(pending, meta):
        ready.update(meta)
        return "published-name"

    installer._ready_pending_version = fake_ready
    installer._record_published_snapshot = lambda version, resolved: snapshots.update(resolved)

    import backend.engines.python.generic as generic_module

    async def fake_activate(*args, **kwargs):
        return None

    monkeypatch.setattr(generic_module, "activate_engine_version", fake_activate)

    execution = await installer._execute_release_plan(
        requested_version=None,
        pending_version="pip",
        existing_version=None,
        force_reinstall=False,
    )
    assert execution.status == "completed", execution.failed_step
    assert ready["install_type"] == "release"
    assert ready["release_tag"] == "v1.3.0"
    assert snapshots["install_type"] == "release"
    assert snapshots["release_tag"] == "v1.3.0"

    # Retry: the frozen tag wins over the package version.
    captured: dict = {}

    async def fake_install_release(version=None, force_reinstall=False, **kwargs):
        captured["version"] = version
        return {"task_id": "x"}

    installer.install_release = fake_install_release
    installer._install_snapshot_for = lambda version: {"resolved_inputs": {"release_tag": "v1.3.0"}}
    await installer.retry_existing_install(
        {
            "version": "v1",
            "install_dir": str(tmp_path / "install"),
            "install_type": "pip",
            "package_version": "1.3.0+cu128",
        }
    )
    assert captured["version"] == "v1.3.0"
    assert ENGINE_ID == installer.engine_id


def _async_none():
    async def _inner(*args, **kwargs):
        return None

    return _inner()
