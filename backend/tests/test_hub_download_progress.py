"""Unit tests for Hugging Face download progress helpers."""

from backend.huggingface import _HubDownloadProgressState, _make_hub_download_tqdm


def test_hub_download_progress_state_snapshot():
    state = _HubDownloadProgressState()
    state.set(n=10, total=100)
    assert state.snapshot() == (10, 100)
    state.set(n=25)
    assert state.snapshot() == (25, 100)


def test_hub_download_tqdm_updates_shared_state():
    state = _HubDownloadProgressState()
    tqdm_cls = _make_hub_download_tqdm(state)
    bar = tqdm_cls(total=1000)
    assert state.snapshot()[1] == 1000
    bar.update(250)
    assert state.snapshot() == (250, 1000)
    bar.update(250)
    assert state.snapshot() == (500, 1000)
    bar.close()
