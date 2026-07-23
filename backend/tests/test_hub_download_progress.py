"""Unit tests for Hugging Face download progress helpers."""

from collections import deque

from backend.huggingface import (
    _HubDownloadProgressState,
    _download_speed_mbps,
    _make_hub_download_tqdm,
)


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


def test_download_speed_mbps_uses_rolling_window_not_poll_spikes():
    samples = deque()
    start = 1_000.0
    speed = 0.0
    # Steady ~10 MB/s for 3 seconds.
    for i in range(13):
        speed = _download_speed_mbps(
            samples,
            now=start + i * 0.25,
            bytes_downloaded=i * int(2.5 * 1024 * 1024),
            start_time=start,
            last_speed=speed,
        )
    assert 8.0 <= speed <= 12.0

    # Incomplete/tqdm catch-up jump: 400MB in one 0.25s poll (~1600 MB/s).
    jumped = _download_speed_mbps(
        samples,
        now=start + 13 * 0.25,
        bytes_downloaded=400 * 1024 * 1024,
        start_time=start,
        last_speed=speed,
    )
    assert jumped <= 12.5

    # After the jump, resume ~10 MB/s growth from the new baseline.
    base = 400 * 1024 * 1024
    t = start + 13 * 0.25
    for i in range(1, 13):
        t += 0.25
        speed = _download_speed_mbps(
            samples,
            now=t,
            bytes_downloaded=base + i * int(2.5 * 1024 * 1024),
            start_time=start,
            last_speed=speed,
        )
    assert 8.0 <= speed <= 12.0
