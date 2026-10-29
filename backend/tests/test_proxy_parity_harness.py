from scripts.proxy_parity import normalize


def test_parity_normalization_removes_only_process_local_fields():
    assert normalize(
        {
            "model": "demo",
            "pid": 123,
            "port": 9999,
            "nested": [{"status": "ready", "created_at": "now"}],
        }
    ) == {"model": "demo", "nested": [{"status": "ready"}]}
