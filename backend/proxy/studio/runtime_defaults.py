"""Default proxy runtime settings (restart-scoped)."""

RUNTIME_DEFAULTS = {
    "idle_ttl_seconds": None,
    "max_body_bytes": None,
    "max_loaded_models": None,
    "global_concurrency": None,
    "per_model_concurrency": None,
    "queue_timeout_seconds": 30,
    "request_timeout_seconds": None,
    "readiness_timeout_seconds": 2700,
    "drain_timeout_seconds": 30,
    "max_queue_size": 100,
    "stream_idle_timeout_seconds": None,
    "pool_timeout_seconds": 10,
    "max_connections": 200,
    "max_keepalive_connections": 40,
    "history_retention_days": 7,
    "history_max_rows": 100000,
    "default_output_tokens": 4096,
    "capture_request_payloads": False,
}
