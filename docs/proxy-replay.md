# Proxy request replay

_Last reviewed: 2026-11-16._

The Studio proxy keeps two different kinds of request evidence:

1. **History metadata** in SQLite (method, path, model, status, latency, token counts,
   sizes). This is bounded by the retention days / max rows settings and survives
   restarts.
2. **Payload capture** in memory only, when `capture_request_payloads` is enabled.
   The last 200 requests keep a sanitized header map and up to 256 KiB of request body.
   Captures are replaced as new requests arrive, are never written to the accounting
   database or logs, and are discarded when the proxy restarts.

Payload capture is **off by default**. Turning it on is an explicit operator choice
because prompts and completions can contain sensitive data.

## Redaction

`authorization`, `proxy-authorization`, `cookie`, `set-cookie`, `x-api-key` and
`api-key` are removed from the captured header map and listed in
`captured_redacted_headers`. Hop-by-hop headers (`host`, `content-length`,
`connection`, `transfer-encoding`, `accept-encoding`) are dropped because the replay
client must compute them again.

## Replay

`POST /api/proxy/requests/{id}/replay` re-sends the captured method, path, headers and
body directly to the still-loaded engine. It is an operator action:

- it never starts a model or falls back to a different one;
- if the model is no longer loaded it returns `409 MODEL_NOT_LOADED`;
- if the capture was evicted or disabled it returns `404 PAYLOAD_NOT_CAPTURED`;
- quota accounting is not applied to the replay itself.

The inspector shows the replay control only for requests that have a captured payload.
