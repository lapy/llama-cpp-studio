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

Headers whose name is a known credential header, or that has an `auth`, `key`,
`token`, `secret`, `cookie` or `password` token (`x-api-key`, `x-goog-api-key`,
`x-custom-api-key`, …), are removed from the captured map and listed in
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

## Authorization and audit

`/api/proxy` is behind the management access middleware: local loopback sessions or an
authenticated remote session with CSRF protection. Replay is therefore an operator
action, not an inference-key action.

Each successful or failed replay appends a bounded proxy-activity record with
`type: replay`/`replay_failed`, model, source request ID, path, status and error name.
The captured body is never copied into activity or logs. Replay intentionally bypasses
quota accounting because it is a debugging operation; the audit record is the control
that makes that bypass observable.
