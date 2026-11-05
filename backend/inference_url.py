"""Public inference URL used by client handoff examples."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def normalize_public_inference_url(value: object) -> str:
    """Return a scheme/host URL, or an empty string when unset.

    Raises ValueError for a non-empty value that is not an http(s) origin.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Public inference URL must start with http:// or https://")
    if parsed.username or parsed.password:
        raise ValueError("Public inference URL cannot include a username or password")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
