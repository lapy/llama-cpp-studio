"""Rewrite audio.cpp's embedded SvelteKit UI for a path prefix.

The upstream HTML is built with ``base: ""`` and fetches ``/v1/...``. llama-swap's
``/upstream/{model}/`` leaves the browser pathname prefixed, so SvelteKit hydrates
a 404 unless ``base`` and those absolute fetches match the public prefix.
"""

from __future__ import annotations

import json
import re

_SVELTEKIT_BASE_RE = re.compile(
    r"(__sveltekit_[A-Za-z0-9]+)\s*=\s*\{\s*base:\s*\"\"\s*\}",
    re.MULTILINE,
)


def _clean_prefix(prefix: str) -> str:
    value = str(prefix or "").strip()
    if not value.startswith("/"):
        value = f"/{value}"
    return value.rstrip("/")


def public_audio_ui_prefix(model_id: str, *, root: str) -> str:
    model = str(model_id or "").strip().strip("/")
    if not model:
        raise ValueError("model id required")
    base = _clean_prefix(root)
    if not base:
        raise ValueError("UI prefix root required")
    return f"{base}/{model}"


def studio_audio_ui_prefix(model_id: str) -> str:
    return public_audio_ui_prefix(model_id, root="/audio-cpp-ui")


def llama_swap_upstream_prefix(model_id: str) -> str:
    return public_audio_ui_prefix(model_id, root="/upstream")


def _bridge_script(prefix: str) -> str:
    prefix_js = json.dumps(prefix)
    return (
        "<script>(function(){"
        f"var p={prefix_js};"
        "function rw(u){"
        'if(typeof u!=="string")return u;'
        'if(u==="/health"||u.startsWith("/health?")||u.startsWith("/v1/"))return p+u;'
        "return u;}"
        "var f=window.fetch;"
        "window.fetch=function(i,n){"
        'if(typeof i==="string")i=rw(i);'
        "else if(i&&typeof i.url===\"string\")i=new Request(rw(i.url),i);"
        "return f.call(this,i,n);};"
        "var o=XMLHttpRequest.prototype.open;"
        "XMLHttpRequest.prototype.open=function(m,u){"
        "arguments[1]=rw(u);return o.apply(this,arguments);};"
        "var E=window.EventSource;"
        "if(E){window.EventSource=function(u,c){return new E(rw(u),c);};"
        "window.EventSource.prototype=E.prototype;}"
        "})();</script>"
    )


def rewrite_audio_cpp_ui_html(
    html: bytes | str,
    model_id: str | None = None,
    *,
    prefix: str | None = None,
) -> bytes:
    """Point SvelteKit ``base`` at ``prefix`` and bridge ``/v1`` fetches."""
    resolved = _clean_prefix(prefix) if prefix else ""
    if not resolved:
        resolved = studio_audio_ui_prefix(model_id or "")
    text = html.decode("utf-8") if isinstance(html, (bytes, bytearray)) else str(html)
    text, count = _SVELTEKIT_BASE_RE.subn(
        lambda match: f"{match.group(1)} = {{ base: {json.dumps(resolved)} }}",
        text,
        count=1,
    )
    if count == 0:
        raise ValueError("audio.cpp UI HTML is missing the SvelteKit base assignment")
    bridge = _bridge_script(resolved)
    lowered = text.lower()
    head = lowered.find("<head")
    insert_at = lowered.find(">", head) + 1 if head >= 0 else 0
    if insert_at <= 0:
        text = bridge + text
    else:
        text = text[:insert_at] + bridge + text[insert_at:]
    return text.encode("utf-8")
