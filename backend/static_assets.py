"""Serve content-hashed frontend assets with negotiated gzip or Brotli."""

from __future__ import annotations

import gzip
import mimetypes
import os
from typing import Optional

from starlette.responses import FileResponse, Response
from starlette.staticfiles import NotModifiedResponse, StaticFiles
from starlette.types import Scope

try:
    import brotli
except ImportError:  # pragma: no cover - optional codec
    brotli = None


def _accept_tokens(scope: Scope) -> set[str]:
    for key, value in scope.get("headers") or []:
        if key.lower() != b"accept-encoding":
            continue
        text = value.decode("latin1").lower()
        return {part.split(";", 1)[0].strip() for part in text.split(",") if part.strip()}
    return set()


def negotiated_asset_path(full_path: str, scope: Scope) -> tuple[str, Optional[str]]:
    """Pick a precompressed sibling when the client accepts it."""
    accepted = _accept_tokens(scope)
    if "br" in accepted and os.path.isfile(full_path + ".br"):
        return full_path + ".br", "br"
    if "gzip" in accepted and os.path.isfile(full_path + ".gz"):
        return full_path + ".gz", "gzip"
    return full_path, None


def ensure_precompressed_assets(directory: str) -> int:
    """Write missing ``.gz`` (and ``.br`` when brotli is installed) siblings.

    Existing compressed files newer than the source are left alone. The
    compressed bytes are what HTTP serves; SSE routes are not part of this tree.
    """
    if not os.path.isdir(directory):
        return 0
    written = 0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(".gz") or name.endswith(".br"):
                continue
            source = os.path.join(root, name)
            try:
                source_mtime = os.path.getmtime(source)
            except OSError:
                continue
            written += _compress_sibling(source, source + ".gz", source_mtime, _gzip_bytes)
            if brotli is not None:
                written += _compress_sibling(source, source + ".br", source_mtime, _brotli_bytes)
    return written


def _compress_sibling(source: str, dest: str, source_mtime: float, encode) -> int:
    try:
        if os.path.isfile(dest) and os.path.getmtime(dest) >= source_mtime:
            return 0
        with open(source, "rb") as handle:
            payload = encode(handle.read())
        tmp = dest + ".tmp"
        with open(tmp, "wb") as handle:
            handle.write(payload)
        os.replace(tmp, dest)
        return 1
    except OSError:
        return 0


def _gzip_bytes(payload: bytes) -> bytes:
    return gzip.compress(payload, compresslevel=6)


def _brotli_bytes(payload: bytes) -> bytes:
    return brotli.compress(payload)


class HashedAssetFiles(StaticFiles):
    """Immutable hashed files, served precompressed when the browser allows it."""

    def file_response(
        self,
        full_path: str,
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        chosen, encoding = negotiated_asset_path(str(full_path), scope)
        if chosen != str(full_path):
            stat_result = os.stat(chosen)
        media_type = mimetypes.guess_type(str(full_path))[0] or "application/octet-stream"
        response = FileResponse(
            chosen,
            status_code=status_code,
            stat_result=stat_result,
            media_type=media_type,
        )
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        if encoding:
            response.headers["Content-Encoding"] = encoding
            response.headers["Vary"] = "Accept-Encoding"
        request_headers = _header_map(scope)
        if self.is_not_modified(response.headers, request_headers):
            return NotModifiedResponse(response.headers)
        return response


def _header_map(scope: Scope):
    from starlette.datastructures import Headers

    return Headers(scope=scope)
