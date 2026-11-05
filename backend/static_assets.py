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


# Equal quality prefers the smaller representation. A higher q always wins,
# including an explicit identity preference over a weakly accepted coding.
_ENCODING_PREFERENCE = ("br", "gzip", "identity")


def _accept_encoding_header(scope: Scope) -> Optional[str]:
    parts: list[str] = []
    for key, value in scope.get("headers") or []:
        if key.lower() == b"accept-encoding":
            parts.append(value.decode("latin1"))
    if not parts:
        return None
    return ",".join(parts)


def _encoding_qualities(scope: Scope) -> Optional[dict[str, float]]:
    """Map coding to q. ``None`` means the header is absent and any coding is acceptable."""
    header = _accept_encoding_header(scope)
    if header is None:
        return None
    qualities: dict[str, float] = {}
    for part in header.split(","):
        segments = [segment.strip() for segment in part.split(";") if segment.strip()]
        if not segments:
            continue
        coding = segments[0].lower()
        if not coding:
            continue
        quality = 1.0
        for segment in segments[1:]:
            if not segment.lower().startswith("q="):
                continue
            try:
                quality = float(segment.split("=", 1)[1].strip())
            except ValueError:
                quality = 0.0
            break
        qualities[coding] = min(1.0, max(0.0, quality))
    return qualities


def _quality_for(qualities: Optional[dict[str, float]], coding: str) -> Optional[float]:
    """Return a q value, or ``None`` when identity is only an unlisted fallback.

    An omitted identity coding stays available when every explicit coding is
    refused, but it does not outrank a coding the client actually accepted.
    ``*;q=0`` refuses that fallback unless identity itself has a positive q.
    """
    if qualities is None:
        return 1.0
    if coding in qualities:
        return qualities[coding]
    if "*" in qualities:
        return qualities["*"]
    if coding == "identity":
        return None
    return 0.0


def _candidate_paths(full_path: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    if os.path.isfile(full_path + ".br"):
        candidates.append(("br", full_path + ".br"))
    if os.path.isfile(full_path + ".gz"):
        candidates.append(("gzip", full_path + ".gz"))
    candidates.append(("identity", full_path))
    return candidates


def negotiated_asset_path(
    full_path: str, scope: Scope
) -> Optional[tuple[str, Optional[str]]]:
    """Pick the acceptable representation with the highest quality.

    Returns ``None`` when every available coding, including identity, is
    refused. Quality 0 is a refusal. A wildcard supplies the quality for
    codings the client did not name.
    """
    qualities = _encoding_qualities(scope)
    rank = {coding: index for index, coding in enumerate(_ENCODING_PREFERENCE)}
    ranked: list[tuple[float, int, str, Optional[str]]] = []
    identity_fallback: Optional[tuple[str, Optional[str]]] = None
    for coding, path in _candidate_paths(full_path):
        quality = _quality_for(qualities, coding)
        encoding = None if coding == "identity" else coding
        if quality is None:
            identity_fallback = (path, None)
            continue
        if quality <= 0:
            continue
        ranked.append((quality, -rank.get(coding, len(rank)), path, encoding))
    if ranked:
        _quality, _rank, path, encoding = max(ranked)
        return path, encoding
    return identity_fallback


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
        negotiated = negotiated_asset_path(str(full_path), scope)
        if negotiated is None:
            rejected = Response(status_code=406)
            rejected.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            rejected.headers["Vary"] = "Accept-Encoding"
            return rejected
        chosen, encoding = negotiated
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
        # Identity and compressed bodies are alternate representations of this URL.
        response.headers["Vary"] = "Accept-Encoding"
        if encoding:
            response.headers["Content-Encoding"] = encoding
        request_headers = _header_map(scope)
        if self.is_not_modified(response.headers, request_headers):
            return NotModifiedResponse(response.headers)
        return response


def _header_map(scope: Scope):
    from starlette.datastructures import Headers

    return Headers(scope=scope)
