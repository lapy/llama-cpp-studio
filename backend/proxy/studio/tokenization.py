"""Engine-owned input-token counts; never guess multimodal/context token costs.

Contracts: llama.cpp chat/responses input_tokens and tokenize APIs; vLLM's
TokenizeChatRequest/TokenizeCompletionRequest; SGLang's ``/v1/tokenize``
chat/prompt request (returns ``count`` and token IDs). See
docs/proxy-operations.md. Unsupported engine builds return None; callers choose
a visible text estimate or require exact accounting. A successful count does
not generate completion tokens.

Optional counting routes are probed once per engine build revision: a 404/405/
501 disables that route for the revision instead of retrying every request.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import httpx

from backend.proxy.studio.accounting import AdmissionError
from backend.engines.capabilities import capabilities_for, probe_cache


@dataclass(frozen=True)
class InputCount:
    tokens: int
    source: str


def supported_engine(engine: str | None) -> bool:
    return capabilities_for(engine).count_adapter != "none"


def count_route_supported(engine: str | None, path: str) -> bool:
    """Whether this engine/route combination has a declared count adapter."""
    adapter = capabilities_for(engine).count_adapter
    if adapter == "none":
        return False
    if path.endswith("/responses") and adapter in ("vllm_tokenize", "sglang_tokenize"):
        return False
    if path.endswith("/completion") and not path.endswith("/completions"):
        return False
    if path.endswith("/generate"):
        return False
    return True


def unsupported_reason(engine: str | None) -> str:
    capabilities = capabilities_for(engine)
    if capabilities.count_adapter == "none":
        return f"{capabilities.engine or 'this engine'} has no verified exact input-count adapter"
    return ""


async def _post(client, upstream, endpoint, payload, *, engine="", revision=""):
    # A build revision scopes the cache. Without one (unit harnesses, older
    # callers) always probe so a prior 404 cannot hide support in another build.
    cacheable = bool(revision)
    cached = probe_cache.get(engine, revision, endpoint) if cacheable else None
    if cached is False:
        return None
    try:
        async with client.stream("POST", upstream + endpoint, json=payload, timeout=10) as response:
            if response.status_code in (404, 405, 501):
                if cacheable:
                    probe_cache.record(engine, revision, endpoint, False)
                return None
            if response.status_code >= 400:
                raise AdmissionError(
                    "Engine could not count this request's input tokens",
                    "token_count_rejected",
                    400 if response.status_code < 500 else 503,
                )
            data = bytearray()
            async for chunk in response.aiter_bytes():
                if len(data) + len(chunk) > 4 * 1024 * 1024:
                    raise AdmissionError(
                        "Engine token-count response exceeded its size limit",
                        "token_count_unavailable",
                        503,
                    )
                data.extend(chunk)
        result = await asyncio.to_thread(json.loads, data)
    except (httpx.HTTPError, ValueError, UnicodeDecodeError) as exc:
        raise AdmissionError(
            "Engine token counting is unavailable", "token_count_unavailable", 503
        ) from exc
    if not isinstance(result, dict):
        raise AdmissionError(
            "Engine returned an invalid token count", "token_count_unavailable", 503
        )
    if cacheable:
        probe_cache.record(engine, revision, endpoint, True)
    return result


def _count(payload, *names):
    for name in names:
        value = payload.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10**12:
            return value
        if (
            isinstance(value, list)
            and value
            and all(
                isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 10**12
                for item in value
            )
        ):
            return sum(value)
    raise AdmissionError("Engine returned an invalid token count", "token_count_unavailable", 503)


def _request_model(payload: dict) -> dict:
    return {"model": payload["model"]} if payload.get("model") else {}


async def count_input(
    client,
    upstream: str,
    engine: str,
    path: str,
    payload: dict,
    *,
    revision: str = "",
) -> InputCount | None:
    adapter = capabilities_for(engine).count_adapter
    if adapter == "none":
        return None
    if path.endswith("/chat/completions") and adapter == "llama_input_tokens":
        endpoint = "/v1/chat/completions/input_tokens"
        result = await _post(client, upstream, endpoint, payload, engine=engine, revision=revision)
        return InputCount(_count(result, "input_tokens"), endpoint) if result is not None else None
    if path.endswith("/responses") and adapter == "llama_input_tokens":
        endpoint = "/v1/responses/input_tokens"
        result = await _post(client, upstream, endpoint, payload, engine=engine, revision=revision)
        return InputCount(_count(result, "input_tokens"), endpoint) if result is not None else None
    if adapter == "vllm_tokenize":
        return await _count_vllm(client, upstream, engine, path, payload, revision)
    if adapter == "sglang_tokenize":
        return await _count_sglang(client, upstream, engine, path, payload, revision)
    if adapter == "llama_input_tokens":
        return await _count_llama_tokenize(client, upstream, engine, path, payload, revision)
    return None


async def _count_vllm(client, upstream, engine, path, payload, revision):
    # Responses input has no verified vLLM tokenize template mapping. Do not
    # send an unrelated prompt shape that could 400 and block auto fallback.
    if path.endswith("/responses"):
        return None
    if path.endswith(("/embeddings", "/completions")) and not path.endswith("/chat/completions"):
        content = payload.get("input") if path.endswith("/embeddings") else payload.get("prompt")
        inputs = (
            content
            if isinstance(content, list) and all(isinstance(item, str) for item in content)
            else [content]
        )
        total = 0
        for item in inputs:
            if not isinstance(item, str):
                return None
            request = {
                "prompt": item,
                "add_special_tokens": payload.get("add_special_tokens", True),
                **_request_model(payload),
            }
            result = await _post(
                client, upstream, "/tokenize", request, engine=engine, revision=revision
            )
            if result is None:
                return None
            total += _count(result, "count")
        return InputCount(total, "/tokenize")
    fields = {
        "model",
        "messages",
        "prompt",
        "tools",
        "documents",
        "chat_template",
        "chat_template_kwargs",
        "add_generation_prompt",
        "continue_final_message",
        "add_special_tokens",
    }
    request = {name: value for name, value in payload.items() if name in fields}
    result = await _post(client, upstream, "/tokenize", request, engine=engine, revision=revision)
    return InputCount(_count(result, "count"), "/tokenize") if result is not None else None


_SGLANG_TOKENIZE_FIELDS = (
    "model",
    "messages",
    "tools",
    "tool_choice",
    "reasoning_effort",
    "continue_final_message",
    "chat_template_kwargs",
    "add_generation_prompt",
    "add_special_tokens",
)


async def _count_sglang(client, upstream, engine, path, payload, revision):
    if path.endswith("/responses"):
        return None
    if path.endswith("/chat/completions"):
        request = {
            name: value for name, value in payload.items() if name in _SGLANG_TOKENIZE_FIELDS
        }
        if not request.get("messages"):
            return None
        result = await _post(
            client,
            upstream,
            "/v1/tokenize",
            request,
            engine=engine,
            revision=revision,
        )
        return InputCount(_count(result, "count"), "/v1/tokenize") if result is not None else None
    if path.endswith(("/completions", "/embeddings")):
        content = payload.get("input") if path.endswith("/embeddings") else payload.get("prompt")
        inputs = (
            content
            if isinstance(content, list) and all(isinstance(item, str) for item in content)
            else [content]
        )
        total = 0
        for item in inputs:
            if not isinstance(item, str):
                return None
            request = {
                "prompt": item,
                "add_special_tokens": payload.get("add_special_tokens", True),
                **_request_model(payload),
            }
            result = await _post(
                client,
                upstream,
                "/v1/tokenize",
                request,
                engine=engine,
                revision=revision,
            )
            if result is None:
                return None
            total += _count(result, "count")
        return InputCount(total, "/v1/tokenize")
    # Responses input conversion is not part of SGLang's TokenizeRequest
    # contract. Exact mode fails closed instead of guessing a template mapping.
    return None


async def _count_llama_tokenize(client, upstream, engine, path, payload, revision):
    if not path.endswith(("/completions", "/embeddings")):
        return None
    content = payload.get("input") if path.endswith("/embeddings") else payload.get("prompt")
    inputs = (
        content
        if isinstance(content, list) and all(isinstance(item, str) for item in content)
        else [content]
    )
    total = 0
    for item in inputs:
        if not isinstance(item, str):
            return None
        result = await _post(
            client,
            upstream,
            "/tokenize",
            {
                "content": item,
                "add_special": payload.get("add_special_tokens", True),
                "parse_special": True,
            },
            engine=engine,
            revision=revision,
        )
        if result is None:
            return None
        tokens = result.get("tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(token, int) and not isinstance(token, bool) for token in tokens
        ):
            raise AdmissionError(
                "Engine returned invalid token IDs", "token_count_unavailable", 503
            )
        total += len(tokens)
    return InputCount(total, "/tokenize")
