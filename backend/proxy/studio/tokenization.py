"""Engine-owned input-token counts; never guess multimodal/context token costs.

Contracts: llama.cpp chat/responses input_tokens and tokenize APIs; vLLM's
TokenizeChatRequest/TokenizeCompletionRequest. See docs/proxy-operations.md.
Unsupported engine versions return None; callers choose a visible text estimate
or require exact accounting. A successful count does not generate completion tokens.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import httpx

from backend.proxy.studio.accounting import AdmissionError

LLAMA_ENGINES = {"llama_cpp", "ik_llama", "unsloth_llama"}
VLLM_ENGINES = {"vllm", "1cat_vllm"}


@dataclass(frozen=True)
class InputCount:
    tokens: int
    source: str


def supported_engine(engine: str | None) -> bool:
    return engine in LLAMA_ENGINES | VLLM_ENGINES


async def _post(client, upstream, endpoint, payload):
    try:
        async with client.stream("POST", upstream + endpoint, json=payload, timeout=10) as response:
            if response.status_code in (404, 405, 501):
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
    return result


def _count(payload, *names):
    for name in names:
        value = payload.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10**12:
            return value
    raise AdmissionError("Engine returned an invalid token count", "token_count_unavailable", 503)


async def count_input(
    client, upstream: str, engine: str, path: str, payload: dict
) -> InputCount | None:
    if not supported_engine(engine):
        return None
    if path.endswith("/chat/completions") and engine in LLAMA_ENGINES:
        endpoint = "/v1/chat/completions/input_tokens"
        result = await _post(client, upstream, endpoint, payload)
        return InputCount(_count(result, "input_tokens"), endpoint) if result is not None else None
    if path.endswith("/responses"):
        endpoint = "/v1/responses/input_tokens"
        result = await _post(client, upstream, endpoint, payload)
        return InputCount(_count(result, "input_tokens"), endpoint) if result is not None else None
    if (
        engine in VLLM_ENGINES
        and path.endswith(("/embeddings", "/completions"))
        and not path.endswith("/chat/completions")
    ):
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
            }
            if payload.get("model"):
                request["model"] = payload["model"]
            result = await _post(client, upstream, "/tokenize", request)
            if result is None:
                return None
            total += _count(result, "count")
        return InputCount(total, "/tokenize")
    if engine in VLLM_ENGINES and path.endswith(("/chat/completions", "/completions")):
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
        result = await _post(client, upstream, "/tokenize", request)
        return InputCount(_count(result, "count"), "/tokenize") if result is not None else None
    if engine in LLAMA_ENGINES and path.endswith(("/completions", "/embeddings")):
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
    return None
