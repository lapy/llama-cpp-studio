from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from backend.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Normalized, high-level view of a GGUF model."""

    architecture: str
    layer_count: int
    block_count: int
    context_length: int
    parameter_count_display: Optional[str]
    vocab_size: Optional[int]
    embedding_length: Optional[int]
    attention_head_count: Optional[int]
    attention_head_count_kv: Optional[int]
    is_moe: bool
    expert_count: Optional[int]
    experts_used_count: Optional[int]
    raw_metadata: Dict[str, Any]


@dataclass
class TensorInfo:
    """Lightweight description of a tensor from GGUF metadata."""

    name: str
    shape: Tuple[int, ...]
    type_id: int
    offset: int


def _parse_numeric_with_suffix(value: Any) -> Optional[int]:
    """
    Parse human-readable numeric strings like '7B', '1.7M', or plain integers.

    Returns an integer number of parameters / units, or None if parsing fails.
    """
    if isinstance(value, (int, float)):
        return int(value) if value > 0 else None

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    # Normalize underscores and commas
    text = text.replace("_", "").replace(",", "")
    last = text[-1].upper()

    multiplier = 1
    number_part = text
    if last in ("K", "M", "B"):
        number_part = text[:-1]
        if last == "K":
            multiplier = int(1e3)
        elif last == "M":
            multiplier = int(1e6)
        else:
            multiplier = int(1e9)

    try:
        num = float(number_part)
        if num <= 0:
            return None
        return int(num * multiplier)
    except (ValueError, TypeError):
        return None


def _format_human_readable(value: Optional[int]) -> Optional[str]:
    """Format an integer as K/M/B string for display, or return None."""
    if value is None:
        return None
    if value >= 1_000_000_000:
        base = value / 1_000_000_000
        return f"{int(base)}B" if base.is_integer() else f"{base:.1f}B"
    if value >= 1_000_000:
        base = value / 1_000_000
        return f"{int(base)}M" if base.is_integer() else f"{base:.1f}M"
    if value >= 1_000:
        base = value / 1_000
        return f"{int(base)}K" if base.is_integer() else f"{base:.1f}K"
    return str(value)


def _find_numeric_candidates(
    metadata: Dict[str, Any],
    include_terms: Iterable[str],
    exclude_terms: Iterable[str] | None = None,
    max_value: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """Return (key, value) pairs whose key and numeric value match the filters."""
    exclude_terms = tuple(exclude_terms or ())
    include_terms = tuple(include_terms)

    candidates: List[Tuple[str, int]] = []
    for key, value in metadata.items():
        key_lower = key.lower()
        if not all(term in key_lower for term in include_terms):
            continue
        if any(term in key_lower for term in exclude_terms):
            continue

        parsed = _parse_numeric_with_suffix(value)
        if parsed is None:
            continue
        if max_value is not None and parsed > max_value:
            continue
        candidates.append((key, parsed))

    return candidates


_INTROSPECTION_CONFIG: Optional[Dict[str, Any]] = None


def _load_introspection_config() -> Dict[str, Any]:
    """
    Load optional JSON config for architecture-specific GGUF introspection rules.

    The file is expected at ``backend/gguf_introspection_config.json``. Any
    errors while loading are logged and result in an empty config.
    """
    global _INTROSPECTION_CONFIG
    if _INTROSPECTION_CONFIG is not None:
        return _INTROSPECTION_CONFIG

    cfg_path = os.path.join(os.path.dirname(__file__), "gguf_introspection_config.json")
    if not os.path.exists(cfg_path):
        _INTROSPECTION_CONFIG = {}
        return _INTROSPECTION_CONFIG

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                _INTROSPECTION_CONFIG = data
            else:
                logger.warning(
                    "gguf_introspection_config.json must contain a JSON object; got %s",
                    type(data),
                )
                _INTROSPECTION_CONFIG = {}
    except Exception as exc:
        logger.warning("Failed to load gguf_introspection_config.json: %s", exc)
        _INTROSPECTION_CONFIG = {}

    return _INTROSPECTION_CONFIG


class GgufIntrospector:
    """
    Data-driven GGUF model introspector.

    Consumes raw GGUF metadata and tensor descriptors and produces a normalized
    ModelInfo structure using generic key-pattern matching and simple heuristics.
    """

    # Sanity limits to defend against corrupted or adversarial metadata
    MAX_CONTEXT = 1_000_000_000
    MAX_LAYERS = 4096
    MAX_HEADS = 8192

    def __init__(
        self,
        metadata: Dict[str, Any],
        tensors: Dict[str, Dict[str, Any]] | None = None,
    ):
        self.metadata = metadata or {}
        self.tensors = tensors or {}
        self.architecture = str(
            self.metadata.get("general.architecture", "") or ""
        ).lower()
        self._config = _load_introspection_config()

    # Public orchestration -------------------------------------------------

    def build_model_info(self) -> ModelInfo:
        context_length = self._extract_context_length()
        block_count, layer_count = self._extract_layer_and_block_counts()
        param_count_int, param_display = self._extract_parameter_count()
        (
            attention_head_count,
            attention_head_count_kv,
        ) = self._extract_attention_heads()
        is_moe, expert_count, experts_used_count = self._extract_moe_info()
        embedding_length = self._extract_embedding_length()
        vocab_size = self._extract_vocab_size()

        return ModelInfo(
            architecture=self.architecture,
            layer_count=layer_count,
            block_count=block_count,
            context_length=context_length,
            parameter_count_display=param_display,
            vocab_size=vocab_size,
            embedding_length=embedding_length,
            attention_head_count=attention_head_count,
            attention_head_count_kv=attention_head_count_kv,
            is_moe=is_moe,
            expert_count=expert_count,
            experts_used_count=experts_used_count,
            raw_metadata=self.metadata,
        )

    # Property extractors --------------------------------------------------

    def _get_property_configs(self, prop: str) -> List[Dict[str, Any]]:
        """
        Return a list of config sections relevant for the given property.

        Order of precedence:
        1. Global section
        2. Architecture-specific sections whose ``match_arch`` entries are
           contained in the lowercased architecture string.
        """
        cfg = self._config or {}
        results: List[Dict[str, Any]] = []

        global_cfg = cfg.get("global")
        if isinstance(global_cfg, dict):
            prop_cfg = global_cfg.get(prop)
            if isinstance(prop_cfg, dict):
                results.append(prop_cfg)

        for name, section in cfg.items():
            if name == "global" or not isinstance(section, dict):
                continue
            match_arch = section.get("match_arch") or []
            if not isinstance(match_arch, list):
                continue
            if not any(
                isinstance(token, str) and token.lower() in self.architecture
                for token in match_arch
            ):
                continue
            prop_cfg = section.get(prop)
            if isinstance(prop_cfg, dict):
                results.append(prop_cfg)

        return results

    def _extract_context_length(self) -> int:
        candidates: List[int] = []

        # 1) Config-driven preferred keys
        for cfg in self._get_property_configs("context_length"):
            preferred = cfg.get("preferred_keys") or []
            for key in preferred:
                if key in self.metadata:
                    parsed = _parse_numeric_with_suffix(self.metadata[key])
                    if parsed is None or parsed <= 0 or parsed > self.MAX_CONTEXT:
                        continue
                    candidates.append(parsed)

            if candidates:
                break

            fallback_terms = cfg.get("fallback_terms") or []
            if fallback_terms:
                for _, value in _find_numeric_candidates(
                    self.metadata,
                    include_terms=tuple(fallback_terms),
                    exclude_terms=("generation", "prefill"),
                    max_value=self.MAX_CONTEXT,
                ):
                    candidates.append(value)

            if candidates:
                break

        # 2) Generic terms for context length (if config did not resolve it)
        if not candidates:
            terms_sets = [
                ("context",),
                ("model_max_length",),
                ("max_position_embeddings",),
                ("max_seq_len",),
                ("max_sequence_length",),
            ]

            for terms in terms_sets:
                for _, value in _find_numeric_candidates(
                    self.metadata,
                    include_terms=terms,
                    exclude_terms=("generation", "prefill"),
                    max_value=self.MAX_CONTEXT,
                ):
                    candidates.append(value)

        if not candidates:
            # As a last resort, look for any key that mentions both "max" and "length"
            for _, value in _find_numeric_candidates(
                self.metadata,
                include_terms=("max", "length"),
                max_value=self.MAX_CONTEXT,
            ):
                candidates.append(value)

        if not candidates:
            return 0

        best = max(candidates)
        if len(set(candidates)) > 1:
            logger.debug(
                "Multiple context length candidates detected %s, using max=%s",
                candidates,
                best,
            )
        return best

    def _extract_layer_and_block_counts(self) -> Tuple[int, int]:
        numeric_candidates: List[int] = []

        # 1) Config-driven preferred keys
        for cfg in self._get_property_configs("layer_count"):
            preferred = cfg.get("preferred_keys") or []
            for key in preferred:
                if key in self.metadata:
                    parsed = _parse_numeric_with_suffix(self.metadata[key])
                    if parsed is None or parsed <= 0 or parsed > self.MAX_LAYERS:
                        continue
                    numeric_candidates.append(parsed)

            if numeric_candidates:
                break

            fallback_terms = cfg.get("fallback_terms") or []
            if fallback_terms:
                for _, value in _find_numeric_candidates(
                    self.metadata,
                    include_terms=tuple(fallback_terms),
                    max_value=self.MAX_LAYERS,
                ):
                    numeric_candidates.append(value)

            if numeric_candidates:
                break

        # 2) Generic key-based candidates
        if not numeric_candidates:
            key_terms = [
                ("block_count",),
                ("layer_count",),
                ("n_layer",),
                ("num_layers",),
                ("num_hidden_layers",),
            ]

            for terms in key_terms:
                for _, value in _find_numeric_candidates(
                    self.metadata,
                    include_terms=terms,
                    max_value=self.MAX_LAYERS,
                ):
                    numeric_candidates.append(value)

        block_count = layer_count = 0
        if numeric_candidates:
            layer_count = max(numeric_candidates)
            block_count = layer_count
            if len(set(numeric_candidates)) > 1:
                logger.debug(
                    "Multiple layer/block candidates detected %s, using max=%s",
                    numeric_candidates,
                    layer_count,
                )
        else:
            # Tensor-based heuristic: count distinct block indices if names contain ".block."
            block_indices = self._infer_blocks_from_tensors()
            if block_indices:
                block_count = len(block_indices)
                layer_count = block_count + 1  # usually add output head
            else:
                # Fallback default for unknown models
                layer_count = 32
                block_count = 32
                logger.debug(
                    "No explicit layer/block metadata found; using default=%s", layer_count
                )

        return block_count, layer_count

    def _infer_blocks_from_tensors(self) -> List[int]:
        indices: set[int] = set()
        for name in self.tensors.keys():
            lower = name.lower()
            # Common patterns: layers.N., layer.N., blk.N., block.N.
            for marker in ("layers.", "layer.", "blk.", "block."):
                if marker in lower:
                    try:
                        after = lower.split(marker, 1)[1]
                        num_str = ""
                        for ch in after:
                            if ch.isdigit():
                                num_str += ch
                            else:
                                break
                        if num_str:
                            indices.add(int(num_str))
                    except Exception:
                        continue
        return sorted(indices)

    def _extract_parameter_count(self) -> Tuple[Optional[int], Optional[str]]:
        # Look for any key mentioning parameters
        raw_candidates: List[int] = []
        for key, value in self.metadata.items():
            key_lower = key.lower()
            if "param" not in key_lower:
                continue
            parsed = _parse_numeric_with_suffix(value)
            if parsed is not None and parsed > 0:
                raw_candidates.append(parsed)

        if not raw_candidates:
            return None, None

        best = max(raw_candidates)
        if len(set(raw_candidates)) > 1:
            logger.debug(
                "Multiple parameter count candidates detected %s, using max=%s",
                raw_candidates,
                best,
            )

        return best, _format_human_readable(best)

    def _extract_attention_heads(self) -> Tuple[Optional[int], Optional[int]]:
        # Attention heads
        att_candidates: List[int] = []
        for _, value in _find_numeric_candidates(
            self.metadata,
            include_terms=("attention", "head"),
            max_value=self.MAX_HEADS,
        ):
            att_candidates.append(value)

        head_count = max(att_candidates) if att_candidates else None

        # KV heads (GQA)
        kv_candidates: List[int] = []
        for _, value in _find_numeric_candidates(
            self.metadata,
            include_terms=("attention", "head", "kv"),
            max_value=self.MAX_HEADS,
        ):
            kv_candidates.append(value)

        head_count_kv = max(kv_candidates) if kv_candidates else None

        return head_count, head_count_kv

    def _extract_moe_info(self) -> Tuple[bool, Optional[int], Optional[int]]:
        architecture = str(self.metadata.get("general.architecture", "") or "").lower()
        is_moe = "moe" in architecture or "experts" in architecture

        expert_candidates: List[int] = []
        experts_used_candidates: List[int] = []

        for key, value in self.metadata.items():
            key_lower = key.lower()
            if "expert" not in key_lower and "experts" not in key_lower:
                continue

            parsed = _parse_numeric_with_suffix(value)
            if parsed is None or parsed <= 0:
                continue

            if any(term in key_lower for term in ("per_tok", "used", "active")):
                experts_used_candidates.append(parsed)
            else:
                expert_candidates.append(parsed)

        expert_count = max(expert_candidates) if expert_candidates else None
        experts_used_count = (
            max(experts_used_candidates) if experts_used_candidates else None
        )

        if expert_count:
            is_moe = True

        # Default active experts if only total experts is known
        if is_moe and experts_used_count is None and expert_count:
            if expert_count >= 64:
                experts_used_count = 8
            elif expert_count >= 32:
                experts_used_count = 4
            else:
                experts_used_count = 2

        return is_moe, expert_count, experts_used_count

    def _extract_embedding_length(self) -> Optional[int]:
        # First try explicit metadata
        candidates: List[int] = []
        for _, value in _find_numeric_candidates(
            self.metadata,
            include_terms=("embedding",),
        ):
            candidates.append(value)

        if candidates:
            return max(candidates)

        # Fallback: use tensor shapes for token embeddings
        best: Optional[int] = None
        for name, info in self.tensors.items():
            lower = name.lower()
            if not any(term in lower for term in ("token_emb", "embed_tokens", "tok_embeddings", "tok_embed")):
                continue
            shape = info.get("shape") or []
            if len(shape) >= 2:
                dim = int(shape[-1])
                if best is None or dim > best:
                    best = dim
        return best

    def _extract_vocab_size(self) -> Optional[int]:
        # Prefer scalar vocab size keys
        candidates: List[int] = []
        for _, value in _find_numeric_candidates(
            self.metadata,
            include_terms=("vocab_size",),
        ):
            candidates.append(value)

        if candidates:
            return max(candidates)

        # Fallback: derive from embedding matrix first dimension
        best: Optional[int] = None
        for name, info in self.tensors.items():
            lower = name.lower()
            if not any(term in lower for term in ("token_emb", "embed_tokens", "tok_embeddings", "tok_embed")):
                continue
            shape = info.get("shape") or []
            if len(shape) >= 2:
                size = int(shape[0])
                if best is None or size > best:
                    best = size
        return best

