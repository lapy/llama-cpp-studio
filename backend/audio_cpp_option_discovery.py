"""Source-assisted audio.cpp option discovery when model --help is incomplete.

Precedence for Studio model profiles:
1. Normalized model-aware ``--help`` rows
2. ``loader.cpp`` CliOptionInfo lists (what help *should* advertise)
3. Docs markdown session/load/request option tables (types/defaults)
4. ``session.cpp`` accepted-key allowlists (unadvertised but valid)
5. Thin curated Studio overlays (path pickers / OpenAI field mapping)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.cli_help_parsers import (
    _audio_apply_keyed_option_typing,
    _audio_options_from_value_spec,
    _human_label,
    group_params_into_sections,
)

_TRANSPORT_PREFIX_RE = re.compile(
    r"^--(?:session|load|request)-option\s+",
    re.IGNORECASE,
)
_OPTION_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_CPP_TRIPLE_RE = re.compile(
    r'\{\s*"([^"]+)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,?\s*\}',
    re.S,
)
# e.g. std::string(kMelBandRoformerFamily) + ".weight_type"
_CPP_CONCAT_TRIPLE_RE = re.compile(
    r'\{\s*std::string\([^)]*\)\s*\+\s*"\.([^"]+)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,?\s*\}',
    re.S,
)
_CPP_CONCAT_KEY_RE = re.compile(
    r'std::string\([^)]*\)\s*\+\s*"\.([^"]+)"'
)
_CPP_OPTION_ASSIGN_RE = re.compile(
    r"(?:\.|\s)(?P<kind>session|load|request)_options\s*=\s*\{",
    re.S,
)
_CPP_ALLOW_KEY_RE = re.compile(
    r'key\s*(?:!=|==)\s*"(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)"'
)
_CPP_FIND_OPTION_RE = re.compile(
    r'(?:find_option|parse_\w+_option)\([^,]+,\s*\{(?P<body>[^}]*)\}',
)
_CPP_FIND_OPTION_KEY_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_.-]*)"')
_CPP_OPTIONS_FIND_RE = re.compile(
    r'options\.options\.find\(\s*"(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)"\s*\)'
)
_CPP_PARSE_FLOAT_KEYS_RE = re.compile(
    r'parse_(?:float|int|bool|size_mb|string|u32)_option\([^,]+,\s*\{(?P<body>[^}]*)\}',
)
_CPP_REQUEST_OPTIONS_RE = re.compile(
    r'(?:parse_\w+_option|find_option)\(\s*request\.options\s*,\s*\{(?P<body>[^}]*)\}',
)
_COMMON_REQUEST_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "min_tokens",
        "seed",
        "do_sample",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "num_beams",
        "length_penalty",
        "language",
        "lookahead_tokens",
        "keep_language_tags",
        "return_timestamps",
        "enable_thinking",
        "punctuation",
        "stream",
    }
)
# Require an explicit transport flag so table cells like `silero_vad` are ignored.
_DOCS_SESSION_ROW_RE = re.compile(
    r"\|\s*`--(?P<transport>session|load|request)-option\s+"
    r"(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)"
    r"(?:=(?P<default_or_spec>[^`|]+))?`\s*"
    r"\|\s*(?P<type>[^|]+?)\s*"
    r"\|\s*(?P<default>[^|]+?)\s*"
    r"\|\s*(?P<desc>[^|]+?)\s*\|",
)
_FAMILY_DOC_HINTS = {
    "sortformer_diar": ("sortformer", "diar"),
    "silero_vad": ("silero", "vad"),
    "marblenet_vad": ("marblenet", "vad"),
    "qwen3_asr": ("qwen3", "asr"),
    "qwen3_tts": ("qwen3", "tts"),
    "nemotron_asr": ("nemotron",),
    "vibevoice_asr": ("vibevoice", "asr"),
    "htdemucs": ("demucs", "htdemucs"),
    "mel_band_roformer": ("roformer", "mel-band", "mel_band"),
}
_REQUEST_INPUT_FLAGS = frozenset(
    {
        "source_audio",
        "target_voice",
        "prosody_ref",
        "style_ref",
        "voice_ref",
        "audio",
        "text",
        "target_text",
        "style_ref_text",
    }
)
# CLI I/O paths/flags from --help that are not Studio request-default knobs.
_REQUEST_CLI_ONLY_FLAGS = frozenset(
    {
        "batch_audio_dir",
        "out",
        "out_dir",
        "text_out",
        "segments_out",
        "turns_out",
        "words_out",
        "vad_chunks_out",
        "audio_out",
        "json_out",
        "log_file",
        "progress",
        "quiet",
        "verbose",
    }
)
_COMMON_WEIGHT_TYPES = ["native", "f32", "f16", "bf16", "q8_0"]
_CONV_WEIGHT_TYPES = ["native", "f32", "f16"]

_SECTION_FOR_KIND = {
    "session": ("model_session_options", "Model session options", "session_option"),
    "load": ("model_load_options", "Model load options", "load_option"),
    "request": ("model_request_options", "Model request options", "request_option"),
}


def normalize_audio_option_key(raw: str) -> Tuple[str, Optional[str]]:
    """Strip transport prefixes and ``=default`` suffixes from option names."""
    text = str(raw or "").strip()
    if not text:
        return "", None
    text = _TRANSPORT_PREFIX_RE.sub("", text).strip()
    default_hint = None
    if "=" in text and " " not in text.split("=", 1)[0]:
        key, _, maybe_default = text.partition("=")
        key = key.strip()
        maybe_default = maybe_default.strip()
        if key and _OPTION_KEY_RE.match(key) and maybe_default:
            return key, maybe_default
    if text.startswith("--"):
        return text.lstrip("-").replace("-", "_"), None
    return text, None


def _infer_value_spec(
    key: str,
    *,
    value_name: str = "",
    description: str = "",
    docs_type: str = "",
) -> str:
    value_name = str(value_name or "").strip()
    docs_type = str(docs_type or "").strip()
    combined = f"{value_name} {docs_type} {description}".lower()
    key_l = key.lower()

    if _audio_options_from_value_spec(value_name):
        return value_name if value_name.startswith("<") else f"<{value_name}>"

    if "true|false" in combined or docs_type.lower() in {"bool", "boolean"}:
        return "<true|false>"
    if key_l.endswith("mem_saver") or key_l.endswith(".mem_saver"):
        return "<true|false>"
    if "weight_type" in key_l:
        if "conv" in key_l or "audio_encoder" in key_l:
            return "<" + "|".join(_CONV_WEIGHT_TYPES) + ">"
        return "<" + "|".join(_COMMON_WEIGHT_TYPES) + ">"
    if key_l.endswith("_path") or key_l.endswith("_dir") or "model_path" in key_l:
        return "<path>"
    if key_l.endswith("_mb") or "arena" in key_l or key_l.endswith("_slots"):
        return "<n>"
    if docs_type:
        token = docs_type.split()[0].strip("`")
        if "|" in token:
            return f"<{token}>"
        if token.lower() in {"float", "int", "integer", "path", "dir", "mb", "n"}:
            return f"<{token.lower() if token.lower() != 'integer' else 'n'}>"
    if value_name:
        return value_name if value_name.startswith("<") else f"<{value_name}>"
    return "<value>"


def _param_from_option(
    key: str,
    *,
    kind: str,
    value_spec: str = "",
    description: str = "",
    default: Any = None,
    discovery_source: str,
) -> Optional[dict]:
    key = str(key or "").strip()
    if not key or not _OPTION_KEY_RE.match(key):
        return None
    if key.startswith("--"):
        return None
    if key in _REQUEST_INPUT_FLAGS and kind != "request":
        return None
    section_id, section_label, transport = _SECTION_FOR_KIND[kind]
    value_spec = _infer_value_spec(key, value_name=value_spec, description=description)
    row = {
        "key": key,
        "label": _human_label(key.replace(".", "_")),
        "flags": [],
        "primary_flag": f"--{transport.replace('_', '-')}",
        "aliases": [],
        "value_spec": value_spec,
        "description": description,
        "default": default,
        "section_id": section_id,
        "section_label": section_label,
        "source": "source_discovery",
        "discovery_source": discovery_source,
    }
    row = _audio_apply_keyed_option_typing(
        row,
        option_name=key,
        option_value_spec=value_spec,
        description=description,
    )
    if default is not None and row.get("default") is None:
        row["default"] = default
    row["scope"] = transport
    row["transport"] = "key_value_option"
    row["read_only"] = transport == "request_option"
    row["reserved"] = False
    row["emission"] = {
        "transport": "key_value_option",
        "flag": row["primary_flag"],
        "option_key": key,
    }
    return row


_FAMILY_DIR_ALIASES = {
    "htdemucs": ["demucs"],
    "mel_band_roformer": ["roformer"],
    "demucs": ["demucs"],
}


def _family_model_dirs(source_root: str, family: str) -> List[str]:
    models_root = os.path.join(source_root, "src", "models")
    if not os.path.isdir(models_root):
        return []
    family = str(family or "").strip()
    if not family:
        return []
    matches: List[str] = []
    candidates = [family, *_FAMILY_DIR_ALIASES.get(family, [])]
    for name in candidates:
        direct = os.path.join(models_root, name)
        if os.path.isdir(direct) and direct not in matches:
            matches.append(direct)
    for dirpath, dirnames, filenames in os.walk(models_root):
        base = os.path.basename(dirpath)
        if base in candidates and dirpath not in matches:
            matches.append(dirpath)
        if "loader.cpp" in filenames:
            try:
                text = open(
                    os.path.join(dirpath, "loader.cpp"), encoding="utf-8", errors="ignore"
                ).read()
            except OSError:
                continue
            if re.search(
                rf'return\s+"{re.escape(family)}"\s*;',
                text,
            ) and dirpath not in matches:
                matches.append(dirpath)
        # prune deep noise
        dirnames[:] = [d for d in dirnames if d not in {".git", "build"}]
    return matches


def _kind_for_block_header(header: str) -> Optional[str]:
    header = header.lower()
    if "session_options" in header:
        return "session"
    if "load_options" in header:
        return "load"
    if "request_options" in header:
        return "request"
    return None


def _extract_balanced_brace_block(text: str, open_index: int) -> str:
    """Return the inside of a ``{...}`` block starting at ``open_index``."""
    if open_index < 0 or open_index >= len(text) or text[open_index] != "{":
        return ""
    depth = 0
    for idx in range(open_index, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : idx]
    return ""


def _loader_triples_from_block(block: str, family: str = "") -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    for raw_name, value_name, description in _CPP_TRIPLE_RE.findall(block):
        triples.append((raw_name, value_name, description))
    for suffix, value_name, description in _CPP_CONCAT_TRIPLE_RE.findall(block):
        prefix = str(family or "").strip()
        key = f"{prefix}.{suffix}" if prefix else suffix
        triples.append((key, value_name, description))
    return triples


def parse_loader_cli_options(loader_text: str, family: str = "") -> List[dict]:
    """Extract CliOptionInfo triples from loader.cpp text."""
    params: List[dict] = []
    text = loader_text or ""
    # Infer family from loader when not provided.
    if not family:
        fam_match = re.search(r'return\s+"([A-Za-z0-9_]+)"\s*;', text)
        if fam_match:
            family = fam_match.group(1)
        else:
            fam_match = re.search(
                r'kMelBandRoformerFamily\s*=\s*"([^"]+)"',
                text,
            )
            if fam_match:
                family = fam_match.group(1)
    for match in _CPP_OPTION_ASSIGN_RE.finditer(text):
        kind = match.group("kind")
        block = _extract_balanced_brace_block(text, match.end() - 1)
        if not block:
            continue
        for raw_name, value_name, description in _loader_triples_from_block(
            block, family=family
        ):
            key, default_hint = normalize_audio_option_key(raw_name)
            if not key:
                continue
            raw_stripped = raw_name.strip()
            is_input_flag = key in _REQUEST_INPUT_FLAGS or raw_stripped in {
                "--source-audio",
                "--target-voice",
                "--prosody-ref",
                "--style-ref",
                "--audio",
                "--text",
            }
            if is_input_flag:
                # Misplaced input flags inside session_options (chatterbox).
                param = _param_from_option(
                    key,
                    kind="request",
                    value_spec=value_name or "wav",
                    description=description,
                    discovery_source="loader.cpp",
                )
                if param:
                    param["read_only"] = True
                    params.append(param)
                continue
            if raw_stripped.startswith("--") and "-option" not in raw_stripped:
                continue
            value_spec = value_name
            if (
                default_hint
                and default_hint.lower() in {"true", "false"}
                and not value_name
            ):
                value_spec = "true|false"
            param = _param_from_option(
                key,
                kind=kind,
                value_spec=value_spec,
                description=description,
                default=(
                    default_hint.lower() == "true"
                    if default_hint and default_hint.lower() in {"true", "false"}
                    else default_hint
                ),
                discovery_source="loader.cpp",
            )
            if param:
                params.append(param)
    return params


def parse_session_accepted_keys(session_text: str, family: str) -> List[dict]:
    """Extract accepted session/request option keys from model C++ sources."""
    text = session_text or ""
    session_keys: List[str] = []
    request_keys: List[str] = []

    for match in _CPP_ALLOW_KEY_RE.finditer(text):
        session_keys.append(match.group("key"))
    for pattern in (_CPP_FIND_OPTION_RE, _CPP_PARSE_FLOAT_KEYS_RE):
        for match in pattern.finditer(text):
            # Skip request.options call sites; handled below.
            if "request.options" in match.group(0):
                continue
            session_keys.extend(_CPP_FIND_OPTION_KEY_RE.findall(match.group("body")))
    for match in _CPP_OPTIONS_FIND_RE.finditer(text):
        session_keys.append(match.group("key"))
    for match in _CPP_CONCAT_KEY_RE.finditer(text):
        suffix = match.group(1)
        if family:
            session_keys.append(f"{family}.{suffix}")
    for match in _CPP_REQUEST_OPTIONS_RE.finditer(text):
        request_keys.extend(_CPP_FIND_OPTION_KEY_RE.findall(match.group("body")))

    prefix = f"{family}." if family else ""
    # Bare keys that are still valid session options for some families
    # (Sortformer postprocess, Silero VAD threshold, etc.).
    bare_allowed = {
        "speaker_threshold",
        "speaker_min_frames",
        "speaker_pad_frames",
        "session_len_sec",
        "threshold",
        "neg_threshold",
        "min_speech_duration_ms",
        "min_silence_duration_ms",
        "speech_pad_ms",
        "max_speech_duration_s",
    }
    params: List[dict] = []
    seen = set()

    def add(key: str, kind: str) -> None:
        if key in seen:
            return
        seen.add(key)
        param = _param_from_option(
            key,
            kind=kind,
            description=(
                f"Accepted by {family} {kind} path "
                "(not always advertised in --help)."
            ),
            discovery_source="session.cpp",
        )
        if param:
            params.append(param)

    for key in session_keys:
        if "." in key:
            if prefix and not (
                key.startswith(prefix) or key.startswith("qwen3_forced_aligner.")
            ):
                continue
        elif key not in bare_allowed:
            continue
        add(key, "session")

    for key in request_keys:
        if "." in key and prefix and not key.startswith(prefix):
            continue
        if "." not in key and key not in _COMMON_REQUEST_KEYS and key not in bare_allowed:
            # Still keep unknown bare request keys — they came from request.options.
            pass
        add(key, "request")

    return params


def parse_docs_option_tables(docs_text: str) -> List[dict]:
    """Parse markdown tables that document --session-option / load / request keys."""
    params: List[dict] = []
    for match in _DOCS_SESSION_ROW_RE.finditer(docs_text or ""):
        key = match.group("key").strip()
        transport = (match.group("transport") or "session").strip().lower()
        raw_default = (match.group("default") or "").strip().strip("`")
        docs_type = (match.group("type") or "").strip()
        desc = (match.group("desc") or "").strip()
        default_or_spec = (match.group("default_or_spec") or "").strip()
        value_spec = default_or_spec if "|" in default_or_spec else docs_type
        default: Any = None
        if raw_default and raw_default.lower() not in {"not set", "—", "-", "n/a"}:
            if raw_default.lower() in {"true", "false"}:
                default = raw_default.lower() == "true"
            elif raw_default.isdigit():
                default = int(raw_default)
            else:
                try:
                    default = float(raw_default) if "." in raw_default else raw_default
                except ValueError:
                    default = raw_default
        kind = transport if transport in {"session", "load", "request"} else "session"
        param = _param_from_option(
            key,
            kind=kind,
            value_spec=value_spec,
            description=desc,
            default=default,
            discovery_source="docs",
        )
        if param:
            params.append(param)
    return params


def _docs_row_belongs_to_family(
    docs_text: str, match_start: int, family: str, key: str
) -> bool:
    family = str(family or "").strip().lower()
    key_l = str(key or "").strip().lower()
    if key_l.startswith(f"{family}."):
        return True
    if family == "qwen3_asr" and key_l.startswith("qwen3_forced_aligner."):
        return True
    if key_l == "return_timestamps" and family.endswith("_asr"):
        return True
    # Bare docs keys (speaker_threshold, threshold, …) only when this family is
    # named nearby — avoid stealing sibling-family rows from shared docs files.
    if "." in key_l:
        return False
    window = docs_text[max(0, match_start - 1200) : match_start].lower()
    if family in window:
        return True
    for hint in _FAMILY_DOC_HINTS.get(family, ()):
        # Prefer distinctive multi-char tokens; skip overly generic stems.
        if len(hint) >= 5 and hint.lower() in window and family.split("_")[0] in window:
            return True
    return False


def _docs_paths_for_family(source_root: str, family: str) -> List[str]:
    docs_dir = os.path.join(source_root, "docs")
    if not os.path.isdir(docs_dir):
        return []
    family = str(family or "").strip().lower()
    aliases = {
        "qwen3_asr": ["qwen3.md", "asr.md"],
        "qwen3_tts": ["qwen3.md", "tts.md"],
        "qwen3_forced_aligner": ["qwen3.md"],
        "nemotron_asr": ["asr.md"],
        "vibevoice_asr": ["asr.md"],
        "hviske_asr": ["asr.md"],
        "higgs_audio_stt": ["asr.md"],
        "citrinet_asr": ["asr.md"],
        "vevo2": ["vevo2.md"],
        "seed_vc": ["seed_vc.md"],
        "stable_audio": ["stable_audio.md"],
        "ace_step": ["ace_step.md"],
        "sortformer_diar": ["speech_analysis.md"],
        "silero_vad": ["speech_analysis.md"],
        "marblenet_vad": ["speech_analysis.md"],
        "htdemucs": ["sep.md", "source_separation.md"],
        "mel_band_roformer": ["sep.md", "source_separation.md"],
        "chatterbox": ["tts.md"],
        "index_tts2": ["tts.md"],
        "irodori_tts": ["tts.md"],
        "moss_tts_nano": ["tts.md"],
        "moss_tts_local": ["tts.md"],
        "omnivoice": ["tts.md"],
        "pocket_tts": ["tts.md"],
        "supertonic": ["tts.md"],
        "voxcpm2": ["tts.md"],
        "vibevoice": ["tts.md"],
        "miotts": ["tts.md"],
        "miocodec": ["tts.md", "codec.md"],
        "heartmula": ["tts.md", "music.md"],
    }
    names = list(aliases.get(family, []))
    stem = family.replace("_asr", "").replace("_tts", "").replace("_vad", "")
    for candidate in (f"{family}.md", f"{stem}.md"):
        if candidate not in names:
            names.append(candidate)
    paths = []
    for name in names:
        path = os.path.join(docs_dir, name)
        if os.path.isfile(path) and path not in paths:
            paths.append(path)
    return paths


def _source_rank(param: dict) -> int:
    return {
        "loader.cpp": 3,
        "session.cpp": 2,
        "docs": 1,
    }.get(str(param.get("discovery_source") or ""), 0)


def _scope_rank(scope: str) -> int:
    return {
        "load_option": 3,
        "session_option": 2,
        "request_option": 1,
    }.get(str(scope or ""), 0)


def discover_family_options(source_root: str, family: str) -> List[dict]:
    """Discover session/load/request options from an audio.cpp source checkout."""
    source_root = os.path.abspath(str(source_root or ""))
    family = str(family or "").strip()
    if not source_root or not family or not os.path.isdir(source_root):
        return []

    by_key: Dict[str, dict] = {}

    def upsert(param: Optional[dict], *, prefer: bool = False) -> None:
        if not param:
            return
        key = str(param.get("key") or "")
        if not key:
            return
        # Unprefixed keys from docs are usually request/runtime flags, not session KV.
        if (
            "." not in key
            and param.get("scope") == "session_option"
            and param.get("discovery_source") == "docs"
        ):
            param = dict(param)
            param["scope"] = "request_option"
            param["section_id"] = "model_request_options"
            param["section_label"] = "Model request options"
            param["transport"] = "key_value_option"
            param["read_only"] = True
            param["primary_flag"] = "--request-option"
            param["emission"] = {
                "transport": "key_value_option",
                "flag": "--request-option",
                "option_key": key,
            }
        existing = by_key.get(key)
        if not existing:
            by_key[key] = param
            return
        chosen = param
        keep = existing
        if prefer or _source_rank(param) > _source_rank(existing):
            keep, chosen = param, existing
        elif _source_rank(param) == _source_rank(existing) and _scope_rank(
            param.get("scope")
        ) > _scope_rank(existing.get("scope")):
            keep, chosen = param, existing
        merged = dict(keep)
        for field in (
            "type",
            "value_kind",
            "scalar_type",
            "options",
            "value_spec",
            "description",
            "default",
            "label",
            "discovery_source",
            "scope",
            "section_id",
            "section_label",
            "transport",
            "read_only",
            "primary_flag",
            "emission",
        ):
            if merged.get(field) in (None, "", [], {}) and chosen.get(field) not in (
                None,
                "",
                [],
                {},
            ):
                merged[field] = chosen[field]
            elif field == "description" and chosen.get("description"):
                if len(str(chosen.get("description") or "")) > len(
                    str(merged.get("description") or "")
                ):
                    merged["description"] = chosen["description"]
        by_key[key] = merged

    for model_dir in _family_model_dirs(source_root, family):
        loader = os.path.join(model_dir, "loader.cpp")
        if os.path.isfile(loader):
            try:
                text = open(loader, encoding="utf-8", errors="ignore").read()
            except OSError:
                text = ""
            for param in parse_loader_cli_options(text, family=family):
                upsert(param, prefer=True)
        for dirpath, _dirnames, filenames in os.walk(model_dir):
            for filename in filenames:
                if not filename.endswith(".cpp"):
                    continue
                path = os.path.join(dirpath, filename)
                try:
                    text = open(path, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                for param in parse_session_accepted_keys(text, family):
                    upsert(param)

    for docs_path in _docs_paths_for_family(source_root, family):
        try:
            text = open(docs_path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for match in _DOCS_SESSION_ROW_RE.finditer(text):
            # Re-parse through helper for typing, but gate by family context.
            key = match.group("key").strip()
            if not _docs_row_belongs_to_family(text, match.start(), family, key):
                continue
            chunk = text[match.start() : match.end()]
            for param in parse_docs_option_tables(chunk):
                key = str(param.get("key") or "")
                if key == "return_timestamps":
                    param = dict(param)
                    param["scope"] = "request_option"
                    param["section_id"] = "model_request_options"
                    param["section_label"] = "Model request options"
                    param["read_only"] = True
                    param["primary_flag"] = "--request-option"
                    param["emission"] = {
                        "transport": "key_value_option",
                        "flag": "--request-option",
                        "option_key": key,
                    }
                upsert(param)

    return sorted(
        by_key.values(), key=lambda item: (item.get("scope"), item.get("key"))
    )


def merge_discovered_options_into_sections(
    sections: Sequence[dict],
    discovered: Sequence[dict],
) -> List[dict]:
    """Fill gaps in help-parsed sections with source-discovered options."""
    params: List[dict] = []
    seen_keys = set()
    for section in sections or []:
        for param in section.get("params") or []:
            key = str(param.get("key") or "")
            scope = str(param.get("scope") or "")
            # Fix malformed keys like chatterbox.mem_saver=true if still present.
            normalized, default_hint = normalize_audio_option_key(key)
            if normalized and normalized != key and "." in normalized:
                param = dict(param)
                param = _audio_apply_keyed_option_typing(
                    param,
                    option_name=normalized,
                    option_value_spec=param.get("value_spec")
                    or ("true|false" if default_hint in {"true", "false"} else ""),
                    description=str(param.get("description") or ""),
                )
                if default_hint in {"true", "false"} and param.get("default") is None:
                    param["default"] = default_hint == "true"
                param["scope"] = scope or param.get("scope")
                key = normalized
            if key and key not in seen_keys:
                seen_keys.add(key)
                params.append(param)

    for param in discovered or []:
        key = str(param.get("key") or "")
        if key and key not in seen_keys:
            seen_keys.add(key)
            params.append(dict(param))

    return group_params_into_sections(params)


def scanned_request_field_groups(sections: Sequence[dict]) -> List[dict]:
    """Build Request Defaults groups from scanned/discovered request_option rows."""
    fields = []
    for section in sections or []:
        for param in section.get("params") or []:
            if param.get("scope") != "request_option":
                continue
            key = str(param.get("key") or "")
            if (
                not key
                or key in _REQUEST_INPUT_FLAGS
                or key in _REQUEST_CLI_ONLY_FLAGS
            ):
                continue
            field = {
                "key": key,
                "label": param.get("label") or _human_label(key.replace(".", "_")),
                "type": param.get("type") or "string",
                "description": param.get("description") or "",
                "nested": True,
                "options_key": key,
            }
            if param.get("options"):
                field["choices"] = [
                    opt.get("value") for opt in param["options"] if opt.get("value")
                ]
            if param.get("default") is not None:
                field["default"] = param["default"]
            fields.append(field)
    if not fields:
        return []
    return [
        {
            "id": "scanned_request_options",
            "label": "Model request options",
            "description": (
                "Autodetected from audiocpp_cli --model … --help and/or audio.cpp "
                "source (loader/docs). Stored under request options."
            ),
            "fields": fields,
        }
    ]


__all__ = [
    "discover_family_options",
    "merge_discovered_options_into_sections",
    "normalize_audio_option_key",
    "parse_docs_option_tables",
    "parse_loader_cli_options",
    "parse_session_accepted_keys",
    "scanned_request_field_groups",
]
