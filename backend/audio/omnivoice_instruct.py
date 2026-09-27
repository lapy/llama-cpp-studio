"""OmniVoice voice-design instruct vocabulary and validation."""

from __future__ import annotations

from typing import Iterable, List, Set

# Canonical comma-separated attributes from OmniVoice voice-design docs.
OMNIVOICE_INSTRUCT_ATTRIBUTES: Set[str] = {
    "male",
    "female",
    "child",
    "teenager",
    "young adult",
    "middle-aged",
    "elderly",
    "very low pitch",
    "low pitch",
    "moderate pitch",
    "high pitch",
    "very high pitch",
    "whisper",
    "american accent",
    "british accent",
    "australian accent",
    "chinese accent",
    "canadian accent",
    "indian accent",
    "korean accent",
    "portuguese accent",
    "russian accent",
    "japanese accent",
    "男",
    "女",
    "儿童",
    "少年",
    "青年",
    "中年",
    "老年",
    "极低音",
    "低音",
    "中音",
    "高音",
    "极高音",
    "耳语",
    "河南话",
    "陕西话",
    "四川话",
    "贵州话",
    "云南话",
    "桂林话",
    "济南话",
    "石家庄话",
    "甘肃话",
    "宁夏话",
    "青岛话",
    "东北话",
}

OMNIVOICE_INSTRUCT_EXAMPLE = "female, young adult, moderate pitch, british accent"


def _normalize_token(token: str) -> str:
    return " ".join(str(token or "").strip().lower().split())


def parse_omnivoice_instruct_items(value: str) -> List[str]:
    """Split an instruct string into comma-separated attribute items."""
    if not str(value or "").strip():
        return []
    return [
        " ".join(part.strip().split())
        for part in str(value).split(",")
        if part.strip()
    ]


def unsupported_omnivoice_instruct_items(value: str) -> List[str]:
    """Return attribute tokens that are not in the OmniVoice vocabulary."""
    unsupported: List[str] = []
    for item in parse_omnivoice_instruct_items(value):
        normalized = _normalize_token(item)
        if normalized not in OMNIVOICE_INSTRUCT_ATTRIBUTES:
            unsupported.append(item)
    return unsupported


def validate_omnivoice_instruct(value: str) -> List[str]:
    """Return user-facing validation errors for an OmniVoice instruct string."""
    unsupported = unsupported_omnivoice_instruct_items(value)
    if not unsupported:
        return []
    quoted = ", ".join(f"'{item}'" for item in unsupported)
    return [
        "OmniVoice instructions must be comma-separated canonical attributes, not free-form prose. "
        f"Unsupported attribute(s): {quoted}. "
        f"Example: {OMNIVOICE_INSTRUCT_EXAMPLE}"
    ]


def format_omnivoice_instruct_hint() -> str:
    """Short UI hint for the OmniVoice instruct field."""
    samples = ", ".join(
        sorted(
            attr
            for attr in OMNIVOICE_INSTRUCT_ATTRIBUTES
            if attr.isascii() and "accent" not in attr and "pitch" not in attr
        )[:8]
    )
    return (
        "Comma-separated OmniVoice attributes only (not natural-language prose). "
        f"Examples: {OMNIVOICE_INSTRUCT_EXAMPLE}. "
        f"Common tokens include {samples}, plus pitch/accent/dialect terms."
    )
