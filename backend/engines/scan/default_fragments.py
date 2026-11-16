"""Default-value fragments parsed out of engine help prose.

Two jobs, one implementation:

* ``normalize_default_fragment`` trims and unwraps a default token from the
  llama-server / audio.cpp help text (comma tails, quotes, "0 = disabled"
  prose). It was extracted from ``help_parsers`` verbatim so this module is the
  single home for default-fragment parsing.
* ``default_from_description`` reads only defaults the help text *states as
  values*. audio.cpp help describes derived behavior in prose too - "optional
  backend implementation; default preserves existing dispatch" - and a naive
  token grab turns "preserves" into a string default. A default is accepted
  only when it is numeric, boolean, quoted, explicitly introduced with
  ``default:``/``default=``, or followed by punctuation; a value followed by
  more prose is documentation, not a value.

``is_documentation_default_artifact`` names the second case so the launch path
can ignore a value a model config copied from the prose (and log why) instead of
sending the engine a flag value that cannot be valid.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

#: ``default cuda`` / ``defaults to 4`` / ``default: 4`` / ``default='auto'``.
#: The clause must follow punctuation, column alignment or the start of the
#: text, which rejects noun phrases such as "the model's default prompt" while
#: accepting the help text's ``; default 3``, ``(default: 4)`` and
#: ``  default cuda`` forms.
_DEFAULT_CLAUSE_RE = re.compile(
    r"(?P<lead>^|[,;(:\[=.!?]\s*|\s{2,})"
    r"default(?:\s+to)?(?:\s*(?P<sep>[:=])\s*|\s+)(?P<val>[^\s,;]+)",
    re.IGNORECASE | re.MULTILINE,
)
_NUMERIC_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?$")
_BOOLEAN_TOKENS = frozenset({"true", "false", "on", "off", "yes", "no"})
#: Characters that end a default clause, so the token before them is a value.
_CLAUSE_END = frozenset(",;.:)]}/\u2014\u2013-")


def looks_like_structured_default(value: str) -> bool:
    """JSON / list / dataclass defaults must keep commas and closing parens."""
    s = (value or "").strip()
    if not s:
        return False
    if s[0] in "{[(":
        return True
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*\(", s))


def normalize_default_fragment(raw: str) -> Optional[str]:
    """Trim and unwrap a single default token from engine help prose."""
    value = (raw or "").strip().strip(",")
    if not value:
        return None
    structured = looks_like_structured_default(value)
    # "40, 0 = disabled" / "8192, -1 - no limit" / "-1; -1 = disabled"
    numeric_head = re.match(r"^([-+]?\d+(?:\.\d+)?)\s*[,;]", value)
    if numeric_head:
        value = numeric_head.group(1)
    else:
        # "0 = model training context size"
        numeric_eq = re.match(r"^([-+]?\d+(?:\.\d+)?)\s*=\s*[A-Za-z]", value)
        if numeric_eq:
            value = numeric_eq.group(1)
    # ``enabled, requires cache-ram`` is prose; ``GET, POST, DELETE`` is a CSV default.
    # Do not split ``EPLBConfig(window_size=1000, step_interval=3000)``.
    if not structured:
        prose_tail = re.match(r"^(\S+)\s*,\s+[a-z]", value)
        if prose_tail:
            value = prose_tail.group(1)
    if re.fullmatch(r"""^['"].*['"]$""", value):
        value = value.strip("'\"")
    elif not structured:
        quoted = re.fullmatch(r"""['"]([^'"]*)['"].*""", value)
        if quoted:
            value = quoted.group(1).strip()
        else:
            value = value.rstrip(").;").strip()
    return value or None


def documented_default_tokens(description: str) -> list:
    """Every ``default ...`` clause in a description, in order.

    Each entry is ``(value, rest, explicit)`` where ``explicit`` is true for
    ``default: 4`` / ``default=auto`` (the help text names a value), and false
    for the ``default 4`` prose form, where ``rest`` decides whether the token
    is a value or the first word of a sentence ("default comes from the spec").
    """
    text = str(description or "")
    return [
        (match.group("val"), text[match.end() :], bool(match.group("sep")))
        for match in _DEFAULT_CLAUSE_RE.finditer(text)
    ]


def documented_default_token(description: str) -> Optional[Tuple[str, str, bool]]:
    """The first ``default ...`` clause in a description."""
    tokens = documented_default_tokens(description)
    return tokens[0] if tokens else None


def default_is_value(value: str, rest: str, explicit: bool) -> bool:
    """True when the token after the word "default" is a value, not prose."""
    if explicit:
        return True
    if len(value) > 1 and value[0] in "'\"" and value[-1] in "'\"":
        return True
    # A number or boolean is a value even when a sentence follows it
    # ("default 3 for the published model"); an identifier is only a value when
    # the clause ends there ("default cuda (rocm is an alias)", "default nproc.").
    if _NUMERIC_RE.match(value) or value.lower() in _BOOLEAN_TOKENS:
        return True
    stripped = (rest or "").lstrip()
    if not stripped:
        return True
    return stripped[0] in _CLAUSE_END or stripped[0] in "([<"


def default_from_description(description: str) -> Optional[str]:
    """The literal default a help description states, if it states one."""
    for value, rest, explicit in documented_default_tokens(description):
        if default_is_value(value, rest, explicit):
            return normalize_default_fragment(value)
    return None


def is_documentation_default_artifact(description: str, value: object) -> bool:
    """True when ``value`` only exists because a description was read as a default.

    A model config saved before the prose heuristic was tightened can carry
    ``backend_impl: "preserves"``, copied from "default preserves existing
    dispatch". That value is not settable in the engine ("unsupported impl
    'preserves'"), so the launch path ignores it and says why. A token the same
    description also states as a value is never treated as an artifact.
    """
    tokens = documented_default_tokens(description)
    if not tokens:
        return False
    target = str(value)
    prose = {
        token for token, rest, explicit in tokens if not default_is_value(token, rest, explicit)
    }
    stated = {token for token, rest, explicit in tokens if default_is_value(token, rest, explicit)}
    return target in prose and target not in stated


__all__ = [
    "default_from_description",
    "default_is_value",
    "documented_default_token",
    "documented_default_tokens",
    "is_documentation_default_artifact",
    "looks_like_structured_default",
    "normalize_default_fragment",
]
