"""Heuristic message normalization for leakage-aware evaluation groups.

This module is deliberately separate from classifier preprocessing. Its output
is used only to keep superficial variants of one message template in the same
cross-validation fold.
"""

import re

_ASSET_NAMES = (
    "ada",
    "bitcoin",
    "btc",
    "cardano",
    "doge",
    "dogecoin",
    "eth",
    "ethereum",
    "litecoin",
    "ltc",
    "matic",
    "polygon",
    "sol",
    "solana",
    "usdc",
    "usdt",
    "xrp",
)
_ASSET_PATTERN = re.compile(
    rf"\b(?:{'|'.join(sorted(_ASSET_NAMES, key=len, reverse=True))})\b",
    flags=re.IGNORECASE,
)
_AMOUNT_PATTERN = re.compile(
    r"(?<!\w)(?:[$€£]\s*)?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%)?(?!\w)"
)
_GREETING_PATTERN = re.compile(
    r"^(?:(?:hello(?:\s+team)?|hi(?:\s+there)?|hey|please\s+help|"
    r"quick\s+question|urgent)\b[\s,.:;!?-]*)+",
    flags=re.IGNORECASE,
)
_CLOSING_PATTERN = re.compile(
    r"[\s,.:;!?-]*(?:(?:thanks|thank\s+you|appreciate\s+any\s+help|"
    r"please\s+advise|let\s+me\s+know|this\s+is\s+time\s+sensitive|"
    r"best(?:\s+regards)?|regards)[\s,.:;!?-]*)+$",
    flags=re.IGNORECASE,
)
_PUNCTUATION_PATTERN = re.compile(r"[^a-z0-9<>]+")


def normalize_template(message: str) -> str:
    """Return a conservative grouping key for superficial template variants.

    The heuristic folds case, anchored greetings and closings, known asset
    names, numeric amounts, punctuation, and repeated whitespace. It does not
    perform semantic clustering and must never be passed to the classifier as
    model input.
    """
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Template normalization requires a non-empty string")

    normalized = message.casefold().strip()
    normalized = _GREETING_PATTERN.sub("", normalized)
    normalized = _CLOSING_PATTERN.sub("", normalized)
    normalized = _ASSET_PATTERN.sub(" <asset> ", normalized)
    normalized = _AMOUNT_PATTERN.sub(" <amount> ", normalized)
    normalized = _PUNCTUATION_PATTERN.sub(" ", normalized)
    return " ".join(normalized.split())


def build_template_groups(messages: list[str]) -> list[str]:
    """Create one deterministic grouping key per original message."""
    return [normalize_template(message) for message in messages]
